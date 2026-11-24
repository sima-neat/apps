"""Detect objects in a MIPI or USB camera stream with YOLO26, shown in Insight.

The camera's frames feed the model. Insight receives the video and every frame's boxes.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import pyneat

# The helpers the examples share live in <apps root>/support/python.
for apps_root in Path(__file__).resolve().parents:
    if (apps_root / "support" / "python" / "sima_examples").is_dir():
        sys.path.insert(0, str(apps_root / "support" / "python"))
        break
else:
    raise SystemExit("support/python not found above this example; reinstall the Apps package")

from sima_examples import detection_egress
from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import RtspStreamInfo
from sima_examples.stop_signals import stop_on_signals

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

INPUT_TYPES = ("mipi", "usb")
# A USB camera carries MJPEG, which the hardware decoder turns into frames.
USB_DECODE_TYPE = pyneat.SimaDecodeType.MJPEG
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def require_strict_zero_copy(backend: str) -> None:
    """Fail when the camera stage copies frames instead of forwarding DMA-BUFs."""
    if "zero_copy=strict" not in backend:
        raise RuntimeError(
            "camera input is not forwarding DMA-BUFs; check the libcamera package and the "
            f"camera bridge: {backend}"
        )


def model_options(
    config: Config, stream: RtspStreamInfo, frame_format: pyneat.Format, max_detections: int
) -> pyneat.ModelOptions:
    """YOLO26 on the camera's frames: NV12, or NV16 from a 4:2:2 USB camera."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16
        if frame_format == pyneat.Format.NV16
        else pyneat.PreprocessColorFormat.NV12
    )
    options.preprocess.input_max_width = stream.width
    options.preprocess.input_max_height = stream.height
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = config.number("inference.min_score")
    options.nms_iou_threshold = config.number("inference.nms_iou")
    options.top_k = max_detections
    return options


def mipi_camera_options(config: Config, stream: RtspStreamInfo) -> pyneat.CameraInputOptions:
    """The camera mode to open, exactly: size, rate, and the buffer the model reads."""
    options = pyneat.CameraInputOptions()
    options.width = stream.width
    options.height = stream.height
    options.framerate_num = stream.fps
    options.buffer_name = config.string("input.source")
    return options


def usb_camera_options(config: Config, stream: RtspStreamInfo) -> pyneat.UsbCameraInputOptions:
    """One of the camera's MJPEG modes, exactly: width, height, and frame rate."""
    options = pyneat.UsbCameraInputOptions()
    options.camera_name = config.string("input.source")
    options.width = stream.width
    options.height = stream.height
    options.framerate_num = stream.fps
    return options


def mipi_camera_graph(options: pyneat.CameraInputOptions) -> pyneat.Graph:
    graph = pyneat.Graph("mipi_camera")
    graph.add(pyneat.nodes.camera_input(options))
    # A composed graph describes its execution plan rather than the camera pipeline, so check
    # the camera stage before connecting any consumer.
    require_strict_zero_copy(graph.describe_backend(False))
    return graph


def usb_camera_graph(
    options: pyneat.UsbCameraInputOptions, camera: pyneat.UsbCameraInfo
) -> pyneat.Graph:
    graph = pyneat.Graph("usb_camera")
    graph.add(pyneat.nodes.usb_camera_input(options, camera))
    return graph


def decoder_graph(stream: RtspStreamInfo, frame_format: pyneat.Format) -> pyneat.Graph:
    """Hardware decode of the USB camera's JPEG: NV12, or NV16 for a 4:2:2 camera."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = USB_DECODE_TYPE
    decode.out_format = frame_format
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    return graph


def detector_graph(model: pyneat.Model, route: pyneat.ModelRouteOptions | None = None) -> pyneat.Graph:
    """Frames -> model -> "detections" output."""
    graph = model.graph(route) if route else model.graph()
    graph.add(pyneat.nodes.output("detections"))
    return graph


def mipi_model_route(config: Config) -> pyneat.ModelRouteOptions:
    """Read the camera's buffer directly, which is how a MIPI frame reaches the model."""
    route = pyneat.ModelRouteOptions()
    route.include_input = False
    route.include_output = False
    route.upstream_name = "images"
    route.buffer_name = config.string("input.source")
    route.advanced_execution.preprocess_target = "EV74"
    route.advanced_execution.postprocess_target = "EV74"
    return route


def mipi_video_input_options(stream: RtspStreamInfo) -> pyneat.InputOptions:
    """The raw frames the video sender encodes for Insight."""
    options = pyneat.InputOptions()
    options.payload_type = pyneat.PayloadType.Image
    options.format = pyneat.Format.NV12
    options.width = stream.width
    options.height = stream.height
    options.fps_n = stream.fps
    options.memory_policy = pyneat.InputMemoryPolicy.Ev74
    return options


def video_options(
    config: Config, input_type: str, stream: RtspStreamInfo
) -> pyneat.VideoSenderOptions:
    """A MIPI camera delivers raw frames, which are encoded here; a USB camera's JPEG is sent
    as it arrives."""
    if input_type == "mipi":
        encode = pyneat.SimaEncodeOptions()
        encode.fps = stream.fps
        options = pyneat.VideoSenderOptions.from_raw(encode)
    else:
        # Passthrough(MJPEG) carries RTP/JPEG, at most 2040 pixels per side.
        options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
    options.host = config.string("output.insight.host")
    options.video_port_base = config.integer("output.insight.video_port_base")
    return options


def realtime_link() -> pyneat.GraphLinkOptions:
    """Send the newest frame, so a slow consumer cannot hold back the camera."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def metadata_options(config: Config) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = config.string("output.insight.host")
    options.metadata_port_base = config.integer("output.insight.metadata_port_base")
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def decode_boxes(sample: pyneat.Sample, stream: RtspStreamInfo, max_detections: int) -> list:
    """The boxes in a detections sample, whose first tensor is the BBOX payload."""
    # Core candidate: decode_bbox_tensor takes a Tensor, so each example picks the BBOX tensor
    # out of the Sample itself.
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        bbox = sample.tensors[0]
    elif sample.kind == pyneat.SampleKind.Tensor:
        bbox = sample.tensor
    else:
        raise RuntimeError(f"expected a BBOX tensor, got {sample.kind}")
    result = pyneat.detections.decode_bbox_tensor(
        bbox, stream.width, stream.height, max_detections, strict=True
    )
    return result.boxes


def send_detections(
    run: pyneat.Run,
    metadata_sender: pyneat.MetadataSender,
    labels: list[str],
    stream: RtspStreamInfo,
    max_detections: int,
    frames: int,
    stop: threading.Event,
) -> int:
    """Send every frame's boxes to Insight until `frames` (0: no limit) or a stop signal; return
    the count."""
    processed = 0
    while not stop.is_set() and (frames == 0 or processed < frames):
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue
        payload = detection_egress.serialize(
            decode_boxes(sample, stream, max_detections),
            labels,
            stream.width,
            stream.height,
            detection_egress.frame_metadata(sample, 0),
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_sender.send_raw_json(payload)
        processed += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    input_type = config.string("input.type")
    if input_type not in INPUT_TYPES:
        raise ValueError(f"input.type must be one of {', '.join(INPUT_TYPES)}: {input_type}")
    # A camera's size and rate are the mode it is opened in, so both come from the config.
    stream = RtspStreamInfo(
        config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")
    )
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    labels = load_labels(Path(config.string("model.labels")))
    model_path = args.config.absolute().parent / config.string("model.path")

    graph = pyneat.Graph()
    video = pyneat.groups.video_sender(video_options(config, input_type, stream))
    if input_type == "mipi":
        model = pyneat.Model(
            str(model_path), model_options(config, stream, pyneat.Format.NV12, max_detections)
        )
        source = mipi_camera_graph(mipi_camera_options(config, stream))
        detector = detector_graph(model, mipi_model_route(config))
        video_frames = pyneat.Graph("insight_video")
        video_frames.add(pyneat.nodes.input("insight_video", mipi_video_input_options(stream)))
        video_frames.add(video)
        branch = pyneat.graphs.branch(config.string("input.source"), ["images", "insight_video"])
        graph.connect(source, branch)
        graph.connect(branch, detector, realtime_link())
        graph.connect(branch, video_frames, realtime_link())
    else:
        camera_options = usb_camera_options(config, stream)
        camera = pyneat.usb_camera_info(camera_options)
        model = pyneat.Model(
            str(model_path),
            model_options(config, stream, camera.decoded_format, max_detections),
        )
        source = usb_camera_graph(camera_options, camera)
        frames = decoder_graph(stream, camera.decoded_format)
        graph.connect(source, frames)
        graph.connect(source, video, realtime_link())
        graph.connect(frames, detector_graph(model))

    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    stop = stop_on_signals()
    run = graph.build(run_options())
    host = config.string("output.insight.host")
    print(
        f"source={config.string('input.source')} type={input_type} "
        f"stream={stream.width}x{stream.height}@{stream.fps} insight={host} "
        f"video={config.integer('output.insight.video_port_base')} "
        f"metadata={config.integer('output.insight.metadata_port_base')}"
    )
    try:
        processed = send_detections(
            run,
            metadata_sender,
            labels,
            stream,
            max_detections,
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()

    print(f"frames={processed}")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
