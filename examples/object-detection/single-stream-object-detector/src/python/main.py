"""Detect objects in one RTSP, HTTP MJPEG, or USB camera stream with YOLO26, shown in Insight.

The decoded frames feed the model. Insight receives the video and every frame's boxes.
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
from sima_examples.rtsp_probe import RtspStreamInfo, probe_rtsp_stream_info
from sima_examples.stop_signals import stop_on_signals

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

RTSP_CODECS = {
    "h264": pyneat.RtspCodec.H264,
    "h265": pyneat.RtspCodec.H265,
    "mjpeg": pyneat.RtspCodec.MJPEG,
}
DECODE_TYPES = {
    "h264": pyneat.SimaDecodeType.H264,
    "h265": pyneat.SimaDecodeType.H265,
    "mjpeg": pyneat.SimaDecodeType.MJPEG,
}
# The codec of each input type: HTTP and USB carry only MJPEG, and RTSP names it in input.codec.
INPUT_CODECS = {"rtsp": "", "http": "mjpeg", "usb": "mjpeg"}
PULL_TIMEOUT_MS = 100


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(
    config: Config, stream: RtspStreamInfo, frame_format: pyneat.Format
) -> pyneat.ModelOptions:
    """YOLO26 on the decoded frames: NV12, or NV16 from a 4:2:2 USB camera."""
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
    options.top_k = config.integer("inference.max_detections")
    return options


def rtsp_options(config: Config, stream: RtspStreamInfo) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = config.string("input.source")
    options.codec = RTSP_CODECS[config.string("input.codec")]
    options.tcp = config.boolean("input.rtsp.tcp")
    options.latency_ms = config.integer("input.rtsp.latency_ms")
    # With input.fps at 0, Core also takes the caps from the stream: above 60 fps, caps set from
    # the probed values alone failed to negotiate in some runs.
    options.auto_caps_from_stream = config.integer("input.fps") == 0
    options.source_fps = stream.fps
    if options.codec == pyneat.RtspCodec.H264:
        options.fallback_h264_width = stream.width
        options.fallback_h264_height = stream.height
        options.fallback_h264_fps = stream.fps
    return options


def http_options(config: Config, stream: RtspStreamInfo) -> pyneat.HttpMjpegDecodedInputOptions:
    """Receive the HTTP MJPEG stream and decode it to NV12 frames of the configured size."""
    options = pyneat.HttpMjpegDecodedInputOptions()
    options.url = config.string("input.source")
    options.source_fps = stream.fps
    options.ssl_strict = config.boolean("app.ssl_strict")
    options.output_caps.enable = True
    options.output_caps.width = stream.width
    options.output_caps.height = stream.height
    options.output_caps.fps = stream.fps
    return options


def usb_camera_options(config: Config, stream: RtspStreamInfo) -> pyneat.UsbCameraInputOptions:
    """One of the camera's MJPEG modes, exactly: width, height, and frame rate."""
    options = pyneat.UsbCameraInputOptions()
    options.camera_name = config.string("input.source")
    options.width = stream.width
    options.height = stream.height
    options.framerate_num = stream.fps
    return options


def camera_graph(
    options: pyneat.UsbCameraInputOptions, camera: pyneat.UsbCameraInfo
) -> pyneat.Graph:
    graph = pyneat.Graph("usb_camera")
    graph.add(pyneat.nodes.usb_camera_input(options, camera))
    return graph


def decoder_graph(codec: str, stream: RtspStreamInfo, frame_format: pyneat.Format) -> pyneat.Graph:
    """Hardware decode of the RTSP stream or the camera's JPEG: NV12, or NV16 for a 4:2:2
    camera."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = DECODE_TYPES[codec]
    decode.out_format = frame_format
    decode.dec_width = stream.width
    decode.dec_height = stream.height
    decode.dec_fps = stream.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    return graph


def detector_graph(model: pyneat.Model) -> pyneat.Graph:
    """Decoded frames -> model -> "detections" output."""
    graph = pyneat.Graph("detector")
    graph.add(model.graph())
    graph.add(pyneat.nodes.output("detections"))
    return graph


def video_options(
    config: Config, input_type: str, codec: str, stream: RtspStreamInfo
) -> pyneat.VideoSenderOptions:
    """RTSP H.264, H.265, and MJPEG and a USB camera's JPEG go to Insight as received; decoded
    HTTP MJPEG frames are encoded as H.264."""
    if input_type == "http":
        # Core candidate: no HTTP MJPEG input keeps the stream encoded, so its decoded frames are
        # encoded again for Insight.
        encode = pyneat.SimaEncodeOptions()
        encode.fps = stream.fps
        options = pyneat.VideoSenderOptions.from_raw(encode)
    else:
        # Passthrough(MJPEG) carries RTP/JPEG, at most 2040 pixels per side.
        options = pyneat.VideoSenderOptions.passthrough(RTSP_CODECS[codec])
    options.host = config.string("output.insight.host")
    options.video_port_base = config.integer("output.insight.video_port_base")
    return options


def camera_video_link() -> pyneat.GraphLinkOptions:
    """Send the camera's newest JPEG, so a slow video sender cannot hold back the camera."""
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
    codec = INPUT_CODECS[input_type] or config.string("input.codec")
    configured = RtspStreamInfo(
        config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")
    )
    # An RTSP stream's size and rate come from the probe for each value the config leaves at 0;
    # HTTP and USB use the configured values.
    stream = (
        probe_rtsp_stream_info(
            config.string("input.source"), config.boolean("input.rtsp.tcp"), configured
        )
        if input_type == "rtsp"
        else configured
    )
    labels = load_labels(Path(config.string("model.labels")))
    camera_options = usb_camera_options(config, stream) if input_type == "usb" else None
    camera = pyneat.usb_camera_info(camera_options) if camera_options else None
    frame_format = camera.decoded_format if camera else pyneat.Format.NV12
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config, stream, frame_format))
    detector = detector_graph(model)
    video = pyneat.groups.video_sender(video_options(config, input_type, codec, stream))

    graph = pyneat.Graph()
    if input_type == "http":
        frames = pyneat.groups.http_mjpeg_decoded_input(http_options(config, stream))
        graph.connect(frames, video)
    elif input_type == "usb":
        source = camera_graph(camera_options, camera)
        frames = decoder_graph(codec, stream, frame_format)
        graph.connect(source, frames)
        graph.connect(source, video, camera_video_link())
    else:
        source = pyneat.groups.rtsp_encoded_input(rtsp_options(config, stream))
        frames = decoder_graph(codec, stream, frame_format)
        graph.connect(source, frames)
        # A plain link lets Core tee the encoded stream to the decoder and the video sender.
        graph.connect(source, video)
    graph.connect(frames, detector)

    metadata_sender = pyneat.MetadataSender(metadata_options(config))
    stop = stop_on_signals()
    run = graph.build(run_options())
    host = config.string("output.insight.host")
    video_port = config.integer("output.insight.video_port_base")
    print(
        f"source={config.string('input.source')} type={input_type} codec={codec} "
        f"stream={stream.width}x{stream.height}@{stream.fps} insight={host} "
        f"video={video_port} metadata={config.integer('output.insight.metadata_port_base')}"
    )
    try:
        processed = send_detections(
            run,
            metadata_sender,
            labels,
            stream,
            config.integer("inference.max_detections"),
            config.integer("inference.frames"),
            stop,
        )
    finally:
        run.close()

    print(f"processed={processed} video_sender={host}:{video_port}")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
