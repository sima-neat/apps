"""Detect objects in one RTSP, HTTP MJPEG, or USB camera stream with YOLO26, shown in Insight.

The decoded frames feed the model. Insight receives the video and every frame's boxes.
"""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import dataclass, replace
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
INPUT_TYPES = ("rtsp", "http", "usb")
PULL_TIMEOUT_MS = 100


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    labels_path: Path
    input_type: str  # rtsp, http, or usb.
    source: str
    codec: str  # mjpeg for HTTP and USB, which carry only MJPEG.
    # For RTSP, each 0 reads that value from the stream with a probe.
    width: int
    height: int
    fps: int
    rtsp_tcp: bool  # RTSP only.
    rtsp_latency_ms: int  # RTSP only.
    frames: int  # 0 runs until a stop signal.
    min_score: float
    nms_iou: float
    max_detections: int
    insight_host: str
    video_port_base: int
    metadata_port_base: int
    ssl_strict: bool  # HTTP only.


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--validate-config-only", action="store_true", help="check the config and exit"
    )
    return parser.parse_args(argv)


def load_app_config(path: Path) -> AppConfig:
    config = Config.load(path)
    input_type = config.string("input.type")
    if input_type not in INPUT_TYPES:
        raise ValueError("input.type must be rtsp, http, or usb")
    rtsp = input_type == "rtsp"
    # Only the keys of input.type are read, so reject_unread_keys() fails on another type's.
    app_config = AppConfig(
        model_path=path.absolute().parent / config.string("model.path"),
        labels_path=Path(config.string("model.labels")),
        input_type=input_type,
        source=config.string("input.source"),
        codec=config.string("input.codec") if rtsp else "mjpeg",
        width=config.integer("input.width"),
        height=config.integer("input.height"),
        fps=config.integer("input.fps"),
        rtsp_tcp=config.boolean("input.rtsp.tcp") if rtsp else False,
        rtsp_latency_ms=config.integer("input.rtsp.latency_ms") if rtsp else 0,
        frames=config.integer("inference.frames"),
        min_score=config.number("inference.min_score"),
        nms_iou=config.number("inference.nms_iou"),
        max_detections=config.integer("inference.max_detections"),
        insight_host=config.string("output.insight.host"),
        video_port_base=config.integer("output.insight.video_port_base"),
        metadata_port_base=config.integer("output.insight.metadata_port_base"),
        ssl_strict=config.boolean("app.ssl_strict") if input_type == "http" else True,
    )
    config.reject_unread_keys()
    if app_config.codec not in RTSP_CODECS:
        raise ValueError("input.codec must be h264, h265, or mjpeg")
    # Core sends to 127.0.0.1 when the host is empty.
    if not app_config.insight_host:
        raise ValueError("output.insight.host must not be empty")
    if app_config.frames < 0:
        raise ValueError("inference.frames must be 0 or more")
    return app_config


def model_options(cfg: AppConfig, frame_format: pyneat.Format) -> pyneat.ModelOptions:
    """YOLO26 on the decoded frames: NV12, or NV16 from a 4:2:2 USB camera."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16
        if frame_format == pyneat.Format.NV16
        else pyneat.PreprocessColorFormat.NV12
    )
    options.preprocess.input_max_width = cfg.width
    options.preprocess.input_max_height = cfg.height
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = cfg.min_score
    options.nms_iou_threshold = cfg.nms_iou
    options.top_k = cfg.max_detections
    return options


def rtsp_options(cfg: AppConfig) -> pyneat.RtspEncodedInputOptions:
    options = pyneat.RtspEncodedInputOptions()
    options.url = cfg.source
    options.codec = RTSP_CODECS[cfg.codec]
    options.tcp = cfg.rtsp_tcp
    options.latency_ms = cfg.rtsp_latency_ms
    options.source_fps = cfg.fps
    if cfg.codec == "h264":
        options.fallback_h264_width = cfg.width
        options.fallback_h264_height = cfg.height
        options.fallback_h264_fps = cfg.fps
    return options


def http_options(cfg: AppConfig) -> pyneat.HttpMjpegDecodedInputOptions:
    """Receive the HTTP MJPEG stream and decode it to NV12 frames of the configured size."""
    options = pyneat.HttpMjpegDecodedInputOptions()
    options.url = cfg.source
    options.source_fps = cfg.fps
    options.ssl_strict = cfg.ssl_strict
    options.output_caps.enable = True
    options.output_caps.width = cfg.width
    options.output_caps.height = cfg.height
    options.output_caps.fps = cfg.fps
    return options


def usb_camera_options(cfg: AppConfig) -> pyneat.UsbCameraInputOptions:
    """One of the camera's MJPEG modes, exactly: width, height, and frame rate."""
    options = pyneat.UsbCameraInputOptions()
    options.camera_name = cfg.source
    options.width = cfg.width
    options.height = cfg.height
    options.framerate_num = cfg.fps
    return options


def camera_graph(cfg: AppConfig, camera: pyneat.UsbCameraInfo) -> pyneat.Graph:
    graph = pyneat.Graph("usb_camera")
    graph.add(pyneat.nodes.usb_camera_input(usb_camera_options(cfg), camera))
    return graph


def decoder_graph(cfg: AppConfig, frame_format: pyneat.Format) -> pyneat.Graph:
    """Hardware decode of the RTSP stream or the camera's JPEG: NV12, or NV16 for a 4:2:2
    camera."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = DECODE_TYPES[cfg.codec]
    decode.out_format = frame_format
    decode.dec_width = cfg.width
    decode.dec_height = cfg.height
    decode.dec_fps = cfg.fps
    graph = pyneat.Graph("decoder")
    graph.add(pyneat.nodes.sima_decode(decode))
    return graph


def detector_graph(model: pyneat.Model) -> pyneat.Graph:
    """Decoded frames -> model -> "detections" output."""
    graph = pyneat.Graph("detector")
    graph.add(model.graph())
    graph.add(pyneat.nodes.output("detections"))
    return graph


def video_options(cfg: AppConfig) -> pyneat.VideoSenderOptions:
    """RTSP H.264, H.265, and MJPEG and a USB camera's JPEG go to Insight as received; decoded
    HTTP MJPEG frames are encoded as H.264."""
    if cfg.input_type == "http":
        # Core candidate: no HTTP MJPEG input keeps the stream encoded, so its decoded frames are
        # encoded again for Insight.
        encode = pyneat.SimaEncodeOptions()
        encode.fps = cfg.fps
        options = pyneat.VideoSenderOptions.from_raw(encode)
    else:
        # Passthrough(MJPEG) carries RTP/JPEG, at most 2040 pixels per side.
        options = pyneat.VideoSenderOptions.passthrough(RTSP_CODECS[cfg.codec])
    options.host = cfg.insight_host
    options.video_port_base = cfg.video_port_base
    return options


def camera_video_link() -> pyneat.GraphLinkOptions:
    """Send the camera's newest JPEG, so a slow video sender cannot hold back the camera."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def metadata_options(cfg: AppConfig) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.metadata_port_base = cfg.metadata_port_base
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def with_stream_info(cfg: AppConfig) -> AppConfig:
    """The config with an RTSP stream's frame size and rate for each of input.width,
    input.height, and input.fps it leaves at 0. HTTP and USB use the configured values."""
    if cfg.input_type != "rtsp":
        return cfg
    configured = RtspStreamInfo(cfg.width, cfg.height, cfg.fps)
    info = probe_rtsp_stream_info(cfg.source, cfg.rtsp_tcp, configured)
    return replace(cfg, width=info.width, height=info.height, fps=info.fps)


def decode_boxes(sample: pyneat.Sample, cfg: AppConfig) -> list:
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
        bbox, cfg.width, cfg.height, cfg.max_detections, strict=True
    )
    return result.boxes


def send_detections(
    run: pyneat.Run,
    metadata_sender: pyneat.MetadataSender,
    labels: list[str],
    cfg: AppConfig,
    stop: threading.Event,
) -> int:
    """Send every frame's boxes to Insight until inference.frames or a stop signal; return the
    count."""
    processed = 0
    while not stop.is_set() and (cfg.frames == 0 or processed < cfg.frames):
        sample = run.pull("detections", PULL_TIMEOUT_MS)
        if sample is None:
            if run.last_error() or not run.running():
                raise RuntimeError(f"detections output closed: {run.last_error()}")
            continue
        payload = detection_egress.serialize(
            decode_boxes(sample, cfg),
            labels,
            cfg.width,
            cfg.height,
            detection_egress.frame_metadata(sample, 0),
        )
        # A nonblocking send drops the message while the socket buffer is full; the drops are
        # counted in the sender's stats and reported when the run ends. Any other failure raises.
        metadata_sender.send_raw_json(payload)
        processed += 1
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_app_config(args.config)
    if args.validate_config_only:
        print(
            f"Config validated: {args.config} (input={cfg.input_type}, codec={cfg.codec}, "
            f"{cfg.width}x{cfg.height}, fps={cfg.fps})"
        )
        return 0

    labels = load_labels(cfg.labels_path)
    cfg = with_stream_info(cfg)
    camera = (
        pyneat.usb_camera_info(usb_camera_options(cfg)) if cfg.input_type == "usb" else None
    )
    frame_format = camera.decoded_format if camera else pyneat.Format.NV12
    model = pyneat.Model(str(cfg.model_path), model_options(cfg, frame_format))
    detector = detector_graph(model)
    video = pyneat.groups.video_sender(video_options(cfg))

    graph = pyneat.Graph()
    if cfg.input_type == "http":
        frames = pyneat.groups.http_mjpeg_decoded_input(http_options(cfg))
        graph.connect(frames, video)
    elif cfg.input_type == "usb":
        source = camera_graph(cfg, camera)
        frames = decoder_graph(cfg, frame_format)
        graph.connect(source, frames)
        graph.connect(source, video, camera_video_link())
    else:
        source = pyneat.groups.rtsp_encoded_input(rtsp_options(cfg))
        frames = decoder_graph(cfg, frame_format)
        graph.connect(source, frames)
        # A plain link lets Core tee the encoded stream to the decoder and the video sender.
        graph.connect(source, video)
    graph.connect(frames, detector)

    metadata_sender = pyneat.MetadataSender(metadata_options(cfg))
    stop = stop_on_signals()
    run = graph.build(run_options())
    print(
        f"source={cfg.source} type={cfg.input_type} codec={cfg.codec} "
        f"stream={cfg.width}x{cfg.height}@{cfg.fps} insight={cfg.insight_host} "
        f"video={cfg.video_port_base} metadata={cfg.metadata_port_base}"
    )
    try:
        processed = send_detections(run, metadata_sender, labels, cfg, stop)
    finally:
        run.close()

    print(f"processed={processed} video_sender={cfg.insight_host}:{cfg.video_port_base}")
    if metadata_sender.stats().send_failures:
        print(f"{metadata_sender.stats().send_failures} metadata messages dropped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
