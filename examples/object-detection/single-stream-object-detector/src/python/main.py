"""Detect objects in one RTSP, HTTP MJPEG, or USB camera stream with YOLO26 and show them in Insight.

The decoded frames feed the model, and Insight receives the video and every frame's boxes.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pyneat

# sima_examples is a link to the helpers the examples share, in <apps root>/support/python.
from sima_examples import detection_egress
from sima_examples.config import Config
from sima_examples.labels import load_labels
from sima_examples.rtsp_probe import probe_rtsp_stream_info

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

RTSP_CODECS = {
    "h264": pyneat.RtspCodec.H264,
    "h265": pyneat.RtspCodec.H265,
    "mjpeg": pyneat.RtspCodec.MJPEG,
}
SOURCE_TYPES = ("rtsp", "http", "usb")
PULL_TIMEOUT_MS = 100
# RTP/JPEG (RFC 2435) carries a USB camera's JPEG to Insight in 8-pixel blocks, 2040 at most.
RTP_JPEG_MAX_SIDE = 2040


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    labels_path: Path
    source_type: str  # input.type: rtsp, http, or usb.
    source: str
    codec: str  # mjpeg for HTTP and USB, which carry only MJPEG.
    width: int
    height: int
    fps: int  # 0 reads an RTSP stream's frame rate with a probe.
    rtsp_tcp: bool  # RTSP only.
    rtsp_latency_ms: int  # RTSP only.
    frames: int  # 0 runs until Ctrl+C.
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
    source_type = config.string("input.type")
    if source_type not in SOURCE_TYPES:
        raise ValueError("input.type must be rtsp, http, or usb")
    rtsp = source_type == "rtsp"
    # Only the keys of input.type are read, so reject_unread_keys() fails on another type's.
    app_config = AppConfig(
        model_path=path.resolve().parent
        / Path(config.string("model.path")).expanduser(),
        labels_path=Path(config.string("model.labels")).expanduser(),
        source_type=source_type,
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
        ssl_strict=config.boolean("app.ssl_strict") if source_type == "http" else True,
    )
    config.reject_unread_keys()
    if app_config.codec not in RTSP_CODECS:
        raise ValueError("input.codec must be h264, h265, or mjpeg")
    if not rtsp and app_config.fps <= 0:
        raise ValueError("input.fps must be above 0 for an HTTP or USB source")
    if app_config.source_type == "usb" and not all(
        0 < side <= RTP_JPEG_MAX_SIDE and side % 8 == 0
        for side in (app_config.width, app_config.height)
    ):
        raise ValueError(
            "input.width and input.height of a USB camera must be multiples of 8 "
            f"and at most {RTP_JPEG_MAX_SIDE} (RTP/JPEG)"
        )
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


def rtsp_options(cfg: AppConfig, fps: int) -> pyneat.RtspDecodedInputOptions:
    """Receive the RTSP stream and decode it to NV12 frames of the configured size."""
    options = pyneat.RtspDecodedInputOptions()
    options.url = cfg.source
    options.codec = RTSP_CODECS[cfg.codec]
    options.tcp = cfg.rtsp_tcp
    options.latency_ms = cfg.rtsp_latency_ms
    options.source_fps = fps
    if cfg.codec == "h264":
        options.fallback_h264_width = cfg.width
        options.fallback_h264_height = cfg.height
    else:
        options.dec_width = cfg.width
        options.dec_height = cfg.height
    options.output_caps.enable = True
    options.output_caps.width = cfg.width
    options.output_caps.height = cfg.height
    options.output_caps.fps = fps
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


def usb_decoder_graph(cfg: AppConfig, frame_format: pyneat.Format) -> pyneat.Graph:
    """Hardware decode of the camera's JPEG in its own chroma layout, NV12 or NV16."""
    decode = pyneat.SimaDecodeOptions()
    decode.type = pyneat.SimaDecodeType.MJPEG
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
    graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    return graph


def video_options(cfg: AppConfig, fps: int) -> pyneat.VideoSenderOptions:
    """Re-encode decoded RTSP and HTTP frames as H.264; send a USB camera's JPEG unchanged."""
    if cfg.source_type == "usb":
        options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
    else:
        encode = pyneat.SimaEncodeOptions()
        encode.fps = fps
        options = pyneat.VideoSenderOptions.from_raw(encode)
        options.encoder.bitrate_kbps = 1000
    options.host = cfg.insight_host
    options.video_port_base = cfg.video_port_base
    return options


def video_link() -> pyneat.GraphLinkOptions:
    """Send the newest frames and bound those the video sender holds, so it cannot starve the
    decoder or the camera."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.max_inflight_per_stream = 2
    link.max_inflight_total = 2
    return link


def metadata_options(cfg: AppConfig) -> pyneat.MetadataSenderOptions:
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.metadata_port_base = cfg.metadata_port_base
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = 3
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    return options


def stream_fps(cfg: AppConfig) -> int:
    """The stream's frame rate: the configured one, or a probe of an RTSP stream when input.fps is 0."""
    if cfg.fps > 0:
        return cfg.fps
    return probe_rtsp_stream_info(cfg.source, cfg.rtsp_tcp).fps


def decode_boxes(sample: pyneat.Sample, cfg: AppConfig) -> list:
    """The boxes in a detections sample, whose first tensor is the BBOX payload."""
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


def frame_metadata(sample: pyneat.Sample) -> detection_egress.FrameMetadata:
    return detection_egress.FrameMetadata(
        stream_index=0,
        stream_id=sample.stream_id,
        frame_id=sample.frame_id,
        pts_ns=sample.pts_ns,
        dts_ns=sample.dts_ns,
        duration_ns=sample.duration_ns,
        input_seq=sample.input_seq,
        orig_input_seq=sample.orig_input_seq,
        rtp_timestamp=(
            detection_egress.rtp_timestamp_from_pts_ns(sample.pts_ns)
            if sample.pts_ns >= 0
            else None
        ),
    )


def send_detections(
    run: pyneat.Run,
    metadata_sender: pyneat.MetadataSender,
    labels: list[str],
    cfg: AppConfig,
) -> int:
    """Send every frame's boxes to Insight until inference.frames or Ctrl+C; return the count."""
    processed = 0
    try:
        while cfg.frames == 0 or processed < cfg.frames:
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
                frame_metadata(sample),
            )
            # A nonblocking send drops the message while the socket buffer is full; the drops
            # are counted in the sender's stats and reported when the run ends. Any other
            # failure raises.
            metadata_sender.send_raw_json(payload)
            processed += 1
    except KeyboardInterrupt:
        pass  # Ctrl+C stops the run.
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_app_config(args.config)
    if args.validate_config_only:
        print(f"Config validated: {args.config} (source={cfg.source_type})")
        return 0

    labels = load_labels(cfg.labels_path)
    fps = stream_fps(cfg)
    camera = (
        pyneat.usb_camera_info(usb_camera_options(cfg))
        if cfg.source_type == "usb"
        else None
    )
    frame_format = camera.decoded_format if camera else pyneat.Format.NV12
    model = pyneat.Model(str(cfg.model_path), model_options(cfg, frame_format))
    detector = detector_graph(model)
    video = pyneat.groups.video_sender(video_options(cfg, fps))

    graph = pyneat.Graph()
    if camera:
        source = camera_graph(cfg, camera)
        frames = usb_decoder_graph(cfg, frame_format)
        graph.connect(source, video, video_link())
        graph.connect(source, frames)
    else:
        if cfg.source_type == "rtsp":
            frames = pyneat.groups.rtsp_decoded_input(rtsp_options(cfg, fps))
        else:
            frames = pyneat.groups.http_mjpeg_decoded_input(http_options(cfg))
        graph.connect(frames, video, video_link())
    graph.connect(frames, detector)

    metadata_sender = pyneat.MetadataSender(metadata_options(cfg))
    run = graph.build(run_options())
    print(
        f"source={cfg.source} type={cfg.source_type} codec={cfg.codec} "
        f"stream={cfg.width}x{cfg.height}@{fps} insight={cfg.insight_host} "
        f"video={cfg.video_port_base} metadata={cfg.metadata_port_base}"
    )
    try:
        processed = send_detections(run, metadata_sender, labels, cfg)
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
