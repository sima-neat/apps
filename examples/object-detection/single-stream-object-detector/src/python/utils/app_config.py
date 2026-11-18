"""This example's settings: the keys it reads from config.yaml, their defaults and their rules.

It needs only PyYAML, so --validate-config-only runs without pyneat or OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import bool_or, float_or, int_or, load_config, section, string_or

DEFAULT_LABELS = Path(__file__).resolve().parents[2] / "common" / "coco_label.txt"


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    source_url: str
    source_type: str = "rtsp"
    source_codec: str = "h264"
    latency_ms: int = 200
    tcp: bool = True
    source_fps: int = 0
    ssl_strict: bool = True
    frames: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    save_dir: str = ""
    save_every: int = 0
    # The key that names the source: source.<type>.url, or source.usb.camera.
    source_key: str = "source.rtsp.url"
    # source.type=usb: the camera's stable ID (usb_camera_ids()) and one of its MJPEG modes.
    camera: str = ""
    source_width: int = 1920
    source_height: int = 1080


def parse_source_type(value: str) -> str:
    lowered = value.lower()
    if lowered in {"rtsp", "http", "https", "usb"}:
        return "http" if lowered == "https" else lowered
    raise ValueError("source.type must be rtsp, http, or usb")


def parse_source_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    if lowered in {"mjpeg", "jpeg"}:
        return "mjpeg"
    raise ValueError("source.rtsp.codec must be h264, h265, or mjpeg")


def validate_config(cfg: AppConfig) -> None:
    if cfg.source_type == "usb":
        if not cfg.camera:
            raise ValueError("source.usb.camera must be set")
        if cfg.source_width <= 0 or cfg.source_height <= 0:
            raise ValueError("source.usb.width and source.usb.height must be > 0")
        if cfg.source_fps <= 0:
            raise ValueError("source.usb.fps must be > 0")
        # The camera's JPEG goes to Insight as RTP/JPEG (RFC 2435): 8-pixel blocks, 2040 at most.
        sides = (cfg.source_width, cfg.source_height)
        if not all(0 < side <= 2040 and side % 8 == 0 for side in sides):
            raise ValueError(
                "source.usb.width and source.usb.height must be multiples of 8 and at most 2040 "
                "(RTP/JPEG)"
            )
    elif not cfg.source_url:
        raise ValueError(f"{cfg.source_key} must be set")
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    # Path("") is ".", so an empty value in the file arrives here as ".".
    if str(cfg.labels_path) in ("", "."):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError(f"source.{cfg.source_type}.latency_ms must be >= 0")
    if cfg.source_fps < 0:
        raise ValueError(f"source.{cfg.source_type}.fps must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_app_config(config_path: Path) -> AppConfig:
    """Read and validate config.yaml. Raises ValueError naming the key at fault."""
    raw = load_config(config_path)

    model = section(raw, "model")
    source = section(raw, "source")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")

    labels_path = string_or(model, "labels", str(DEFAULT_LABELS))
    # Only the section source.type names is read; the others may hold placeholders.
    # The key that names the source is kept so --validate-config-only can report
    # it without echoing the URL, which can carry credentials.
    source_type = parse_source_type(string_or(source, "type", "rtsp"))
    source_key = f"source.{source_type}.{'camera' if source_type == 'usb' else 'url'}"
    source = section(source, source_type)
    # HTTP streams and USB cameras are MJPEG; only RTSP names a codec.
    source_codec = (
        parse_source_codec(string_or(source, "codec", "h264")) if source_type == "rtsp" else "mjpeg"
    )

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(labels_path),
        source_url=string_or(source, "url"),
        source_key=source_key,
        source_type=source_type,
        camera=string_or(source, "camera"),
        source_width=int_or(source, "width", 1920),
        source_height=int_or(source, "height", 1080),
        source_codec=source_codec,
        latency_ms=int_or(source, "latency_ms", 200),
        tcp=bool_or(source, "tcp", True),
        source_fps=int_or(source, "fps", 0),
        ssl_strict=bool_or(source, "ssl_strict", True),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        save_dir=string_or(output, "save_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg
