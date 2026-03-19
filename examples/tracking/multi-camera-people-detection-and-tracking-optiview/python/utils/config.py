"""YAML-backed configuration for the Python multi-camera people tracking example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    model: str
    rtsp_urls: list[str]
    output_dir: str | None
    frames: int
    optiview_host: str
    optiview_video_port_base: int
    optiview_json_port_base: int
    fps: int
    bitrate_kbps: int
    save_every: int
    profile: bool
    person_class_id: int
    detection_threshold: float | None
    nms_iou_threshold: float | None
    top_k: int | None
    tracker_iou_threshold: float
    tracker_max_missing: int
    latency_ms: int
    tcp: bool


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_string(mapping: dict[str, Any], key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_float_or_none(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric or null")
    return float(value)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")

    input_cfg = _mapping(raw.get("input"), "input")
    inference = _mapping(raw.get("inference"), "inference")
    tracking = _mapping(raw.get("tracking"), "tracking")
    output = _mapping(raw.get("output"), "output")
    optiview = _mapping(output.get("optiview"), "output.optiview")

    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    rtsp_urls: list[str] = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, str) or not stream.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(stream)

    cfg = AppConfig(
        model=model,
        rtsp_urls=rtsp_urls,
        output_dir=output.get("debug_dir"),
        frames=_optional_int(inference, "frames", 0),
        optiview_host=_required_string(optiview, "host", "output.optiview"),
        optiview_video_port_base=_optional_int(optiview, "video_port_base", 9000),
        optiview_json_port_base=_optional_int(optiview, "json_port_base", 9100),
        fps=_optional_int(inference, "fps", 0),
        bitrate_kbps=_optional_int(inference, "bitrate_kbps", 2500),
        save_every=_optional_int(output, "save_every", 0),
        profile=bool(inference.get("profile", False)),
        person_class_id=_optional_int(inference, "person_class_id", 0),
        detection_threshold=_optional_float_or_none(inference, "detection_threshold"),
        nms_iou_threshold=_optional_float_or_none(inference, "nms_iou_threshold"),
        top_k=inference.get("top_k"),
        tracker_iou_threshold=float(tracking.get("iou_threshold", 0.3)),
        tracker_max_missing=_optional_int(tracking, "max_missing_frames", 15),
        latency_ms=_optional_int(input_cfg, "latency_ms", 200),
        tcp=bool(input_cfg.get("tcp", False)),
    )

    if cfg.optiview_video_port_base <= 0:
        raise ValueError("optiview.video_port_base must be > 0")
    if cfg.optiview_json_port_base <= 0:
        raise ValueError("optiview.json_port_base must be > 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if cfg.bitrate_kbps <= 0:
        raise ValueError("inference.bitrate_kbps must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.tracker_max_missing < 0:
        raise ValueError("tracking.max_missing_frames must be >= 0")
    if not 0.0 <= cfg.tracker_iou_threshold <= 1.0:
        raise ValueError("tracking.iou_threshold must be between 0 and 1")
    if cfg.detection_threshold is not None and not 0.0 <= cfg.detection_threshold <= 1.0:
        raise ValueError("inference.detection_threshold must be between 0 and 1")
    if cfg.nms_iou_threshold is not None and not 0.0 <= cfg.nms_iou_threshold <= 1.0:
        raise ValueError("inference.nms_iou_threshold must be between 0 and 1")
    if cfg.top_k is not None:
        if not isinstance(cfg.top_k, int):
            raise ValueError("inference.top_k must be an integer or null")
        if cfg.top_k <= 0:
            raise ValueError("inference.top_k must be > 0")
    if cfg.output_dir is not None and not isinstance(cfg.output_dir, str):
        raise ValueError("output.debug_dir must be a string or null")

    return cfg
