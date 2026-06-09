"""YAML-backed configuration for the Python multistream object detection example."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class VideoMode(Enum):
    CLEAN = "clean"
    ANNOTATED = "annotated"


@dataclass(frozen=True)
class ModelConfig:
    path: str


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    rtsp_urls: list[str]
    tcp: bool = False
    latency_ms: int = 100
    worker_count: int = 1
    mailbox_depth: int = 1
    profile: bool = False
    frames: int = 0
    fps: int = 0
    min_score: float = 0.25
    nms_iou: float = 0.45
    max_detections: int = 100
    insight_host: str = ""
    insight_video_port_base: int = 9000
    insight_metadata_port_base: int = 9100
    insight_metadata_offset_ms: float = 0.0
    video_enabled: bool = True
    video_mode: VideoMode = VideoMode.CLEAN
    output_dir: str | None = None
    save_every: int = 0


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
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _optional_float(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_bool(mapping: dict[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return bool(value)


def _parse_video_mode(value: str | None) -> VideoMode:
    lowered = "clean" if value is None else str(value).strip().lower()
    if lowered == "clean":
        return VideoMode.CLEAN
    if lowered == "annotated":
        return VideoMode.ANNOTATED
    raise ValueError("output.video_mode must be one of [clean, annotated]")


def metadata_output_enabled(cfg: AppConfig) -> bool:
    return (not cfg.video_enabled) or cfg.video_mode is VideoMode.CLEAN


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = _mapping(raw.get("model"), "model")
    input_cfg = _mapping(raw.get("input"), "input")
    runtime = _mapping(raw.get("runtime"), "runtime")
    inference = _mapping(raw.get("inference"), "inference")
    output = _mapping(raw.get("output"), "output")
    insight = _mapping(output.get("insight"), "output.insight")

    streams_raw = raw.get("streams")
    if not isinstance(streams_raw, list) or not streams_raw:
        raise ValueError("streams must be a non-empty list")
    if "family" in model:
        raise ValueError(
            "model.family is no longer supported; this example infers YOLOv8 from model.path"
        )
    rtsp_urls: list[str] = []
    for index, stream in enumerate(streams_raw):
        if not isinstance(stream, str) or not stream.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(stream)

    cfg = AppConfig(
        model=ModelConfig(
            path=_required_string(model, "path", "model"),
        ),
        rtsp_urls=rtsp_urls,
        tcp=_optional_bool(input_cfg, "tcp", False),
        latency_ms=_optional_int(input_cfg, "latency_ms", 100),
        worker_count=_optional_int(runtime, "worker_count", 1),
        mailbox_depth=_optional_int(runtime, "mailbox_depth", 1),
        profile=_optional_bool(runtime, "profile", False),
        frames=_optional_int(inference, "frames", 0),
        fps=_optional_int(inference, "fps", 0),
        min_score=_optional_float(inference, "min_score", 0.25),
        nms_iou=_optional_float(inference, "nms_iou", 0.45),
        max_detections=_optional_int(inference, "max_detections", 100),
        insight_host=_required_string(insight, "host", "output.insight"),
        insight_video_port_base=_optional_int(insight, "video_port_base", 9000),
        insight_metadata_port_base=_optional_int(insight, "metadata_port_base", 9100),
        insight_metadata_offset_ms=_optional_float(insight, "metadata_offset_ms", 0.0),
        video_enabled=_optional_bool(output, "video_enabled", True),
        video_mode=_parse_video_mode(output.get("video_mode")),
        output_dir=output.get("debug_dir"),
        save_every=_optional_int(output, "save_every", 0),
    )

    if cfg.worker_count <= 0:
        raise ValueError("runtime.worker_count must be > 0")
    if cfg.mailbox_depth <= 0:
        raise ValueError("runtime.mailbox_depth must be > 0")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.insight_video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.insight_metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if cfg.output_dir is not None and not isinstance(cfg.output_dir, str):
        raise ValueError("output.debug_dir must be a string or null")

    return cfg
