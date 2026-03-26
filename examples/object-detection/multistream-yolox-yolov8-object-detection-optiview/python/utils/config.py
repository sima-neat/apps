"""YAML-backed configuration for the multistream YOLOv8 example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .model_family import parse_model_family


VALID_VIDEO_MODES = {"clean", "annotated"}


@dataclass(frozen=True)
class ModelConfig:
    path: str
    family: str


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    rtsp_urls: list[str]
    tcp: bool
    latency_ms: int
    worker_count: int
    mailbox_depth: int
    profile: bool
    frames: int
    fps: int
    min_score: float
    nms_iou: float
    max_detections: int
    optiview_host: str
    optiview_video_port_base: int
    optiview_json_port_base: int
    optiview_json_offset_ms: float
    video_enabled: bool
    video_mode: str
    output_dir: str | None
    save_every: int


def json_output_enabled(cfg: AppConfig) -> bool:
    return cfg.video_mode == "clean"


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


def _optional_float(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model_cfg = _mapping(raw.get("model"), "model")
    input_cfg = _mapping(raw.get("input"), "input")
    runtime_cfg = _mapping(raw.get("runtime"), "runtime")
    inference_cfg = _mapping(raw.get("inference"), "inference")
    output_cfg = _mapping(raw.get("output"), "output")
    optiview_cfg = _mapping(output_cfg.get("optiview"), "output.optiview")

    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    rtsp_urls: list[str] = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, str) or not stream.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(stream)

    video_mode = str(output_cfg.get("video_mode", "clean")).strip().lower()
    if video_mode not in VALID_VIDEO_MODES:
        raise ValueError(f"output.video_mode must be one of {sorted(VALID_VIDEO_MODES)}")

    cfg = AppConfig(
        model=ModelConfig(
            path=_required_string(model_cfg, "path", "model"),
            family=parse_model_family(str(model_cfg.get("family", "auto"))),
        ),
        rtsp_urls=rtsp_urls,
        tcp=bool(input_cfg.get("tcp", False)),
        latency_ms=_optional_int(input_cfg, "latency_ms", 100),
        worker_count=_optional_int(runtime_cfg, "worker_count", 1),
        mailbox_depth=_optional_int(runtime_cfg, "mailbox_depth", 1),
        profile=bool(runtime_cfg.get("profile", False)),
        frames=_optional_int(inference_cfg, "frames", 0),
        fps=_optional_int(inference_cfg, "fps", 0),
        min_score=_optional_float(inference_cfg, "min_score", 0.25),
        nms_iou=_optional_float(inference_cfg, "nms_iou", 0.45),
        max_detections=_optional_int(inference_cfg, "max_detections", 100),
        optiview_host=_required_string(optiview_cfg, "host", "output.optiview"),
        optiview_video_port_base=_optional_int(optiview_cfg, "video_port_base", 9000),
        optiview_json_port_base=_optional_int(optiview_cfg, "json_port_base", 9100),
        optiview_json_offset_ms=_optional_float(optiview_cfg, "json_offset_ms", 0.0),
        video_enabled=bool(output_cfg.get("video_enabled", True)),
        video_mode=video_mode,
        output_dir=output_cfg.get("debug_dir"),
        save_every=_optional_int(output_cfg, "save_every", 0),
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
    if cfg.optiview_video_port_base <= 0:
        raise ValueError("output.optiview.video_port_base must be > 0")
    if cfg.optiview_json_port_base <= 0:
        raise ValueError("output.optiview.json_port_base must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if cfg.output_dir is not None and not isinstance(cfg.output_dir, str):
        raise ValueError("output.debug_dir must be a string or null")

    return cfg
