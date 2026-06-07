"""YAML-backed configuration for the Python pose estimation example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    path: str


@dataclass(frozen=True)
class IoConfig:
    input_dir: str
    output_dir: str


@dataclass(frozen=True)
class RuntimeConfig:
    infer_size: int = 640
    timeout_ms: int = 5000
    upsample_factor: float = 4.0


@dataclass(frozen=True)
class DecodeConfig:
    keypoint_score: float = 0.1
    nms_radius: int = 6
    paf_score: float = 0.05
    paf_success_ratio: float = 0.8
    paf_samples: int = 10
    min_valid_joints: int = 3
    min_avg_person_score: float = 0.2


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    io: IoConfig
    runtime: RuntimeConfig
    decode: DecodeConfig


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


def _optional_int(mapping: dict[str, Any], key: str, default: int, section: str) -> int:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{section}.{key} must be an integer")
    return int(value)


def _optional_float(mapping: dict[str, Any], key: str, default: float, section: str) -> float:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{section}.{key} must be numeric")
    return float(value)


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = _mapping(raw.get("model"), "model")
    io_cfg = _mapping(raw.get("io"), "io")
    runtime = _mapping(raw.get("runtime"), "runtime")
    decode = _mapping(raw.get("decode"), "decode")

    cfg = AppConfig(
        model=ModelConfig(path=_required_string(model, "path", "model")),
        io=IoConfig(
            input_dir=_required_string(io_cfg, "input_dir", "io"),
            output_dir=_required_string(io_cfg, "output_dir", "io"),
        ),
        runtime=RuntimeConfig(
            infer_size=_optional_int(runtime, "infer_size", 640, "runtime"),
            timeout_ms=_optional_int(runtime, "timeout_ms", 20000, "runtime"),
            upsample_factor=_optional_float(runtime, "upsample_factor", 4.0, "runtime"),
        ),
        decode=DecodeConfig(
            keypoint_score=_optional_float(decode, "keypoint_score", 0.1, "decode"),
            nms_radius=_optional_int(decode, "nms_radius", 6, "decode"),
            paf_score=_optional_float(decode, "paf_score", 0.05, "decode"),
            paf_success_ratio=_optional_float(
                decode, "paf_success_ratio", 0.8, "decode"
            ),
            paf_samples=_optional_int(decode, "paf_samples", 10, "decode"),
            min_valid_joints=_optional_int(decode, "min_valid_joints", 3, "decode"),
            min_avg_person_score=_optional_float(
                decode, "min_avg_person_score", 0.2, "decode"
            ),
        ),
    )

    if cfg.runtime.infer_size <= 0:
        raise ValueError("runtime.infer_size must be > 0")
    if cfg.runtime.timeout_ms <= 0:
        raise ValueError("runtime.timeout_ms must be > 0")
    if cfg.runtime.upsample_factor <= 0.0:
        raise ValueError("runtime.upsample_factor must be > 0")
    if not 0.0 <= cfg.decode.keypoint_score <= 1.0:
        raise ValueError("decode.keypoint_score must be between 0 and 1")
    if cfg.decode.nms_radius < 0:
        raise ValueError("decode.nms_radius must be >= 0")
    if not 0.0 <= cfg.decode.paf_score <= 1.0:
        raise ValueError("decode.paf_score must be between 0 and 1")
    if not 0.0 <= cfg.decode.paf_success_ratio <= 1.0:
        raise ValueError("decode.paf_success_ratio must be between 0 and 1")
    if cfg.decode.paf_samples < 2:
        raise ValueError("decode.paf_samples must be >= 2")
    if cfg.decode.min_valid_joints <= 0:
        raise ValueError("decode.min_valid_joints must be > 0")
    if cfg.decode.min_avg_person_score < 0.0:
        raise ValueError("decode.min_avg_person_score must be >= 0")

    return cfg
