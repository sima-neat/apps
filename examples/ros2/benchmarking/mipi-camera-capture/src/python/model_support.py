"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass

import gc

import json

import math

from pathlib import Path

import statistics

import sys

import time

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_mipi_camera_capture')) / "common" / "config.yaml"

MAX_RETAINED_SNAPSHOT_BYTES = 128 * 1024 * 1024

@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    pixel_format: str
    capture_buffers: int
    strict_zero_copy: bool
    queue_depth: int

@dataclass(frozen=True)
class CaptureConfig:
    duration_seconds: float
    sample_times_seconds: tuple[float, ...]
    pull_timeout_ms: int

@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig
    capture: CaptureConfig
    output_directory: Path

@dataclass(frozen=True)
class PendingCapture:
    target_seconds: float
    elapsed_seconds: float
    frame: int
    filename: str
    payload: bytes

def section(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value

def integer(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value

def number(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)

def boolean(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value

def string(raw: dict, key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value

def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    camera_raw = section(raw, "camera")
    capture_raw = section(raw, "capture")
    output_raw = section(raw, "output")
    sample_times = capture_raw.get("sample_times_seconds", [3, 15, 27])
    if not isinstance(sample_times, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in sample_times
    ):
        raise ValueError("capture.sample_times_seconds must be a list of numbers")

    output_directory = string(output_raw, "directory", "sandbox/mipi-camera-capture")
    if not output_directory.strip():
        raise ValueError("output.directory must not be empty")

    config = AppConfig(
        camera=CameraConfig(
            name=string(camera_raw, "name", ""),
            width=integer(camera_raw, "width", 1920),
            height=integer(camera_raw, "height", 1080),
            fps_num=integer(camera_raw, "fps_num", 30),
            fps_den=integer(camera_raw, "fps_den", 1),
            pixel_format=string(camera_raw, "format", "NV12").upper(),
            capture_buffers=integer(camera_raw, "capture_buffers", 0),
            strict_zero_copy=boolean(camera_raw, "strict_zero_copy", True),
            queue_depth=integer(camera_raw, "queue_depth", 2),
        ),
        capture=CaptureConfig(
            duration_seconds=number(capture_raw, "duration_seconds", 30),
            sample_times_seconds=tuple(float(value) for value in sample_times),
            pull_timeout_ms=integer(capture_raw, "pull_timeout_ms", 2000),
        ),
        output_directory=Path(output_directory),
    )
    validate_config(config)
    return config

def validate_config(config: AppConfig) -> None:
    camera = config.camera
    capture = config.capture
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError("camera width and height must be positive")
    if camera.width % 2 or camera.height % 2:
        raise ValueError("NV12 camera width and height must be even")
    if camera.fps_num <= 0 or camera.fps_den <= 0:
        raise ValueError("camera frame-rate numerator and denominator must be positive")
    if camera.pixel_format != "NV12":
        raise ValueError("camera.format must be NV12")
    if not 0 <= camera.capture_buffers <= 128:
        raise ValueError("camera.capture_buffers must be between 0 and 128")
    if camera.queue_depth <= 0:
        raise ValueError("camera.queue_depth must be positive")
    if not math.isfinite(capture.duration_seconds) or capture.duration_seconds <= 0:
        raise ValueError("capture.duration_seconds must be finite and positive")
    if capture.pull_timeout_ms <= 0:
        raise ValueError("capture.pull_timeout_ms must be positive")
    if any(
        not math.isfinite(value)
        or value < 0
        or value >= capture.duration_seconds
        for value in capture.sample_times_seconds
    ):
        raise ValueError(
            "sample times must be finite, at least zero, and earlier than "
            "capture.duration_seconds"
        )
    if tuple(sorted(set(capture.sample_times_seconds))) != capture.sample_times_seconds:
        raise ValueError("sample times must be unique and in increasing order")
    retained_snapshot_bytes = (
        len(capture.sample_times_seconds) * camera.width * camera.height * 3 // 2
    )
    if retained_snapshot_bytes > MAX_RETAINED_SNAPSHOT_BYTES:
        raise ValueError(
            f"snapshot schedule may retain {retained_snapshot_bytes} bytes; maximum is "
            f"{MAX_RETAINED_SNAPSHOT_BYTES}"
        )
    frame_period_seconds = camera.fps_den / camera.fps_num
    if any(
        value > capture.duration_seconds - frame_period_seconds
        for value in capture.sample_times_seconds
    ):
        raise ValueError(
            "sample times must leave at least one requested frame period before "
            "capture.duration_seconds"
        )

def pts_interarrival_stats_ms(pts_values: list[int]) -> tuple[float | None, float | None]:
    """Return mean and maximum intervals from consecutive monotonic PTS values."""
    intervals = [
        (current - previous) / 1e6
        for previous, current in zip(pts_values, pts_values[1:])
        if current >= previous
    ]
    if not intervals:
        return None, None
    return statistics.fmean(intervals), max(intervals)

def reached_target_indices(
    targets: tuple[float, ...],
    target_index: int,
    elapsed_seconds: float,
) -> range:
    """Return every pending snapshot target reached by the current frame."""
    end = target_index
    while end < len(targets) and elapsed_seconds >= targets[end]:
        end += 1
    return range(target_index, end)

def frame_stats(payload: bytes, width: int, height: int) -> dict[str, float | int]:
    """Compute compact luminance statistics without requiring NumPy or OpenCV."""
    y_size = width * height
    if len(payload) < y_size:
        raise RuntimeError(f"short NV12 Y plane: {len(payload)} bytes, expected {y_size}")
    histogram = [0] * 256
    total = 0
    for value in payload[:y_size]:
        histogram[value] += 1
        total += value

    percentiles: dict[str, int] = {}
    cumulative = 0
    for value, frequency in enumerate(histogram):
        cumulative += frequency
        for label, fraction in (("p01", 0.01), ("p50", 0.50), ("p99", 0.99)):
            if label not in percentiles and cumulative >= y_size * fraction:
                percentiles[label] = value

    return {
        "y_mean": total / y_size,
        "y_min": next(index for index, count in enumerate(histogram) if count),
        "y_max": next(index for index in range(255, -1, -1) if histogram[index]),
        "y_p01": percentiles["p01"],
        "y_p50": percentiles["p50"],
        "y_p99": percentiles["p99"],
        "y_low_clip_pct": 100.0 * sum(histogram[:5]) / y_size,
        "y_high_clip_pct": 100.0 * sum(histogram[251:]) / y_size,
    }

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")

def write_captures(
    pending_captures: list[PendingCapture],
    output_dir: Path,
    width: int,
    height: int,
) -> list[dict]:
    """Materialize snapshots after the live camera run has stopped."""
    captures = []
    for pending in pending_captures:
        with (output_dir / pending.filename).open("wb") as output:
            output.write(pending.payload)
        entry = {
            "target_s": pending.target_seconds,
            "elapsed_s": pending.elapsed_seconds,
            "frame": pending.frame,
            "file": pending.filename,
            **frame_stats(pending.payload, width, height),
        }
        captures.append(entry)
        print("CAPTURE " + json.dumps(entry, sort_keys=True), flush=True)
    return captures
