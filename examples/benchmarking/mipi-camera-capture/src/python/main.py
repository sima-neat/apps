#!/usr/bin/env python3
"""Capture and measure frames from a MIPI/libcamera camera with pyneat."""

from __future__ import annotations

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


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


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
            capture_buffers=integer(camera_raw, "capture_buffers", 32),
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
    if not 1 <= camera.capture_buffers <= 128:
        raise ValueError("camera.capture_buffers must be between 1 and 128")
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


def bounded_pull(
    configured_timeout_ms: int,
    remaining_seconds: float,
) -> tuple[int, bool]:
    """Return the bounded timeout and whether the run deadline imposed it."""
    deadline_limited = remaining_seconds * 1000.0 < configured_timeout_ms
    timeout_ms = max(
        1,
        min(configured_timeout_ms, math.ceil(remaining_seconds * 1000.0)),
    )
    return timeout_ms, deadline_limited


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


def nv12_contiguous_payload(tensor, width: int, height: int, y_role, uv_role) -> bytes:
    """Copy a possibly strided composite NV12 tensor into a tightly packed byte string."""
    if not tensor.is_nv12():
        raise RuntimeError("camera tensor is not NV12")
    tensor_width = tensor.width()
    tensor_height = tensor.height()
    if tensor_width > 0 and tensor_width != width:
        raise RuntimeError(f"camera tensor width {tensor_width} does not match {width}")
    if tensor_height > 0 and tensor_height != height:
        raise RuntimeError(f"camera tensor height {tensor_height} does not match {height}")

    planes = {plane.role: plane for plane in tensor.planes}
    try:
        y_plane = planes[y_role]
        uv_plane = planes[uv_role]
    except KeyError as exc:
        raise RuntimeError("NV12 tensor is missing its Y or UV plane") from exc

    raw = tensor.copy_payload_bytes()
    packed = bytearray(width * height * 3 // 2)

    def copy_plane(plane, rows: int, destination_offset: int) -> None:
        source_offset = int(plane.byte_offset)
        stride = int(plane.strides_bytes[0]) if plane.strides_bytes else width
        if source_offset < 0 or stride < width:
            raise RuntimeError("NV12 tensor has an invalid plane offset or stride")
        source_end = source_offset + stride * (rows - 1) + width
        if source_end > len(raw):
            raise RuntimeError("NV12 tensor plane exceeds its mapped payload")
        for row in range(rows):
            source = source_offset + row * stride
            destination = destination_offset + row * width
            packed[destination : destination + width] = raw[source : source + width]

    copy_plane(y_plane, height, 0)
    copy_plane(uv_plane, height // 2, width * height)
    return bytes(packed)


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


def capture_frames(config: AppConfig, pyneat) -> dict:
    camera = pyneat.CameraInputOptions()
    if config.camera.name:
        camera.camera_name = config.camera.name
    camera.width = config.camera.width
    camera.height = config.camera.height
    camera.framerate_num = config.camera.fps_num
    camera.framerate_den = config.camera.fps_den
    camera.format = config.camera.pixel_format
    camera.buffer_name = "camera0"
    camera.queue_depth = config.camera.queue_depth
    camera.allow_cpu_fallback = not config.camera.strict_zero_copy

    graph = pyneat.Graph("mipi_camera_capture")
    graph.add(
        pyneat.nodes.camera_input(
            camera,
            capture_buffer_count=config.camera.capture_buffers,
        )
    )
    graph.add(pyneat.nodes.output("frames", pyneat.OutputOptions.latest()))

    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Realtime
    options.queue_depth = config.camera.queue_depth
    options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    options.advanced.copy_input = False

    output_dir = config.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(graph.describe(), flush=True)
    run = graph.build(options)
    started = time.monotonic()
    deadline = started + config.capture.duration_seconds
    pts_values: list[int] = []
    captures: list[dict] = []
    frames = 0
    timeouts = 0
    target_index = 0
    error = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            pull_timeout_ms, deadline_limited_pull = bounded_pull(
                config.capture.pull_timeout_ms,
                remaining,
            )
            sample = run.pull("frames", pull_timeout_ms)
            now = time.monotonic()
            if sample is None:
                if deadline_limited_pull:
                    continue
                timeouts += 1
                print(f"timeout elapsed={now - started:.3f}", flush=True)
                continue

            frames += 1
            pts = getattr(sample, "pts_ns", -1)
            if pts is not None and pts >= 0:
                pts_values.append(int(pts))

            elapsed = now - started
            targets = config.capture.sample_times_seconds
            reached_targets = reached_target_indices(targets, target_index, elapsed)
            if reached_targets:
                tensors = list(sample.tensors)
                if not tensors:
                    raise RuntimeError(f"camera sample at {elapsed:.3f}s contains no tensor")
                payload = nv12_contiguous_payload(
                    tensors[0],
                    config.camera.width,
                    config.camera.height,
                    pyneat.PlaneRole.Y,
                    pyneat.PlaneRole.UV,
                )
                expected = config.camera.width * config.camera.height * 3 // 2
                if len(payload) != expected:
                    raise RuntimeError(
                        f"NV12 frame has {len(payload)} bytes; expected exactly {expected}"
                    )
                stats = frame_stats(payload, config.camera.width, config.camera.height)
                for capture_index in reached_targets:
                    filename = f"frame_{capture_index:02d}_{elapsed:07.3f}s.nv12"
                    with (output_dir / filename).open("wb") as output:
                        output.write(payload)
                    entry = {
                        "target_s": targets[capture_index],
                        "elapsed_s": elapsed,
                        "frame": frames,
                        "file": filename,
                        **stats,
                    }
                    captures.append(entry)
                    print("CAPTURE " + json.dumps(entry, sort_keys=True), flush=True)
                target_index = reached_targets.stop
            if frames % 300 == 0:
                print(f"progress frames={frames} elapsed={elapsed:.3f}", flush=True)
        if target_index != len(config.capture.sample_times_seconds):
            raise RuntimeError(
                f"captured {target_index} of {len(config.capture.sample_times_seconds)} "
                "requested snapshots"
            )
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        try:
            elapsed = time.monotonic() - started
            interarrival_mean_ms, interarrival_max_ms = pts_interarrival_stats_ms(
                pts_values
            )
            summary = {
                "camera": config.camera.name or "<libcamera-default>",
                "width": config.camera.width,
                "height": config.camera.height,
                "requested_fps": config.camera.fps_num / config.camera.fps_den,
                "strict_zero_copy": config.camera.strict_zero_copy,
                "capture_buffers": config.camera.capture_buffers,
                "elapsed_s": elapsed,
                "frames_pulled": frames,
                "pull_fps": frames / elapsed if elapsed else 0.0,
                "timeouts": timeouts,
                "captures": captures,
                "interarrival_mean_ms": interarrival_mean_ms,
                "interarrival_max_ms": interarrival_max_ms,
                "pts_span_s": (
                    (pts_values[-1] - pts_values[0]) / 1e9
                    if len(pts_values) >= 2
                    else None
                ),
                "error": error,
            }
        finally:
            try:
                run.close()
            finally:
                del run
                gc.collect()
        write_json(output_dir / "summary.json", summary)
        print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.validate_config_only:
        print(f"config valid: {args.config}")
        return 0

    try:
        import pyneat
    except ImportError:
        print("pyneat is not importable. Run: source ~/pyneat/bin/activate", file=sys.stderr)
        return 3

    try:
        capture_frames(config, pyneat)
    except KeyboardInterrupt:
        print("capture interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
