"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

import argparse

import json

import sys

from datetime import datetime, timezone

from pathlib import Path

import yaml

DECODE_TYPES = {"yolo26-det": "YoloV26", "yolo26-seg": "YoloV26Seg"}

BOXDECODE_TOP_K = 100

DEFAULT_MIN_DURATION_MS = 30000

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload

def spec_strings(model, method_name: str) -> list[str]:
    try:
        return [str(spec) for spec in getattr(model, method_name)()]
    except Exception as exc:
        return [f"unavailable: {exc}"]

def route_fields(model) -> dict:
    """Describe the resolved route, without discarding a completed benchmark if info() fails."""
    try:
        info = model.info()
    except Exception as exc:
        return {"resolved_postprocess": f"unavailable: {exc}", "output_topology": None}
    return {
        "resolved_postprocess": info.selection.selected_post_kind,
        "output_topology": {
            "physical": info.output_topology.physical_outputs,
            "logical": info.output_topology.logical_outputs,
            "packed": info.output_topology.packed_outputs,
        },
    }

def telemetry_status(report) -> str | None:
    try:
        return str(report.telemetry.status)
    except Exception:
        return None

def write_report(path: Path, model_path: Path, frames: int, min_duration_ms: int, decode_type,
                 model, report) -> None:
    data = {
        "benchmark": {
            "type": "model.synthetic",
            "frames": frames,
            "min_duration_ms": min_duration_ms,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "path": str(model_path),
            "file": model_path.name,
            "requested_decode_type": decode_type,
            "boxdecode_top_k": None if decode_type is None else BOXDECODE_TOP_K,
            **route_fields(model),
            "input_specs": spec_strings(model, "input_specs"),
            "output_specs": spec_strings(model, "output_specs"),
        },
        "metrics": {
            "latency_ms": report.latency_ms,
            "latency_p50_ms": report.latency_p50_ms,
            "latency_p95_ms": report.latency_p95_ms,
            "latency_p99_ms": report.latency_p99_ms,
            "fps": report.fps,
            "throughput_samples": report.throughput_samples,
            "throughput_duration_ms": report.throughput_duration_ms,
            "avg_power_watts": report.avg_power_watts,
            "energy_joules": report.energy_joules,
            "telemetry_status": telemetry_status(report),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
