#!/usr/bin/env python3
"""Run pyneat.Model.benchmark() for one compiled model package."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


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


def write_report(path: Path, model_path: Path, frames: int, model, report) -> None:
    data = {
        "benchmark": {
            "type": "model.synthetic",
            "frames": frames,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "model": {
            "path": str(model_path),
            "file": model_path.name,
            "input_specs": spec_strings(model, "input_specs"),
            "output_specs": spec_strings(model, "output_specs"),
        },
        "metrics": {
            "latency_ms": report.latency_ms,
            "fps": report.fps,
            "avg_power_watts": report.avg_power_watts,
            "energy_joules": report.energy_joules,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--model", type=Path, help="compiled model package to benchmark")
    parser.add_argument("--frames", type=int, help="measured synthetic frames")
    parser.add_argument("--output-json", type=Path, help="benchmark report JSON path")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        model_path = (
            args.model
            if args.model is not None
            else Path(str(config.get("model", {}).get("path", "")))
        )
        frames = (
            args.frames
            if args.frames is not None
            else int(config.get("benchmark", {}).get("frames", 1000))
        )
        report_path = args.output_json if args.output_json is not None else Path(
            str(config.get("output", {}).get("report_json", "sandbox/model-benchmark/report.json"))
        )
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if frames <= 0:
        print("benchmark.frames must be > 0", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"model file does not exist: {model_path}", file=sys.stderr)
        return 2

    try:
        import pyneat
    except ImportError:
        print("pyneat is not importable. Run: source ~/pyneat/bin/activate", file=sys.stderr)
        return 3

    try:
        model = pyneat.Model(str(model_path))
        report = model.benchmark(frames)
        write_report(report_path, model_path, frames, model, report)
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 4

    print(f"latency_ms={report.latency_ms}")
    print(f"fps={report.fps}")
    print(f"avg_power_watts={report.avg_power_watts}")
    print(f"energy_joules={report.energy_joules}")
    print(f"report_json={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
