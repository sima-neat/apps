#!/usr/bin/env python3
"""Run all model benchmarks and refresh BENCHMARK_RESULTS.md."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
APPS_ROOT = Path(__file__).resolve().parents[4]
MAIN_PY = EXAMPLE_ROOT / "src" / "python" / "main.py"
RESULTS_MD = EXAMPLE_ROOT / "BENCHMARK_RESULTS.md"
REPORTS_DIR = APPS_ROOT / "sandbox" / "model-benchmark" / "runs"


@dataclass(frozen=True)
class ModelRow:
    model_id: str
    package: str
    used_by: str
    preferred_path: str


@dataclass(frozen=True)
class BenchmarkFailure:
    model_id: str
    reason: str


GROUPS: tuple[tuple[str, tuple[ModelRow, ...]], ...] = (
    (
        "General Models",
        (
            ModelRow("resnet_50", "resnet_50_mpk.tar.gz", "image-classifier",
                     "assets/models/resnet_50_mpk.tar.gz"),
            ModelRow("depth_anything_v2_vits", "depth_anything_v2_vits_mpk.tar.gz",
                     "depth-estimator", "assets/models/depth_anything_v2_vits_mpk.tar.gz"),
            ModelRow("retinaface_mobilenet25", "retinaface_mobilenet25_mod_0_mpk.tar.gz",
                     "face-detector", "assets/models/retinaface_mobilenet25_mod_0_mpk.tar.gz"),
            ModelRow("detr_resnet50_modified_class_embed_bbox_embed",
                     "detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz",
                     "detr-object-detector",
                     "assets/models/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz"),
            ModelRow("yolo_v8n_seg", "yolo_v8n_seg_mpk.tar.gz",
                     "yolov8-instance-segmenter", "assets/models/yolo_v8n_seg_mpk.tar.gz"),
        ),
    ),
    (
        "YOLO26 Detection",
        (
            ModelRow("yolo26n-det-bf16-mla_tess-b1",
                     "yolo26n-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-people-tracker",
                     "assets/models/YOLO26-DETECTION/yolo26n-det-bf16-mla_tess-b1.tar.gz"),
            ModelRow("yolo26s-det-bf16-mla_tess-b1",
                     "yolo26s-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-people-tracker",
                     "assets/models/YOLO26-DETECTION/yolo26s-det-bf16-mla_tess-b1.tar.gz"),
            ModelRow("yolo26m-det-bf16-mla_tess-b1",
                     "yolo26m-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "yolo26-object-detector, detection-to-vlm-assistant, "
                     "multi-stream-people-tracker",
                     "assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz"),
            ModelRow("yolo26l-det-bf16-mla_tess-b1",
                     "yolo26l-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-people-tracker",
                     "assets/models/YOLO26-DETECTION/yolo26l-det-bf16-mla_tess-b1.tar.gz"),
            ModelRow("yolo26x-det-bf16-mla_tess-b1",
                     "yolo26x-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-people-tracker",
                     "assets/models/YOLO26-DETECTION/yolo26x-det-bf16-mla_tess-b1.tar.gz"),
            ModelRow("yolo26m-det-bf16-b1", "yolo26m-det-bf16-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "detection-to-vlm-assistant, multi-stream-people-tracker",
                     "assets/models/YOLO26-DETECTION/yolo26m-det-bf16-b1.tar.gz"),
            ModelRow("yolo26m-det-int8-b1", "yolo26m-det-int8-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "detection-to-vlm-assistant, multi-stream-people-tracker",
                     "assets/models/YOLO26-DETECTION/yolo26m-det-int8-b1.tar.gz"),
        ),
    ),
    (
        "YOLO26 Segmentation",
        (
            ModelRow("yolo26n-seg-bf16-mla_tess", "yolo26n-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26n-seg-bf16-mla_tess.tar.gz"),
            ModelRow("yolo26s-seg-bf16-mla_tess", "yolo26s-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26s-seg-bf16-mla_tess.tar.gz"),
            ModelRow("yolo26m-seg-bf16-mla_tess", "yolo26m-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26m-seg-bf16-mla_tess.tar.gz"),
            ModelRow("yolo26l-seg-bf16-mla_tess", "yolo26l-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26l-seg-bf16-mla_tess.tar.gz"),
            ModelRow("yolo26x-seg-bf16-mla_tess", "yolo26x-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26x-seg-bf16-mla_tess.tar.gz"),
            ModelRow("yolo26m-seg-bf16-b1", "yolo26m-seg-bf16-b1.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/yolo26m-seg-bf16-b1.tar.gz"),
            ModelRow("yolo26m-seg-bf16-mla_tess-b1",
                     "yolo26m-seg-bf16-mla_tess-b1.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26m-seg-bf16-mla_tess-b1.tar.gz"),
            ModelRow("yolo26m-seg-int8-b1", "yolo26m-seg-int8-b1.tar.gz",
                     "single-stream-instance-segmenter",
                     "assets/models/YOLO26-SEGMENTATION/yolo26m-seg-int8-b1.tar.gz"),
        ),
    ),
)


def all_rows() -> tuple[ModelRow, ...]:
    return tuple(row for _, rows in GROUPS for row in rows)


def selected_rows(model_ids: list[str] | None) -> tuple[ModelRow, ...]:
    rows = all_rows()
    if not model_ids:
        return rows
    known = {row.model_id: row for row in rows}
    missing = sorted(set(model_ids) - set(known))
    if missing:
        raise ValueError(f"unknown model id(s): {', '.join(missing)}")
    return tuple(known[model_id] for model_id in model_ids)


def existing_context_value(name: str, default: str) -> str:
    if not RESULTS_MD.exists():
        return default
    match = re.search(rf"^\| {re.escape(name)} \| (.*?) \|$", RESULTS_MD.read_text(), re.M)
    return match.group(1) if match else default


def detect_sdk() -> str:
    try:
        result = subprocess.run(["neat", "--offline"], cwd=APPS_ROOT, check=False,
                                text=True, capture_output=True)
    except FileNotFoundError:
        return existing_context_value("SDK", "unknown")
    match = re.search(r"^\s*SDK Version\s+(.+)$", result.stdout, re.M)
    return match.group(1).strip() if match else existing_context_value("SDK", "unknown")


def resolve_model_path(row: ModelRow) -> Path:
    preferred = APPS_ROOT / row.preferred_path
    if preferred.exists():
        return preferred
    matches = sorted((APPS_ROOT / "assets" / "models").rglob(row.package))
    if not matches:
        raise FileNotFoundError(f"model package not found: {row.package}")
    return matches[0]


def run_benchmarks(frames: int, reports_dir: Path, rows: tuple[ModelRow, ...]) -> list[BenchmarkFailure]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    failures: list[BenchmarkFailure] = []
    for row in rows:
        try:
            model_path = resolve_model_path(row)
        except FileNotFoundError as exc:
            failures.append(BenchmarkFailure(row.model_id, str(exc)))
            print(f"[failed] {row.model_id}: {exc}", file=sys.stderr, flush=True)
            continue
        out_path = reports_dir / f"{row.model_id}.json"
        print(f"[benchmark] {row.model_id}: {model_path}", flush=True)
        result = subprocess.run([
            sys.executable,
            str(MAIN_PY),
            "--model",
            str(model_path),
            "--frames",
            str(frames),
            "--output-json",
            str(out_path),
        ], cwd=APPS_ROOT, check=False)
        if result.returncode != 0:
            failures.append(BenchmarkFailure(row.model_id, f"exit {result.returncode}"))
            print(f"[failed] {row.model_id}: exit {result.returncode}", file=sys.stderr,
                  flush=True)
    return failures


def report_metrics(reports_dir: Path, row: ModelRow) -> tuple[float, float]:
    report_path = reports_dir / f"{row.model_id}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing benchmark report: {report_path}")
    data = json.loads(report_path.read_text())
    metrics = data["metrics"]
    return float(metrics["latency_ms"]), float(metrics["fps"])


def report_date(reports_dir: Path) -> str:
    dates: list[str] = []
    for row in all_rows():
        report_path = reports_dir / f"{row.model_id}.json"
        if not report_path.exists():
            continue
        timestamp = json.loads(report_path.read_text()).get("benchmark", {}).get("timestamp_utc")
        if timestamp:
            dates.append(datetime.fromisoformat(timestamp).date().isoformat())
    return max(dates) if dates else date.today().isoformat()


def render_markdown(frames: int, sdk: str, run_date: str, reports_dir: Path) -> str:
    lines = [
        "# Model Benchmark Results",
        "",
        "This file is a manually maintained reference table for Apps-supported compiled model packages.",
        "The numbers come from `pyneat.Model.benchmark()` through `src/python/main.py`.",
        "",
        "These results measure the compiled model package only. They do not include camera ingest, RTSP decode, Insight output, metadata conversion, overlays, or application postprocessing.",
        "",
        "## Run Context",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Target | Modalix (`aarch64`) |",
        f"| SDK | {sdk} |",
        f"| Date | {run_date} |",
        f"| Frames | {frames} |",
        "| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py --model <model-package>` |",
        "| Refresh Script | `python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py --run` |",
        "| JSON Reports | `sandbox/model-benchmark/runs/<model-id>.json` |",
        "| Power Columns | Omitted |",
        "",
    ]
    for title, rows in GROUPS:
        lines.extend([
            f"## {title}",
            "",
            "| Model ID | Package | Used By | Latency / FPS |",
            "| --- | --- | --- | ---: |",
        ])
        for row in rows:
            latency_ms, fps = report_metrics(reports_dir, row)
            lines.append(
                f"| `{row.model_id}` | `{row.package}` | {row.used_by} | "
                f"{latency_ms:.3f} / {fps:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run benchmarks before refreshing")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--sdk", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--models", nargs="*", help="optional model IDs to rerun")
    parser.add_argument("--allow-partial-update", action="store_true",
                        help="refresh Markdown even when a requested benchmark fails")
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("--frames must be > 0")

    rows = selected_rows(args.models)
    failures: list[BenchmarkFailure] = []
    if args.run:
        failures = run_benchmarks(args.frames, args.reports_dir, rows)

    if failures and not args.allow_partial_update:
        print("\nBenchmark failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure.model_id}: {failure.reason}", file=sys.stderr)
        failed_ids = " ".join(failure.model_id for failure in failures)
        print("\nRerun only failed model(s):", file=sys.stderr)
        print("  python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py "
              f"--run --frames {args.frames} --models {failed_ids}", file=sys.stderr)
        print("BENCHMARK_RESULTS.md was not updated.", file=sys.stderr)
        return 1

    sdk = args.sdk or detect_sdk()
    run_date = args.date or (date.today().isoformat() if args.run else report_date(args.reports_dir))
    RESULTS_MD.write_text(render_markdown(args.frames, sdk, run_date, args.reports_dir),
                          encoding="utf-8")
    print(f"updated {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
