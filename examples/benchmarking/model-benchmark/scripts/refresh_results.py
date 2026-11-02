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
MODELS_DIR = APPS_ROOT / "models"
# Modalix can boot with the MLA capped below its 1 GHz maximum, which lowers every result.
MLA_FREQ = Path("/sys/class/devfreq/5000000.mla/cur_freq")


@dataclass(frozen=True)
class ModelRow:
    model_id: str
    package: str  # Path under the Model Zoo's models/modalix/ directory.
    used_by: str
    decode_type: str | None = None

    @property
    def file(self) -> str:
        return Path(self.package).name


@dataclass(frozen=True)
class BenchmarkFailure:
    model_id: str
    reason: str


GROUPS: tuple[tuple[str, tuple[ModelRow, ...]], ...] = (
    (
        "Classification",
        (
            ModelRow("resnet_50",
                     "resnet_50_mpk.tar.gz",
                     "image-classification-explorer, mipi-multi-model"),
            ModelRow("resnet_18",
                     "resnet_18_mpk.tar.gz",
                     "image-classification-explorer"),
            ModelRow("efficientnet_b0",
                     "efficientnet_b0_mpk.tar.gz",
                     "image-classification-explorer"),
            ModelRow("densenet_121",
                     "densenet_121_mpk.tar.gz",
                     "image-classification-explorer"),
        ),
    ),
    (
        "Detection",
        (
            ModelRow("detr_resnet50_modified_class_embed_bbox_embed",
                     "detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz",
                     "detr-object-detector"),
            ModelRow("ssd_mobilenet_v2_modalix_int8_tess_mla",
                     "ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz",
                     "ssd-mobilenet-object-detector, mipi-multi-model"),
            ModelRow("yolo_11s",
                     "yolo_11s_mpk.tar.gz",
                     "multi-stream-multi-model"),
            ModelRow("yolo26n_plc",
                     "yolo26n_plc_mpk.tar.gz",
                     "pcb-defect-detector"),
            ModelRow("yolo26n_p2_tiny_drone_int8_qat_b1",
                     "yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz",
                     "yolo26-tiny-drone-tracker, parallel-apps"),
            ModelRow("yolo26n_obb_w8a8",
                     "yolo26n_obb_w8a8_mpk.tar.gz",
                     "multi-stream-obb-detector"),
            ModelRow("yolo26n_obb_w16a16",
                     "yolo26n_obb_w16a16_mpk.tar.gz",
                     "multi-stream-obb-detector"),
        ),
    ),
    (
        "Face Detection",
        (
            ModelRow("retinaface_mobilenet25",
                     "retinaface_mobilenet25_mod_0_mpk.tar.gz",
                     "face-detector"),
            ModelRow("yolov5s_face_raw_split",
                     "yolov5s_face_raw_split_mpk.tar.gz",
                     "single-stream-thermal-face-detector"),
        ),
    ),
    (
        "YOLO26 Detection",
        (
            ModelRow("yolo26n-det-bf16-mla_tess-b1",
                     "yolo26-detection/yolo26n-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, "
                     "mipi-multi-model"),
            ModelRow("yolo26s-det-bf16-mla_tess-b1",
                     "yolo26-detection/yolo26s-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant"),
            ModelRow("yolo26m-det-bf16-mla_tess-b1",
                     "yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant"),
            ModelRow("yolo26l-det-bf16-mla_tess-b1",
                     "yolo26-detection/yolo26l-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant"),
            ModelRow("yolo26x-det-bf16-mla_tess-b1",
                     "yolo26-detection/yolo26x-det-bf16-mla_tess-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant"),
            ModelRow("yolo26m-det-bf16-b1",
                     "yolo26-detection/yolo26m-det-bf16-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant"),
            ModelRow("yolo26n-det-int8-b1",
                     "yolo26-detection/yolo26n-det-int8-b1.tar.gz",
                     "high-density-multi-stream-object-detector, "
                     "pcie-high-density-multi-stream-object-detector, mipi-multi-model"),
            ModelRow("yolo26m-det-int8-b1",
                     "yolo26-detection/yolo26m-det-int8-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, "
                     "multi-stream-blazepose3d, multi-stream-multi-model, parallel-apps"),
            ModelRow("yolo26m-det-int8-b1-boxdecode",
                     "yolo26-detection/yolo26m-det-int8-b1.tar.gz",
                     "single-stream-object-detector, multi-stream-object-detector, "
                     "multi-stream-tracker, yolo26-object-detector, detection-to-vlm-assistant, "
                     "multi-stream-blazepose3d, multi-stream-multi-model, parallel-apps",
                     decode_type="yolo26-det"),
        ),
    ),
    (
        "YOLO26 Segmentation",
        (
            ModelRow("yolo26n-seg-bf16-mla_tess",
                     "yolo26-segmentation/yolo26n-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter, mipi-multi-model"),
            ModelRow("yolo26s-seg-bf16-mla_tess",
                     "yolo26-segmentation/yolo26s-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo26m-seg-bf16-mla_tess",
                     "yolo26-segmentation/yolo26m-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo26l-seg-bf16-mla_tess",
                     "yolo26-segmentation/yolo26l-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo26x-seg-bf16-mla_tess",
                     "yolo26-segmentation/yolo26x-seg-bf16-mla_tess.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo26m-seg-bf16-b1",
                     "yolo26-segmentation/yolo26m-seg-bf16-b1.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo26m-seg-bf16-mla_tess-b1",
                     "yolo26-segmentation/yolo26m-seg-bf16-mla_tess-b1.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo26m-seg-int8-b1",
                     "yolo26-segmentation/yolo26m-seg-int8-b1.tar.gz",
                     "single-stream-instance-segmenter, parallel-apps"),
            ModelRow("yolo26m-seg-int8-b1-boxdecode",
                     "yolo26-segmentation/yolo26m-seg-int8-b1.tar.gz",
                     "single-stream-instance-segmenter, parallel-apps",
                     decode_type="yolo26-seg"),
        ),
    ),
    (
        "YOLO26 Pose",
        (
            ModelRow("yolo26n-pose-bf16-mla_tess-b1",
                     "yolo26-pose/yolo26n-pose-bf16-mla_tess-b1.tar.gz",
                     "multi-stream-pose-estimator, mipi-multi-model"),
            ModelRow("yolo26s-pose-bf16-mla_tess-b1",
                     "yolo26-pose/yolo26s-pose-bf16-mla_tess-b1.tar.gz",
                     "multi-stream-pose-estimator"),
            ModelRow("yolo26m-pose-bf16-mla_tess-b1",
                     "yolo26-pose/yolo26m-pose-bf16-mla_tess-b1.tar.gz",
                     "multi-stream-pose-estimator"),
            ModelRow("yolo26l-pose-bf16-mla_tess-b1",
                     "yolo26-pose/yolo26l-pose-bf16-mla_tess-b1.tar.gz",
                     "multi-stream-pose-estimator"),
            ModelRow("yolo26x-pose-bf16-mla_tess-b1",
                     "yolo26-pose/yolo26x-pose-bf16-mla_tess-b1.tar.gz",
                     "multi-stream-pose-estimator"),
            ModelRow("yolo26m-pose-bf16-b1",
                     "yolo26-pose/yolo26m-pose-bf16-b1.tar.gz",
                     "multi-stream-pose-estimator"),
            ModelRow("yolo26m-pose-int8-b1",
                     "yolo26-pose/yolo26m-pose-int8-b1.tar.gz",
                     "multi-stream-pose-estimator, multi-stream-multi-model, parallel-apps"),
        ),
    ),
    (
        "Other Segmentation",
        (
            ModelRow("yolo_v8n_seg",
                     "yolo_v8n_seg_mpk.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo_v8s_seg",
                     "yolo_v8s_seg_mpk.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo_v8m_seg",
                     "yolo_v8m_seg_mpk.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo_v8l_seg",
                     "yolo_v8l_seg_mpk.tar.gz",
                     "single-stream-instance-segmenter"),
            ModelRow("yolo_11s_seg",
                     "yolo_11s_seg_mpk.tar.gz",
                     "multi-stream-multi-model"),
        ),
    ),
    (
        "Depth, Keypoints, Pose, And Anomaly",
        (
            ModelRow("depth_anything_v2_vits",
                     "depth_anything_v2_vits_mpk.tar.gz",
                     "depth-estimator"),
            ModelRow("midas_v21_small_256",
                     "midas_v21_small_256_mpk.tar.gz",
                     "mipi-multi-model"),
            ModelRow("superpoint",
                     "superpoint_mpk.tar.gz",
                     "superpoint-feature-extractor"),
            ModelRow("blazepose_ghum_heavy_w16a16",
                     "blazepose_ghum_heavy_w16a16_mpk.tar.gz",
                     "multi-stream-blazepose3d, parallel-apps"),
            ModelRow("fastflow_demo",
                     "fastflow_demo_mpk.tar.gz",
                     "fastflow-anomaly-detector"),
            ModelRow("patchcore_wide_resnet50_2_bf16_mla",
                     "patchcore_wide_resnet50_2_bf16_mla.tar.gz",
                     "patchcore-anomaly-detector"),
        ),
    ),
    (
        "Multi-Package Model Parts",
        (
            ModelRow("rf_detr_nano_w8a8_w16a16_backbone",
                     "rf_detr_nano_w8a8_w16a16_backbone_mpk.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rf_detr_nano_w8a8_w16a16_decoder",
                     "rf_detr_nano_w8a8_w16a16_decoder_mpk.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rf_detr_small_w8a8_w16a16_backbone",
                     "rf_detr_small_w8a8_w16a16_backbone_mpk.tar.gz",
                     "rfdetr-detection-segmentation, parallel-apps"),
            ModelRow("rf_detr_small_w8a8_w16a16_decoder",
                     "rf_detr_small_w8a8_w16a16_decoder_mpk.tar.gz",
                     "rfdetr-detection-segmentation, parallel-apps"),
            ModelRow("rf_detr_medium_w8a8_w16a16_backbone",
                     "rf_detr_medium_w8a8_w16a16_backbone_mpk.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rf_detr_medium_w8a8_w16a16_decoder",
                     "rf_detr_medium_w8a8_w16a16_decoder_mpk.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rf_detr_large_w8a8_w16a16_backbone",
                     "rf_detr_large_w8a8_w16a16_backbone_mpk.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rf_detr_large_w8a8_w16a16_decoder",
                     "rf_detr_large_w8a8_w16a16_decoder_mpk.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rfdetr-seg-medium-backbone",
                     "rfdetr-seg-medium-backbone.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("rfdetr-seg-medium-transformer",
                     "rfdetr-seg-medium-transformer.tar.gz",
                     "rfdetr-detection-segmentation"),
            ModelRow("efficientsam3",
                     "efficientsam3_mpk.tar.gz",
                     "efficientsam3-promptable-segmenter"),
            ModelRow("efficientsam3_text_encoder",
                     "efficientsam3_text_encoder_mpk.tar.gz",
                     "efficientsam3-promptable-segmenter"),
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


def detect_mla_clock() -> str:
    try:
        hz = int(MLA_FREQ.read_text())
    except (OSError, ValueError):
        return existing_context_value("MLA Clock", "unknown")
    return f"{hz // 1_000_000} MHz"


def resolve_model_path(row: ModelRow, models_dir: Path) -> Path:
    preferred = models_dir / row.package
    if preferred.exists():
        return preferred
    matches = sorted(models_dir.rglob(row.file))
    if not matches:
        raise FileNotFoundError(f"model package not found under {models_dir}: {row.file}")
    return matches[0]


def benchmark_command(row: ModelRow, model_path: Path, frames: int, min_duration_ms: int,
                      out_path: Path) -> list[str]:
    command = [sys.executable, str(MAIN_PY), "--model", str(model_path),
               "--frames", str(frames), "--min-duration-ms", str(min_duration_ms),
               "--output-json", str(out_path)]
    if row.decode_type:
        command += ["--decode-type", row.decode_type]
    return command


def failure_path(reports_dir: Path, row: ModelRow) -> Path:
    return reports_dir / f"{row.model_id}.failure.txt"


def record_failure(reports_dir: Path, row: ModelRow, reason: str) -> BenchmarkFailure:
    failure_path(reports_dir, row).write_text(reason + "\n", encoding="utf-8")
    print(f"[failed] {row.model_id}: {reason}", file=sys.stderr, flush=True)
    return BenchmarkFailure(row.model_id, reason)


def run_benchmarks(frames: int, min_duration_ms: int, reports_dir: Path, models_dir: Path,
                   rows: tuple[ModelRow, ...]) -> list[BenchmarkFailure]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    failures: list[BenchmarkFailure] = []
    for row in rows:
        out_path = reports_dir / f"{row.model_id}.json"
        out_path.unlink(missing_ok=True)
        failure_path(reports_dir, row).unlink(missing_ok=True)
        try:
            model_path = resolve_model_path(row, models_dir)
        except FileNotFoundError as exc:
            failures.append(record_failure(reports_dir, row, str(exc)))
            continue
        print(f"[benchmark] {row.model_id}: {model_path}", flush=True)
        result = subprocess.run(benchmark_command(row, model_path, frames, min_duration_ms,
                                                  out_path),
                                cwd=APPS_ROOT, check=False, stderr=subprocess.PIPE, text=True)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            last_line = next((line for line in reversed(result.stderr.splitlines()) if line),
                             "")
            failures.append(record_failure(reports_dir, row,
                                           f"exit {result.returncode}: {last_line}".strip()))
    return failures


def read_report(reports_dir: Path, row: ModelRow) -> dict:
    report_path = reports_dir / f"{row.model_id}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"missing benchmark report: {report_path}")
    return json.loads(report_path.read_text())


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


def result_row(reports_dir: Path, row: ModelRow) -> str:
    prefix = f"| `{row.model_id}` | `{row.file}` | {row.used_by} |"
    failure = failure_path(reports_dir, row)
    if failure.exists() and not (reports_dir / f"{row.model_id}.json").exists():
        reason = failure.read_text(encoding="utf-8").strip().replace("|", "\\|")
        return f"{prefix} Not measured: {reason} | | | |"
    report = read_report(reports_dir, row)
    model, metrics = report["model"], report["metrics"]
    specs = model["output_specs"]
    outputs = "n/a" if specs[0].startswith("unavailable:") else len(specs)
    latency = " / ".join(f"{float(metrics[key]):.2f}"
                         for key in ("latency_ms", "latency_p50_ms", "latency_p99_ms"))
    return (f"{prefix} `{model['resolved_postprocess']}` | {outputs} | {latency} | "
            f"{float(metrics['fps']):.1f} |")


def render_markdown(frames: int, min_duration_ms: int, sdk: str, mla_clock: str, run_date: str,
                    reports_dir: Path) -> str:
    lines = [
        "# Model Benchmark Results",
        "",
        "This file is a manually maintained reference table for Apps-supported compiled model "
        "packages: every package an example README downloads from the Model Zoo.",
        "The numbers come from `pyneat.Model.benchmark()` through `src/python/main.py`.",
        "",
        "These results measure the compiled model package only, through the route the package "
        "declares, with synthetic input. They do not include camera ingest, RTSP decode, Insight "
        "output, metadata conversion, overlays, application postprocessing, or decode and "
        "preprocess options an application sets.",
        "",
        "Latency is per sample from the latency pass. FPS comes from the separate throughput pass, "
        "which keeps several samples in flight.",
        "",
        "## Run Context",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Target | Modalix (`aarch64`) |",
        f"| SDK | {sdk} |",
        f"| MLA Clock | {mla_clock} |",
        f"| Date | {run_date} |",
        f"| Minimum Frames per Pass | {frames} |",
        f"| Minimum Duration per Pass | {min_duration_ms} ms |",
        "| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py "
        "--model <model-package> --frames <frames> --min-duration-ms <ms> "
        "[--decode-type <yolo26-det\\|yolo26-seg>]` |",
        "| Refresh Script | `python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py "
        "--run --models-dir <dir>` |",
        "| JSON Reports | `sandbox/model-benchmark/runs/<model-id>.json` |",
        "| Power Columns | Omitted: board telemetry measures the whole board, not the model |",
        "| BoxDecode Options | `top_k=100`; thresholds come from the model package |",
        "",
    ]
    for title, rows in GROUPS:
        lines.extend([
            f"## {title}",
            "",
        ])
        if title == "Multi-Package Model Parts":
            lines.extend([
                "Each of these models runs as two packages. Every part is measured alone, so no "
                "row is the cost of the whole model.",
                "",
            ])
        lines.extend([
            "| Model ID | Package | Used By | Postprocess | Outputs | "
            "Latency ms (mean / p50 / p99) | FPS |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ])
        lines.extend(result_row(reports_dir, row) for row in rows)
        lines.append("")
        if title == "YOLO26 Segmentation":
            lines.extend([
                "The BoxDecode row builds a 160x160 mask on the A65 for every detection it keeps, "
                "and synthetic input fills all `top_k=100` slots, so it is a worst case. With "
                "`top_k=10`, the same package measured 163.6 FPS (19.41 ms) on the same board "
                "at 1000 MHz.",
                "",
            ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run benchmarks before refreshing")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--min-duration-ms", type=int, default=30000)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="directory holding the downloaded model packages")
    parser.add_argument("--sdk", default=None)
    parser.add_argument("--mla-clock", default=None,
                        help="MLA clock of the measured board, such as '1000 MHz'")
    parser.add_argument("--date", default=None)
    parser.add_argument("--models", nargs="*", help="optional model IDs to rerun")
    parser.add_argument("--allow-partial-update", action="store_true",
                        help="refresh Markdown even when a requested benchmark fails")
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("--frames must be > 0")
    if args.min_duration_ms < 0:
        parser.error("--min-duration-ms must be >= 0")

    rows = selected_rows(args.models)
    failures: list[BenchmarkFailure] = []
    if args.run:
        failures = run_benchmarks(args.frames, args.min_duration_ms, args.reports_dir,
                                  args.models_dir, rows)

    if failures and not args.allow_partial_update:
        print("\nBenchmark failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure.model_id}: {failure.reason}", file=sys.stderr)
        failed_ids = " ".join(failure.model_id for failure in failures)
        print("\nRerun only failed model(s):", file=sys.stderr)
        print("  python3 examples/benchmarking/model-benchmark/scripts/refresh_results.py "
              f"--run --frames {args.frames} --min-duration-ms {args.min_duration_ms} "
              f"--models {failed_ids}", file=sys.stderr)
        print("BENCHMARK_RESULTS.md was not updated.", file=sys.stderr)
        return 1

    sdk = args.sdk or detect_sdk()
    mla_clock = args.mla_clock or detect_mla_clock()
    run_date = args.date or (date.today().isoformat() if args.run else report_date(args.reports_dir))
    RESULTS_MD.write_text(render_markdown(args.frames, args.min_duration_ms, sdk, mla_clock,
                                          run_date, args.reports_dir),
                          encoding="utf-8")
    print(f"updated {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
