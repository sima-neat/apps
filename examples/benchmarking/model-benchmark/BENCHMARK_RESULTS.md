# Model Benchmark Results

This is the readable summary for model-only synthetic benchmarks from `pyneat.Model.benchmark()`.
For sorting, filtering, or spreadsheet import, use `benchmark-results.csv` in this directory.

These results measure the compiled model package only. They do not include camera ingest, RTSP decode, Insight output, metadata conversion, overlays, or application postprocessing.

## Run Context

| Field | Value |
| --- | --- |
| Command | `python3 examples/benchmarking/model-benchmark/src/python/main.py --model <model-package>` |
| Config | `examples/benchmarking/model-benchmark/src/common/config.yaml` |
| Frames | 1000 |
| JSON Report | `sandbox/model-benchmark/report.json` |
| Spreadsheet Table | `examples/benchmarking/model-benchmark/benchmark-results.csv` |
| Power Columns | Omitted |
| Status | Pending: local DevKit `192.168.2.7:22` was unreachable, so benchmark runs did not start. |

## Result Table

Fill `benchmark-results.csv` after each benchmark run. Keep this Markdown file as the customer-facing summary and use the CSV as the table source when many rows need sorting or spreadsheet review.

| Family | Models | Status |
| --- | ---: | --- |
| General | 5 | Pending |
| YOLO26 detection | 7 | Pending |
| YOLO26 segmentation | 8 | Pending |

## CSV Columns

| Column | Meaning |
| --- | --- |
| `model_id` | Stable model ID from `tests/test-scope.yaml`. |
| `package` | Compiled model package filename. |
| `family` | Model family used for grouping and filtering. |
| `used_by` | Apps examples that currently use or support the model. |
| `target` | Target board or runtime used for the benchmark. |
| `sdk` | SDK version used for the benchmark. |
| `frames` | Measured synthetic frames passed to `Model.benchmark()`. |
| `latency_ms` | Reported model latency in milliseconds. |
| `fps` | Reported model throughput. |
| `date` | Benchmark date in `YYYY-MM-DD` format. |

## Updating Results

1. Run `main.py` for one compiled model package.
2. Read `sandbox/model-benchmark/report.json`.
3. Copy `latency_ms` and `fps` into `benchmark-results.csv`.
4. Record the target, SDK, frame count, and date in the same CSV row.
5. Keep power values out of the CSV unless the benchmark policy changes.
