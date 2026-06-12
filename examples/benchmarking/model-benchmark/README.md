# Model Benchmark

## Metadata
| Field | Value |
| --- | --- |
| Category | benchmarking |
| Difficulty | Beginner |
| Tags | benchmarking, model, performance, pyneat |
| Languages | Python |
| Status | experimental |
| Binary Name | model-benchmark |
| Model | Any compiled model package |

## Concept
Before building a full camera or Insight pipeline, benchmark the compiled model package by itself. This example loads one `.tar.gz` model package, runs `pyneat.Model.benchmark()`, and writes a JSON report with latency, FPS, power, and energy.

This is a model-only synthetic benchmark. It does not measure RTSP input, camera decode, Insight video, metadata, overlays, or application postprocessing.

## Preview
The benchmark writes a report that can be compared across model packages:

![Model benchmark preview](../../../assets/portal/benchmarking/model-benchmark/image.png)

## Supported Models
This example accepts any compiled model package supported by `pyneat.Model`.

`tests/test-scope.yaml` lists the compiled model packages already used by Apps examples. `BENCHMARK_RESULTS.md` is the manually maintained table for reference benchmark results.

## Prerequisites
- Installed Neat SDK.
- Activated `pyneat` environment.
- A compiled model package available locally, usually under `assets/models/`.

## Important Behavior
- `benchmark.frames` controls the measured synthetic frames passed to `Model.benchmark()`.
- The default report path is `sandbox/model-benchmark/report.json`.
- The JSON report is the stable output for comparing model packages.

## Command-Line Options
### Python
- Invocation:
  `python3 examples/benchmarking/model-benchmark/src/python/main.py [--config <path>] [--model <path>] [--frames <n>] [--output-json <path>]`
- Optional arguments:
  `--config <path>`: YAML config path. Defaults to `src/common/config.yaml`.
  `--model <path>`: compiled model package path. Overrides `model.path`.
  `--frames <n>`: measured synthetic frames. Overrides `benchmark.frames`.
  `--output-json <path>`: report path. Overrides `output.report_json`.

## Run
From the Apps repo root:

```bash
source ~/pyneat/bin/activate
pip install -r examples/benchmarking/model-benchmark/src/python/requirements.txt

python3 examples/benchmarking/model-benchmark/src/python/main.py \
  --model assets/models/my_model.tar.gz \
  --frames 1000 \
  --output-json sandbox/model-benchmark/my_model.json
```

The command prints the headline metrics and writes the JSON report:

```text
latency_ms=...
fps=...
avg_power_watts=...
energy_joules=...
report_json=sandbox/model-benchmark/my_model.json
```

## Benchmark Results
See `BENCHMARK_RESULTS.md` for the manually maintained reference results for Apps-supported model packages.

## Debugging Notes
- If `pyneat` is missing, activate the Neat Python environment with `source ~/pyneat/bin/activate`.
- If the model path fails, confirm the `.tar.gz` file exists on the target device.
- If power is zero, the benchmark still completed; board power telemetry was unavailable for that run.

## Source Files
- Python source: `src/python/main.py`
- Python tests: `tests/python/test_unit.py`, `tests/python/test_e2e.py`
- Shared config: `src/common/config.yaml`
- Test scope: `tests/test-scope.yaml`
- Reference results: `BENCHMARK_RESULTS.md`
