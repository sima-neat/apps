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

![Model benchmark preview](../../../portal/assets/examples/benchmarking/model-benchmark/image.png)

## Supported Models
This example accepts any compiled model package supported by `pyneat.Model`.

`tests/test-scope.yaml` lists the compiled model packages already used by Apps examples. `BENCHMARK_RESULTS.md` is the manually maintained table for reference benchmark results.

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- Activated `pyneat` environment.
- Model artifacts are user-managed and should be downloaded into `models/`.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) with the [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/) installed for setup and compilation.

Clone and build the apps repo inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Configure
Edit `examples/benchmarking/model-benchmark/src/common/config.yaml`, or override these values from the command line.

```yaml
model:
  path: <model-path>                          # Path to the model package.

benchmark:
  frames: 1000                                # Number of synthetic frames.

output:
  report_json: sandbox/model-benchmark/report.json # JSON report path.
```

## Run
On the Modalix/DevKit board, from the Apps repo root:

```bash
source ~/pyneat/bin/activate
pip install -r examples/benchmarking/model-benchmark/src/python/requirements.txt

python3 examples/benchmarking/model-benchmark/src/python/main.py \
  --model models/my_model.tar.gz \
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
