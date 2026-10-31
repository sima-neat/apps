# Model Benchmark

## Metadata

| Field | Value |
| --- | --- |
| Category | benchmarking |
| Difficulty | Beginner |
| Tags | benchmarking, model, performance, pyneat |
| Languages | Python |
| Status | stable |
| Binary Name | model-benchmark |
| Model | Any compiled model package |

## Concept

Measures latency, throughput, power, and energy for any compiled model package, then saves the results as JSON.

This synthetic model benchmark does not measure input decoding, Insight output, overlays, or application postprocessing.

## Preview

![Model benchmark preview](../../../portal/assets/examples/benchmarking/model-benchmark/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- A compiled model package supported by `pyneat.Model`.

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. To leave the installed Core and Insight unchanged, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/benchmarking/model-benchmark
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Use any compatible compiled MPK. Apps CI exercises the benchmark with `yolo26m-det-int8-b1.tar.gz`; it is not a required default.

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with the package to benchmark:

```bash
export MODELZOO_VERSION="3.0.0"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/<model-file>"
cd ..
```

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`. You can also change the minimum frames and duration of each benchmark pass and the JSON report path, or pass those values on the command line.

## Run

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --model models/<model-file>.tar.gz \
  --frames 100 \
  --min-duration-ms 30000 \
  --output-json sandbox/model-benchmark/report.json
```

The command prints headline metrics and writes the complete report to the selected JSON path.

The benchmark runs a latency pass and a throughput pass. Each pass continues until it has measured at least `--frames` samples and run for at least `--min-duration-ms` milliseconds (30000 by default, `benchmark.min_duration_ms` in the config). Pass `--min-duration-ms 0` for a quick check that stops after `--frames` samples.

Without `--decode-type`, the model runs through the route its package declares. Add `--decode-type yolo26-det` or `--decode-type yolo26-seg` to benchmark a YOLO26 package through its BoxDecode route instead. Every report records the requested decode type and the postprocess Core resolved, so the two routes stay distinguishable.

## Benchmark Results

See the maintained [benchmark results](https://github.com/sima-neat/apps/blob/main/examples/benchmarking/model-benchmark/BENCHMARK_RESULTS.md) for measurements from Apps-supported packages.

## Expected Result

The application prints headline metrics and the path it wrote the full report to:

```text
latency_ms=6.520167780941697
latency_p99_ms=7.116750399999998
fps=334.74326851174976
avg_power_watts=14.64375
energy_joules=4.374621203020843
report_json=sandbox/model-benchmark/report.json
```

The `report_json=` line names where the full report was written; trust it rather
than either setting, because `--output-json` takes precedence over
`output.report_json` and the documented run above passes that flag.

These numbers depend on the model package and the board, so treat them as an
example of the shape rather than a target; compare your own against the
maintained benchmark results linked in the `Benchmark Results` section above.

## Troubleshooting

- Activate `~/pyneat` if the `pyneat` module is unavailable.
- Confirm the model path points to a compiled `.tar.gz` package.
- A zero power result means board power telemetry was unavailable; the model benchmark still completed.
- If results fall well below the maintained table, check the MLA clock with `cat /sys/class/devfreq/5000000.mla/cur_freq`. The table was measured at 1000 MHz; a board that boots at 600 MHz measures about 40% lower FPS.

## Source Files

- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
