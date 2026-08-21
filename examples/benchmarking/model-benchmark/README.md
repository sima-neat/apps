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

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- A compiled model package supported by `pyneat.Model`.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/benchmarking/model-benchmark
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Use any compatible compiled MPK. Apps CI exercises the benchmark with `yolo26m-det-int8-b1.tar.gz`; it is not a required default.

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Download a Model Zoo package:

```bash
export MODELZOO_VERSION="2.1.3"
mkdir -p models
cd models
sima-cli modelzoo -v "${MODELZOO_VERSION}" get <model-name>
cd ..
```

For a model published as a direct artifact:

```bash
mkdir -p models
cd models
sima-cli download <model-url>
cd ..
```

## Configure

Open `${APP_DIR}/src/common/config.yaml` and set `model.path`. You can also change the number of benchmark frames and the JSON report path, or pass those values on the command line.

## Run

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --model models/<model-file>.tar.gz \
  --frames 1000 \
  --output-json sandbox/model-benchmark/report.json
```

The command prints headline metrics and writes the complete report to the selected JSON path.

Without `--decode-type`, the model runs through the route its package declares. Add `--decode-type yolo26-det` or `--decode-type yolo26-seg` to benchmark a YOLO26 package through its BoxDecode route instead. Every report records the requested decode type and the postprocess Core resolved, so the two routes stay distinguishable.

## Benchmark Results

See the maintained [benchmark results](https://github.com/sima-neat/apps/blob/main/examples/benchmarking/model-benchmark/BENCHMARK_RESULTS.md) for measurements from Apps-supported packages.

## Troubleshooting

- Activate `~/pyneat` if the `pyneat` module is unavailable.
- Confirm the model path points to a compiled `.tar.gz` package.
- A zero power result means board power telemetry was unavailable; the model benchmark still completed.

## Source Files

- Python source: `src/python/main.py`
- Shared config: `src/common/config.yaml`

## Development From Source

To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
