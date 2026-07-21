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

Benchmark a compiled model before integrating it into a full pipeline. The example runs `pyneat.Model.benchmark()` and writes latency, FPS, power, and energy measurements to JSON.

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
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

This example accepts any compatible compiled MPK. Apps CI exercises the benchmark with `yolo26m-det-int8-b1.tar.gz`; it is not a required default.

Check the installed platform version, then set `PLATFORM_VERSION` to the displayed `DISTRO_VERSION` value. Download a Model Zoo package:

```bash
cat /etc/buildinfo
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli modelzoo -v "${PLATFORM_VERSION}" get <model-name>
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

Edit `examples/benchmarking/model-benchmark/src/common/config.yaml`, or provide the same values on the command line.

```yaml
model:
  path: <model-path>

benchmark:
  frames: 1000

output:
  report_json: sandbox/model-benchmark/report.json
```

## Run

```bash
source ~/pyneat/bin/activate
pip install -r examples/benchmarking/model-benchmark/src/python/requirements.txt
python3 examples/benchmarking/model-benchmark/src/python/main.py \
  --model models/<model-file>.tar.gz \
  --frames 1000 \
  --output-json sandbox/model-benchmark/report.json
```

The command prints headline metrics and writes the complete report to the selected JSON path.

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
