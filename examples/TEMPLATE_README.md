# <Example Name>

## Metadata

| Field | Value |
| --- | --- |
| Category | <benchmarking / classification / object-detection / tracking / face-detection / segmentation / pose-estimation / depth-estimation / feature-extraction / genai / multi-model / throughput / docker> |
| Difficulty | <Beginner / Intermediate / Advanced> |
| Tags | <comma-separated tags> |
| Languages | C++, Python |
| Status | stable |
| Binary Name | <binary> |
| Model | <default-model> |

## Concept

<In one or two plain-English sentences, say what the application does and produces. Name the main model or model family when useful. Keep this paragraph under 200 characters because the portal uses it as the card summary.>

## Preview

Add an application-specific image under `portal/assets/examples/<category>/<example>/`.

```md
![Demo screenshot](../../../portal/assets/examples/<category>/<example>/image.png)
```

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on the Modalix DevKit, or in a Neat SDK container paired with it.
- <Example-specific input, service, or hardware requirements.>

## Install Apps

Install on the Modalix DevKit, or under `/workspace` in a Neat SDK container paired with it. Only the example itself must run on the DevKit. `sima-cli neat install apps` also offers to install the compatible Neat Library and Insight, and skips them when they are already installed. To keep the Neat Library and Insight you have without being asked, run the install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps`. See [Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/<category>/<example>
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The default model is `<default-model>`.

| Model file | Role |
| --- | --- |
| `<default-model>` | Default |
| `<supported-model>` | Supported |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Replace `<model-file>` with a file from the table.

```bash
export MODELZOO_VERSION="3.0.0"
mkdir -p models
cd models
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/<model-file>"
cd ..
```

Set `model.path` in the example config to the downloaded package.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Name only the values users must change, such as the model path, input, output, stream URLs, or Insight host.

Do not copy the packaged configuration into the README. Keep a focused snippet only when the workflow creates an override or generated configuration that the package does not already provide.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/<binary> \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Troubleshooting

- Confirm `model.path` points to a readable model package.
- Confirm input paths exist and output directories are writable.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared runtime files: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
