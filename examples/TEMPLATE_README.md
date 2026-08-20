# <Example Name>

## Metadata

| Field | Value |
| --- | --- |
| Category | <benchmarking / classification / object-detection / tracking / face-detection / segmentation / pose-estimation / depth-estimation / feature-extraction / genai / throughput> |
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

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- <Example-specific input, service, or hardware requirements.>

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/<category>/<example>
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The default model is `<default-model>`.

| Model | Role | Source |
| --- | --- | --- |
| `<default-model>` | Default | <Model Zoo / direct artifact> |
| `<supported-model>` | Supported | <Model Zoo / direct artifact> |

Model packages come from the Model Zoo release below, which can differ from the installed platform version. Use the command that matches the model source and delete the other command.

Model Zoo:

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models
cd models
sima-cli modelzoo -v "${MODELZOO_VERSION}" get <model-name>
cd ..
```

Direct artifact:

```bash
mkdir -p models
cd models
sima-cli download <model-url>
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
