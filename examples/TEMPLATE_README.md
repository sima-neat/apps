# <Example Name>

## Metadata

| Field | Value |
| --- | --- |
| Category | <benchmarking / classification / object-detection / tracking / face-detection / segmentation / pose-estimation / depth-estimation / genai / throughput> |
| Difficulty | <Beginner / Intermediate / Advanced> |
| Tags | <comma-separated tags> |
| Languages | C++, Python |
| Status | <experimental / stable> |
| Binary Name | <binary> |
| Model | <default-model> |

## Concept

<Explain what the example demonstrates and which Neat Library capabilities it uses.>

## Preview

Optional. Place portal images under `portal/assets/examples/<category>/<example>/`.

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
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The default model is `<default-model>`.

| Model | Role | Source |
| --- | --- | --- |
| `<default-model>` | Default | <Model Zoo / direct artifact> |
| `<supported-model>` | Supported | <Model Zoo / direct artifact> |

Check the installed platform version, then set `PLATFORM_VERSION` to the displayed `DISTRO_VERSION` value. Use the command that matches the model source and delete the other command.

Model Zoo:

```bash
cat /etc/buildinfo
export PLATFORM_VERSION="<platform-version>"
mkdir -p models
cd models
sima-cli modelzoo -v "${PLATFORM_VERSION}" get <model-name>
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

Edit `examples/<category>/<example>/src/common/config.yaml`.

```yaml
model:
  path: <model-path>

io:
  input_dir: assets/datasets/coco
  output_dir: sandbox/<example>
```

## Run

### C++

```bash
./examples/<category>/<example>/src/cpp/pre-built/<binary> \
  --config examples/<category>/<example>/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/<category>/<example>/src/python/requirements.txt
python3 examples/<category>/<example>/src/python/main.py \
  --config examples/<category>/<example>/src/common/config.yaml
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
