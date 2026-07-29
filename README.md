# SiMa Neat Apps

![Vulcan CI](https://github.com/sima-neat/apps/actions/workflows/vulcan-ci.yml/badge.svg)
![Neat Development Environment](https://img.shields.io/badge/Neat%20Development%20Environment-2.1.3-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

SiMa Neat Apps provides runnable C++ and Python examples for detection,
segmentation, streaming, tracking, benchmarking, and GenAI. Install Apps to
run the packaged examples. Clone this repository only when modifying or
building their source.

## Install Apps

On a supported Modalix or DevKit target, install the latest Apps runtime from main and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Use `-d <install-root>` to install `prebuilt-apps/` under another directory. The Apps installer installs the Core selected for the Apps package and installs Insight through `sima-cli`. Run the remaining commands from the installed `prebuilt-apps/` directory.

Run the same command with the same installation root to upgrade Apps. Before replacing an existing installation, the installer asks for confirmation, carries `models/` into the new bundle, and retains the remaining previous files under `.prebuilt-apps-backup/`. If that backup already exists, the installer warns before replacing it.

## Run an Example

Each example README identifies its model, inputs, configuration, and exact run
commands.

C++ applications use the packaged executable:

```bash
./examples/<category>/<example>/src/cpp/pre-built/<binary> \
  --config examples/<category>/<example>/src/common/config.yaml
```

Python applications use their packaged entrypoint:

```bash
source ~/pyneat/bin/activate
pip install -r examples/<category>/<example>/src/python/requirements.txt
python3 examples/<category>/<example>/src/python/main.py \
  --config examples/<category>/<example>/src/common/config.yaml
```

Some Python applications provide `setup.sh` and `run.sh` wrappers instead.
Follow the selected example README.

Browse the [Neat Apps Portal](https://developer.sima.ai/examples) or the
[examples directory](./examples) to choose an example.

## Models and Datasets

Models are user-managed and are not included in the Apps package. Store them
under `models/`, or set `model.path` to another readable package. Each
example README lists its default and additional supported models with the
corresponding `sima-cli` or example-specific installation command.

Check the installed platform version to confirm the target is supported:

```bash
cat /etc/buildinfo
```

Model packages are versioned separately from the platform. They come from the
Model Zoo release, which can differ from the installed platform version.
Before downloading a model, set the Model Zoo version:

```bash
export MODELZOO_VERSION="2.1.2"
```

Runtime sample data is included under `assets/datasets/`.

## Installed Layout

| Path | Purpose |
| --- | --- |
| `examples/<category>/<example>/README.md` | Example setup and run instructions |
| `examples/<category>/<example>/src/cpp/pre-built/` | Packaged C++ executables |
| `examples/<category>/<example>/src/cpp/` | C++ implementation reference |
| `examples/<category>/<example>/src/python/` | Python implementation and dependencies |
| `examples/<category>/<example>/src/common/` | Shared runtime configuration and data |
| `assets/datasets/` | Shipped runtime sample data |
| `models/` | User-managed model packages |
| `deps/neat-apps.json` | Installed Apps ref and exact commit |
| `deps/neat-core.json` | Core artifact selected for this Apps build |

The installed bundle does not contain CMake files or tests. Clone the repository
to compile source or run the Apps test suite.

## Streaming Sources

[Insight](https://developer.sima.ai/software/tools/insight/) can host videos as
streaming sources. Install videos directly from the Insight catalog or through
YouTube support, start the required streams in the Insight Web UI, and copy
their source URLs into the example configuration. Each example README identifies
the required transport, including HTTP MJPEG where supported. Ensure each URL is
reachable from the target.

## Development From Source

Clone and build Apps inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

`build.sh` builds the repository; it does not run tests. Contributor,
scaffolding, build, and test instructions live in
[CONTRIBUTING.md](./CONTRIBUTING.md).

To fetch one standalone example without cloning the complete repository:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- <example>
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `examples/` | Application examples organized by category |
| `assets/datasets/` | Runtime datasets shipped with Apps |
| `assets/datasets-test/` | Repository-only test fixtures |
| `models/` | Ignored user-managed model packages |
| `tests/` | Test runner and repository test support |
| `deps/manifest.json` | Source dependency and platform declaration |
| `portal/` | Examples portal source |

## Dependency Selection

`deps/manifest.json` selects Core and records the SDK channel, the platform version, and the Model Zoo version used for model downloads. Development uses the `snap` policy, while a release-preparation branch can select an exact Core artifact. Vulcan records the resolved Core target under package metadata and the Apps installer uses that same target. Insight is not pinned in the Apps manifest.

## Support

Use GitHub issues for bug reports and feature requests. Include the example,
command, inputs, environment, and logs required to reproduce the problem.

For direct help, contact `support@sima.ai`.
