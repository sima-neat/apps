# SiMa Neat Apps

![Vulcan CI](https://github.com/sima-neat/apps/actions/workflows/vulcan-ci.yml/badge.svg)
![Neat Development Environment](https://img.shields.io/badge/Neat%20Development%20Environment-2.1.2-green)
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

The Apps installer installs the Core selected for the Apps package and installs Insight through `sima-cli`. The installed bundle is placed under `prebuilt-apps/`. Run the remaining commands from that directory.

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

Check the installed platform version:

```bash
cat /etc/buildinfo
```

Before downloading a model, set `PLATFORM_VERSION` to the displayed `DISTRO_VERSION` value. Runtime sample data is included under `assets/datasets/`.

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

The installed bundle does not contain CMake files or tests. Clone the repository
to compile source or run the Apps test suite.

## RTSP Streams

[Insight](https://developer.sima.ai/software/tools/insight/) can host videos as
RTSP sources. Install videos directly from the Insight catalog or through
YouTube support, start the sources in the Insight Web UI, and copy their RTSP
URLs into the example configuration. Ensure each URL is reachable from the
target.

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

`deps/manifest.json` selects Core and records the SDK channel and platform version. Development uses the `snap` policy, while a release-preparation branch can select an exact Core artifact. Vulcan records the resolved Core target under package metadata and the Apps installer uses that same target. Insight is not pinned in the Apps manifest.

## Support

Use GitHub issues for bug reports and feature requests. Include the example,
command, inputs, environment, and logs required to reproduce the problem.

For direct help, contact `support@sima.ai`.
