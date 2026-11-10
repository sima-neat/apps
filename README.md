<p align="center">
  <a href="https://developer.sima.ai/examples"><img src="portal/assets/readme/hero.png" width="100%" alt="Outputs of Neat Apps examples: object detection, depth estimation, promptable segmentation, 3D pose estimation, instance segmentation, PCB defect detection, and drone tracking"></a>
</p>

<h1 align="center">SiMa.ai Neat Apps</h1>

<p align="center">
  Ready-to-run computer vision and GenAI applications for SiMa.ai Modalix,<br>
  written in C++ and Python on the Neat Library.
</p>

<p align="center">
  <a href="https://github.com/sima-neat/apps/actions/workflows/vulcan-ci.yml"><img src="https://github.com/sima-neat/apps/actions/workflows/vulcan-ci.yml/badge.svg" alt="Vulcan CI"></a>
  <img src="https://img.shields.io/badge/Neat%20Development%20Environment-3.0.0-green" alt="Neat Development Environment 3.0.0">
  <img src="https://img.shields.io/badge/C%2B%2B-20-informational" alt="C++ 20">
  <img src="https://img.shields.io/badge/Python-3-informational" alt="Python 3">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License Apache 2.0"></a>
</p>

<p align="center">
  <a href="https://developer.sima.ai/examples">Apps Portal</a> ·
  <a href="https://developer.sima.ai/software/getting-started/">Neat Documentation</a> ·
  <a href="https://developer.sima.ai/software/tools/sima-cli/">sima-cli</a> ·
  <a href="https://developer.sima.ai/software/tools/insight/">Insight</a> ·
  <a href="https://github.com/sima-neat/apps/issues">Issues</a>
</p>

One installed bundle contains every application. Most have a pre-built C++
executable and a Python entrypoint, with all settings in one `config.yaml`.
Each has a README with the model, the run commands, and the expected result.
Install the bundle to run the examples. Clone this repository only to change
or build the source.

## Quick Start

Run the first example on a Modalix DevKit with internet access. For a DevKit
without internet access, see [Install Apps](#install-apps).

1. Install Apps and enter the bundle. The installer also installs the matching
   Core and Insight.

   ```bash
   sima-cli neat install apps
   cd prebuilt-apps
   APP_DIR=examples/object-detection/yolo26-object-detector
   ```

2. Download the YOLO26 model. Downloads need your SiMa Developer Portal
   login. Run `sima-cli login` once, or follow the prompt the first download
   prints.

   ```bash
   export MODELZOO_VERSION="3.0.0"
   mkdir -p models
   cd models
   sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/yolo26-detection/yolo26m-det-bf16-mla_tess-b1.tar.gz"
   cd ..
   ```

3. Point the configuration at the model. The input images and the output
   folder are already set.

   ```bash
   sed -i 's#<model-path>#models/yolo26m-det-bf16-mla_tess-b1.tar.gz#' ${APP_DIR}/src/common/config.yaml
   ```

4. Run the detector.

   ```bash
   ./${APP_DIR}/src/cpp/pre-built/yolo26-object-detector \
     --config ${APP_DIR}/src/common/config.yaml
   ```

   It writes 21 annotated images to `sandbox/yolo26-object-detector/`. The
   [example README](examples/object-detection/yolo26-object-detector/README.md)
   covers the Python entrypoint, the other supported models, and
   troubleshooting.

## Examples

Browse all applications in the [Apps Portal](https://developer.sima.ai/examples),
with previews, search, and filters by category and model. The sources are in
the [examples directory](./examples), one folder per application.

## Inside an Application

Examples follow one layout:

```text
examples/<category>/<example>/
├── README.md                model, settings, run commands, expected result
├── src/common/config.yaml   all settings: model, paths, hosts, ports, thresholds
├── src/common/              labels and data shared by C++ and Python
├── src/cpp/                 C++ source, plus pre-built/<binary> in the bundle
└── src/python/              Python source and requirements.txt
```

Each entrypoint, `src/cpp/main.cpp` or `src/python/main.py`, shows graph
construction, model loading, execution, and teardown with the public Neat API
and no wrappers. The C++ and Python implementations use the same models,
thresholds, and outputs. Applications with live output send video and metadata
to [Insight](https://developer.sima.ai/software/tools/insight/), the Neat
visualization UI.

## Install Apps

Install on the Modalix DevKit. If the DevKit has no internet access, install
under `/workspace` in a Neat SDK container paired with it. Pairing shares
`/workspace` over NFS, so the bundle appears at the same path on the DevKit.
The examples always run on the DevKit.

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Add `-d <install-root>` to install `prebuilt-apps/` under another directory.
Run all further commands from `prebuilt-apps/`.

The installer also installs the Core version this Apps package needs and the
latest Insight. When run from a paired SDK container, it installs Core on the
DevKit too.

A paired DevKit already has the SDK's Core, and Insight runs in the SDK
container. If that Core is the version this package needs, skip both installs:

```bash
NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps
```

The installer does not check that the versions match. A mismatch causes errors
when an example runs. See
[Dependency Selection](CONTRIBUTING.md#dependency-selection).

To upgrade, run the same command with the same install root. The installer
asks before replacing an installation, keeps `models/`, and moves the other
old files to `.prebuilt-apps-backup/`. If that backup already exists, it warns
before replacing it.

## Models and Datasets

Models are not included in the Apps package. Store them under `models/`, or
set `model.path` in the example's `config.yaml` to another location. Each
example README lists its default and supported models with the download
command.

Check the platform version installed on the DevKit:

```bash
cat /etc/buildinfo
```

Model packages come from the Model Zoo release, whose version can differ from
the platform version. Downloads need your SiMa Developer Portal login. Set the
Model Zoo version before downloading:

```bash
export MODELZOO_VERSION="3.0.0"
```

Sample data ships under `assets/datasets/`.

## Streaming Sources

[Insight](https://developer.sima.ai/software/tools/insight/) hosts videos as
streaming sources. Install videos from the Insight catalog or through its
YouTube support, start the streams in the Insight Web UI, and copy their source
URLs into the example configuration. Each example README names the transport
it needs, such as HTTP MJPEG. The DevKit must be able to reach each URL.

## Build From Source

Build inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

`build.sh` does not run tests. To fetch one example without cloning the
repository:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- <example>
```

Layouts, dependency selection, the example scaffold, tests, and the review
checklist are in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Support

Report bugs and request features in GitHub issues. Include the example,
command, inputs, environment, and logs needed to reproduce the problem. For
direct help, contact `support@sima.ai`.

## License

Neat Apps is released under the [Apache License 2.0](./LICENSE).
