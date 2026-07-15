# SiMa Neat Apps

![Vulcan CI](https://github.com/sima-neat/apps/actions/workflows/vulcan-ci.yml/badge.svg)
![Neat Development Environment](https://img.shields.io/badge/Neat%20Development%20Environment-2.1.2-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

SiMa Neat Apps is a source-first collection of editable applications and reference
examples for real Modalix workflows: detection, segmentation, streaming,
tracking, benchmarking, and GenAI.

Use this repo when you want runnable examples that keep the important Neat
Library C++ and Python API calls visible in the source. Use the Neat Library
docs for the runtime API contract, and use this repo for working application
patterns.

`core` owns the Neat Library runtime and tooling. `apps` owns customer-facing
examples and sample application code.

## Start

Install the Neat Development Environment and Neat Library first:

- [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/install-the-environment)
- [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/install-or-update)
- [sima-cli](https://developer.sima.ai/software/tools/sima-cli/)

For Neat Development Environment 2.1.2:

```bash
sima-cli install ghcr:sima-neat/sdk:v2.1.2
sima-cli sdk neat
```

Clone and build this repo from the SDK shell:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

Python examples use the PyNeat environment installed with the Neat Library:

```bash
source ~/pyneat/bin/activate
```

For the public examples index, use the
[Neat Apps Portal](https://developer.sima.ai/examples). For broader application
development docs, use [Develop Apps](https://developer.sima.ai/software/develop-apps/).

## Fetch One Example

To fetch one example without cloning the full repo:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- <example>
```

Example:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- multimodal-assistant
cd multimodal-assistant
./setup.sh
./run.sh
```

## Repo Layout

| Path | Purpose |
| --- | --- |
| `examples/` | Application examples organized by category |
| `assets/` | Runtime assets, including models and test media |
| `tests/` | Test runner, fixtures, and test documentation |
| `deps/manifest.json` | Neat Library and platform dependency declaration |
| `portal/` | Source for the examples portal |

## Examples

Examples live under `examples/<category>/<example>/`.

The common shape is:

| Path | Purpose |
| --- | --- |
| `README.md` | Example-specific setup and run instructions |
| `src/common/` | Shared config, labels, and assets |
| `src/cpp/main.cpp` | C++ entrypoint, when present |
| `src/python/main.py` | Python entrypoint, when present |
| `tests/` | Example-specific unit and e2e tests |

Start with the example README. It is the source for model downloads, runtime
inputs, config edits, and run commands for that example.

Contributor details live in [CONTRIBUTING.md](./CONTRIBUTING.md). The README
template lives at [examples/TEMPLATE_README.md](./examples/TEMPLATE_README.md).

## Build

`build.sh` builds the repo. It does not run tests.

Common commands:

```bash
./build.sh          # configure and build
./build.sh --clean  # clean build directory first
```

Compile C++ examples in the Neat Development Environment. Run Neat runtime and
e2e checks on SiMa hardware such as Modalix or DevKit.

## RTSP Streams

For quick RTSP sources, use
[`tool-mediasources`](https://github.com/SiMa-ai/tool-mediasources):

```bash
sima-cli install gh:sima-ai/tool-mediasources
./mediasrc.sh <video-dir>
```

If a board or DevKit consumes host-streamed RTSP sources, use the host IP in the
RTSP URL instead of `127.0.0.1`.

For fixed 16-, 24-, and 48-stream RTSP-to-Insight profiles, see the
[high-density multi-stream object detector](examples/object-detection/high-density-multi-stream-object-detector/README.md).

## Neat Library Dependency

`deps/manifest.json` declares the Neat Library dependency and platform version.

The normal branch value is:

```json
{
  "neat-core": {
    "policy": "snap"
  }
}
```

`snap` resolves from the dependency branch: `main` uses `main-latest`, `develop`
uses `develop-latest`, and custom branches use a matching core branch when one
exists or fall back to `develop-latest`.

## Support

Use GitHub issues for bug reports and feature requests. Include the exact
example, command, inputs, environment, and logs needed to reproduce the issue.

For direct help, contact `support@sima.ai`.
