# SiMa NEAT Apps

![Standard Build & Archive](https://img.shields.io/badge/Standard%20Build%20%26%20Archive-TBA-lightgrey)
![Fuzz Nightly](https://img.shields.io/badge/Fuzz%20Nightly-TBA-lightgrey)
![Crash Correctness Nightly](https://img.shields.io/badge/Crash%20Correctness%20Nightly-TBA-lightgrey)
![SDK](https://img.shields.io/badge/SDK-2.0-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

Source-first example applications for SiMa NEAT.

This repo is intentionally separate from `core`:
- `core` installs the NEAT SDK/runtime/tooling
- `apps` holds customer-facing applications and sample code

## Customer Workflow (Early Release)

1. Install NEAT/core dependencies by following the official guide:
   https://neat.modalix.info/getting-started/install

2. Clone this repo and build/run examples.

This keeps examples editable and easy to customize.

## Repo Layout

- `examples/`: examples organized by task/category (each example has C++ and Python implementations)
- `support/`: shared C++ helper code used by multiple examples
- `assets/`: user-managed runtime assets (models under `assets/models/`, test media under `assets/test_images/`)
- `tests/`: centralized test infrastructure (runner, env setup, pytest config/docs)
- `neat-core.json`: NEAT core SDK dependency declaration (branch and version)

## Build (`build.sh` only)

Install `nlohmann-json3-dev` before building:

```bash
sudo apt update
sudo apt install nlohmann-json3-dev
```

`build.sh` is only for configure/build (and optional NEAT SDK install). It does not run tests.

Common commands:

```bash
# Build only (default)
./build.sh

# Clean build directory then build
./build.sh --clean

# Install NEAT core from neat-core.json, then build
./build.sh --all

# Install NEAT core only, no build
./build.sh --only-install-neat-core

# Override neat-core.json for one run (branch:version)
./build.sh --all --neat-core-version main:latest

# Debug build
./build.sh --debug

# Build to a custom directory
./build.sh --build-dir out/build
```

Main `build.sh` args:

- `--all`: install NEAT core then build
- `--only-install-neat-core`: install NEAT core and exit
- `--neat-core-version <branch:version>`: override `neat-core.json`
- `--clean`: remove build dir before configure
- `--debug` / `--release`: build type
- `--build-dir <dir>`: build directory
- `--no-cpp`: skip C++ build
- `--python`: enable Python tooling placeholder flag in CMake config

If NEAT is installed in a non-standard prefix, set `CMAKE_PREFIX_PATH`:

```bash
CMAKE_PREFIX_PATH=/opt/sima-neat ./build.sh
```

### Cross Compilation

`build.sh` auto-selects `cmake/toolchains/aarch64-modalix.cmake` when cross
environment variables are present (`SYSROOT`, `CROSS_COMPILE`, `CC`, `CXX`).

Typical usage:

```bash
./build.sh
```

## Test (`tests/test.sh` only)

Testing is documented in [tests/README.md](./tests/README.md).

Summary:

- `build.sh` is build-only.
- `tests/test.sh` is test-only.
- For RTSP e2e tests, make sure RTSP stream source(s) are running before invoking `tests/test.sh`.

## RTSP Streams

If you want a quick RTSP source for testing, [`tool-mediasources`](https://github.com/SiMa-ai/tool-mediasources) on the host is one option:

```bash
sima-cli install gh:sima-ai/tool-mediasources
./mediasrc.sh <video-dir>
```

If you use [`tool-mediasources`](https://github.com/SiMa-ai/tool-mediasources), you can check the streams with:

```bash
open preview.html
```

If you use host-streamed sources from a board/devkit, use the host IP in the RTSP URL instead of `127.0.0.1`. Any other RTSP source also works.

## NEAT Core Dependency

`neat-core.json` declares which NEAT core SDK branch and version this repo depends on.
`./build.sh --all` reads this file and uses the hosted `install-neat-from-a-branch.sh`
installer to install the correct SDK before building.

## Support

Use the GitHub issue templates on the repo's `New issue` page and keep reports detailed.

For bug reports, include:

- a clear explanation of the bug
- the exact example, script, or command that was run
- steps to reproduce
- expected behavior and actual behavior
- the full error message, traceback, or relevant logs
- environment details such as platform, OS, SDK version, and input assets

For feature requests, include:

- exactly what needs to be added, removed, or changed
- why the change is necessary
- the examples, docs, scripts, or workflows that should change
- the expected outcome or acceptance criteria

If you are blocked or need help triaging an issue, contact us at `support@sima.ai`.
