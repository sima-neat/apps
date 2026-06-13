# SiMa Neat Apps

![Standard Build & Archive](https://img.shields.io/badge/Standard%20Build%20%26%20Archive-TBA-lightgrey)
![Fuzz Nightly](https://img.shields.io/badge/Fuzz%20Nightly-TBA-lightgrey)
![Crash Correctness Nightly](https://img.shields.io/badge/Crash%20Correctness%20Nightly-TBA-lightgrey)
![Neat Development Environment](https://img.shields.io/badge/Neat%20Development%20Environment-2.0-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

Neat Apps contains editable Neat applications and reference examples built around real workflows such as detection, segmentation, and streaming pipelines. The goal is to make the apps easy to run, easy to modify, and easy to learn from.

The core idea is simple: the Neat Library provides the C++ and Python runtime APIs, while `apps` shows how to use them in customer-facing examples where the important Neat Library API calls stay visible in the source.

This repo is intentionally separate from `core`:
- `core` owns the Neat Library runtime and tooling
- `apps` holds customer-facing applications and sample code

## Customer Workflow (Early Release)

1. Install the Neat Library by following the official [Neat Library installation guide](https://developer.sima.ai/software/getting-started/installation/neat-library).

2. Clone this repo:

   ```bash
   git clone https://github.com/sima-neat/apps.git
   cd apps
   ```

3. `sima-cli` is used by some examples and tools in this repo. Install it by following the official [sima-cli guide](https://developer.sima.ai/software/tools/sima-cli/).

4. For first-time setup, run:

   ```bash
   sudo apt update
   sudo apt-get install -y nlohmann-json3-dev nodejs npm
   ./build.sh --clean
   ```

   This configures and builds the apps from the local checkout. Follow each example README for model downloads and run commands.

   **Note:** If the Neat Library was installed into the default `pyneat` virtual environment, activate it with `source ~/pyneat/bin/activate` to run Python examples.

5. For broader Neat Library concepts and platform documentation, see:
   https://developer.sima.ai/

This keeps examples editable and easy to customize.

### Fetch A Single Example

Users who only need one example can fetch it with the helper script:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- <example>
```

For example:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- multimodal-assistant
cd multimodal-assistant
./setup.sh
./run.sh
```

The helper fetches the apps archive and leaves only the requested example
directory in the current working directory.

## Repo Layout

- `examples/`: examples organized by task/category (each example has C++ and Python implementations)
- `support/`: shared C++ helper code used by multiple examples
- `assets/`: user-managed runtime assets (models under `assets/models/`, test media under `assets/test_images/`)
- `tests/`: centralized test infrastructure (runner, env setup, pytest config/docs)
- `deps/manifest.json`: Neat Library and platform-version dependency declaration

## Example Structure

Examples are organized under `examples/<category>/<example>`. Each example is source-first and meant to be readable from the entrypoints, with the main application flow visible in both `src/cpp/main.cpp` and `src/python/main.py`.

### Required Layout

| Path | Purpose |
| --- | --- |
| `examples/<category>/<example>/README.md` | Example-specific usage and setup instructions |
| `examples/<category>/<example>/tests/test-scope.yaml` | Internal test selection and e2e model source metadata |
| `examples/<category>/<example>/src/cpp/CMakeLists.txt` | C++ example build configuration |
| `examples/<category>/<example>/src/cpp/main.cpp` | C++ entrypoint with visible Neat Library API flow |
| `examples/<category>/<example>/tests/cpp/test_unit.cpp` | C++ unit tests |
| `examples/<category>/<example>/tests/cpp/test_e2e.cpp` | C++ end-to-end tests |
| `examples/<category>/<example>/src/python/main.py` | Python entrypoint with visible Neat Library API flow |
| `examples/<category>/<example>/src/python/requirements.txt` | Python example dependencies |
| `examples/<category>/<example>/tests/python/test_unit.py` | Python unit tests |
| `examples/<category>/<example>/tests/python/test_e2e.py` | Python end-to-end tests |
| `examples/<category>/<example>/src/common/` | Files shared by the C++ and Python implementations |

> **IMPORTANT:** Use this structure when reading, extending, or adding examples.
> Follow the instructions inside each example `README.md`, or visit the [SiMa.ai Neat Apps Portal](<https://apps.sima-neat.com/portal/index.html>).

### Contributor Guidelines

For the full contributor rules, required layout, and authoring expectations, see [guidelines.md](./guidelines.md).

## Build (`build.sh`)

`build.sh` has three main modes:

- build/configure the apps locally
- install the Neat Library dependency only
- run the full release/package flow with `--all`, which installs the Neat Library dependency, builds the apps, builds the portal, and packages `neat-apps-runtime`

It does not run tests.

Common commands:

```bash
# Full release/package flow
./build.sh --all --clean

# Build only (default)
./build.sh

# Clean build directory then build
./build.sh --clean

# Install Neat Library dependency, build apps, build portal, then package
./build.sh --all

# Install Neat Library dependency only, no build
./build.sh --only-install-neat-core

# Override deps/manifest.json for one run (branch-version)
./build.sh --all --neat-core-version main-latest

# Debug build
./build.sh --debug

# Build to a custom directory
./build.sh --build-dir out/build
```

Main `build.sh` args:

- `--all`: install Neat Library dependency, build apps, build portal, then package
- `--only-install-neat-core`: install Neat Library dependency and exit
- `--neat-core-version <branch-version>`: override `deps/manifest.json`
- `--clean`: remove build dir before configure
- `--debug` / `--release`: build type
- `--build-dir <dir>`: build directory
- `--no-cpp`: skip C++ example build (layout/metadata only)

When `--all` is used, the script also:

- builds `portal/` with `npm` for local testing
- stages a distributable `neat-apps-runtime/` tree
- writes a packaged `neat-core.json` into that staged runtime
- creates `neat-apps-<branch>-<sha>.tar.gz` at the repo root

If Neat is installed in a custom location and CMake cannot find it automatically, set `CMAKE_PREFIX_PATH`:

```bash
CMAKE_PREFIX_PATH=/path/to/neat/install ./build.sh
```

### Cross Compilation

`build.sh` auto-selects `cmake/toolchains/aarch64-modalix.cmake` when cross
environment variables are present (`SYSROOT`, `CROSS_COMPILE`, `CC`, `CXX`).
Set `CMAKE_TOOLCHAIN_FILE` explicitly if you want to override that default.

Typical usage:

```bash
./build.sh
```

## Test (`tests/test.sh` only)

Testing is documented in [tests/README.md](./tests/README.md).

Summary:

- `build.sh` does not run tests.
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

## Neat Library Dependency

`deps/manifest.json` declares the Neat Library dependency and the platform version.
The normal value is `"neat-core": {"policy": "snap"}`. Snap resolves from the
dependency branch: `main` uses `main-latest`, `develop` uses `develop-latest`,
and custom branches use a matching core branch when available or fall back to
`develop-latest`. Explicit manifest pins can use `{"branch": "...", "spec": "..."}`
or `{"ref": "...", "spec": "..."}`. Legacy string pins and the
`--neat-core-version <branch-version>` override remain supported.

## Support

Use the GitHub issue templates on the repo's `New issue` page and keep reports detailed.

For bug reports, include:

- a clear explanation of the bug
- the exact example, script, or command that was run
- steps to reproduce
- expected behavior and actual behavior
- the full error message, traceback, or relevant logs
- environment details such as platform, OS, Neat Development Environment version, and input assets

For feature requests, include:

- exactly what needs to be added, removed, or changed
- why the change is necessary
- the examples, docs, scripts, or workflows that should change
- the expected outcome or acceptance criteria

If you are blocked or need help triaging an issue, contact us at `support@sima.ai`.
