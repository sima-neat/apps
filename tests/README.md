# SiMa NEAT Apps Testing

`tests/test.sh` is the single test entrypoint for CI, VS Code tasks, and manual
SDK runs. Build and NEAT install stay in `build.sh`.

## Quick Start

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

./build.sh
./tests/test.sh --unit
```

Run everything that can run with the available local resources:

```bash
./tests/test.sh --all
```

Run CI-like strict e2e validation:

```bash
./tests/test.sh --e2e --strict
```

## Local E2E Config

Tracked, non-secret e2e semantic parameters live in `tests/configs/e2e.yaml`.
Use it for shared thresholds, NMS values, top-k limits, and validation
expectations that should be identical between Python and C++ tests.

Machine-specific values belong in a local config file, not in tracked scripts.
`tests/configs/.env.example` is the committed template. `tests/configs/.env.local` is your
ignored local copy with real board URLs and interpreter paths.

```bash
cp tests/configs/.env.example tests/configs/.env.local
```

Edit `tests/configs/.env.local`:

```bash
SIMANEAT_APPS_TEST_RTSP_URL=rtsp://<host>:<port>/<stream>
SIMANEAT_APPS_TEST_RTSP_URLS=rtsp://<host>:<port>/<stream0>,rtsp://<host>:<port>/<stream1>
```

`tests/configs/.env.local` is ignored by git and is auto-loaded by `tests/test.sh`.
Use `--config <file>` to load a different config:

```bash
./tests/test.sh --all --config /path/to/board.env
```

Process environment variables override values loaded from the config file.

## Test Layout

```text
tests/
  test.sh              # local, VS Code, CI entrypoint
  README.md            # test workflow documentation
  pytest.ini           # pytest markers and discovery settings
  conftest.py          # pytest fixture wiring
  configs/
    e2e.yaml           # tracked non-secret e2e thresholds and validation values
    .env.example       # documented local runtime overrides
    .env.local         # ignored local runtime overrides
  utils/
    e2e_config.py      # shared Python config and output helpers
  scripts/
    testing/           # VS Code / DevKit task helpers
```

Generated e2e artifacts are written under `sandbox/test-runs` by default:

```text
sandbox/test-runs/
  python/<example>/<test>/
    config.yaml
    out/*
  cpp/<example>/<test>/
    config.yaml
    out/*
```

## Commands

```bash
# All unit tests (C++ + Python)
./tests/test.sh --unit

# All e2e tests (C++ + Python)
./tests/test.sh --e2e

# C++ only
./tests/test.sh --unit --cpp
./tests/test.sh --e2e --cpp

# Python only
./tests/test.sh --unit --python
./tests/test.sh --e2e --python

# Everything
./tests/test.sh --all

# Strict mode: missing prerequisites or skipped e2e tests fail
./tests/test.sh --e2e --strict
```

## Python Test Interpreter

Python tests run through `PYTHON_TEST_BIN`. If it is unset, `tests/test.sh`
uses common pyneat locations first, then the active virtual environment, then
system `python3`.

The selected interpreter must have `pytest` installed:

```bash
${PYTHON_TEST_BIN:-python3} -m pip install pytest
```

For a persistent local override, set `PYTHON_TEST_BIN` in `tests/configs/.env.local`.

## VS Code From SDK

Launch the VS Code tasks from the eLxr SDK workspace under `/workspace`.
The task wrapper builds in the SDK, recovers the board through `dk`, then calls
`tests/test.sh` on the board-side workspace.

```bash
bash tests/scripts/testing/run_vscode_test_task.sh --unit
bash tests/scripts/testing/run_vscode_test_task.sh --all
bash tests/scripts/testing/run_vscode_test_task.sh --all --strict
```

The same `tests/configs/.env.local` file is visible to the board when the workspace is
shared under `/workspace`.

## Run Individual Tests

Use native runners when you need one specific test.

### C++ (CTest)

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}/build"

ctest -N
ctest -R "single-stream-object-detector\.metadata_json_e2e" --verbose
```

### Python (pytest)

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

python3 -m pytest \
  examples/object-detection/multi-stream-object-detector/python/tests/test_unit.py::TestMainEntrypoint::test_missing_config_file_fails_cleanly \
  -v
```

## Environment Variables

`tests/test.sh` reads these variables:

- `SIMANEAT_APPS_TEST_MODELS_DIR` (default: `${APPS_ROOT}/assets/models`)
- `SIMANEAT_APPS_TEST_INPUT_DIR` (default: `${APPS_ROOT}/assets/test_images`)
- `SIMANEAT_APPS_TEST_OUTPUT_DIR` (default: `${APPS_ROOT}/sandbox/test-runs`)
- `SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE` (default: `${APPS_ROOT}/assets/test_images_classification/goldfish.jpeg`)
- `SIMANEAT_APPS_TEST_KEEP_OUTPUT` (`1` keeps e2e output dirs, default: `1`)
- `SIMANEAT_APPS_TEST_RTSP_URL` (single RTSP stream URL)
- `SIMANEAT_APPS_TEST_RTSP_URLS` (comma-separated RTSP URLs for multistream tests)
- `SIMANEAT_APPS_TEST_TIMEOUT_MS` (default: `180000`)
- `SIMANEAT_APPS_TEST_REQUIRE_E2E` (backward-compatible strict e2e env flag; prefer `--strict`)
- `SIMANEAT_APPS_TEST_LABELS_FILE` (optional labels file override)
- `SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT` (optional Insight video port override)
- `SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT` (optional Insight metadata port override)
- `PYTHON_TEST_BIN` (optional Python interpreter override)

## RTSP E2E Prerequisites

RTSP e2e tests require live reachable RTSP streams at test time:

- `single-stream-object-detector` (C++/Python)
- `multi-stream-object-detector` (C++/Python)

Any RTSP source works. If streams are host-served, use the host IP in the RTSP
URLs instead of `127.0.0.1`.

## Two-Stage CI

- Stage 1 (eLxr runner): `./build.sh --all --clean` builds and packages apps.
- Stage 2 (Modalix runner): installs the packaged runtime and runs `tests/test.sh`.
- Regular CI runs unit tests with `./tests/test.sh --unit`.
- Nightly/manual e2e runs `./tests/test.sh --e2e --strict`.
