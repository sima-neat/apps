# SiMa NEAT Apps Testing

`tests/test.sh` is the single test entrypoint for CI, VS Code tasks, and manual
Neat Development Environment runs. Build and Neat Library install stay in `build.sh`.

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

E2E tests validate the shipped examples. Semantic values such as model path,
thresholds, NMS, top-k, and validation expectations live in each example's
`src/common/config.yaml`. Python and C++ e2e tests read those values and only
override harness-specific values such as input paths, output paths, finite frame
counts, RTSP URLs, and local ports.

Machine-specific values belong in a local config file, not in tracked scripts.
`tests/configs/.env.example` is the committed template. `tests/configs/.env.local` is your
ignored local copy with real board URLs and interpreter paths.

```bash
cp tests/configs/.env.example tests/configs/.env.local
```

Edit `tests/configs/.env.local`:

```bash
SIMANEAT_TEST_RTSP_H264_URL=rtsp://<host>:<port>/<stream>
SIMANEAT_TEST_RTSP_H264_URLS=rtsp://<host>:<port>/<stream0>,rtsp://<host>:<port>/<stream1>
SIMANEAT_TEST_RTSP_MJPEG_URL=rtsp://<host>:<port>/<stream>
SIMANEAT_TEST_RTSP_MJPEG_URLS=rtsp://<host>:<port>/<stream0>,rtsp://<host>:<port>/<stream1>
SIMANEAT_TEST_HTTP_MJPEG_URL=http://<host>:<port>/<stream>.mjpg
SIMANEAT_TEST_HTTP_MJPEG_URLS=http://<host>:<port>/<stream0>.mjpg,http://<host>:<port>/<stream1>.mjpg
```

`tests/configs/.env.local` is ignored by git and is auto-loaded by `tests/test.sh`.
Use `--config <file>` to load a different config:

```bash
./tests/test.sh --all --config /path/to/board.env
```

Process environment variables override values loaded from the config file.

E2E runs ensure the models selected by each example's `tests/test-scope.yaml` are
available by calling `scripts/download_models.sh`, which skips models already
present under `SIMANEAT_APPS_TEST_MODELS_DIR`. Set
`NEAT_APPS_SKIP_MODEL_DOWNLOAD=1` to disable this step.

The README `Model` row remains customer-facing metadata. Test selection and
test model downloads are controlled by `examples/*/*/tests/test-scope.yaml`, so large
or blocked examples can stay documented without blocking CI. If a test is
enabled in the scope file but the matching Python or C++ test file is missing,
`tests/test.sh` fails before running tests. If a test is disabled, it is skipped
even if a test file exists.

Package installation does not download models by default. Use
`NEAT_APPS_DOWNLOAD_MODELS_ON_INSTALL=1` only when an install step should also
preload the scoped e2e models.

## Test Layout

```text
tests/
  test.sh              # local, VS Code, CI entrypoint
  README.md            # test workflow documentation
  pytest.ini           # pytest markers and discovery settings
  conftest.py          # pytest fixture registration hook
  configs/
    .env.example       # documented local runtime overrides
    .env.local         # ignored local runtime overrides
  utils/
    e2e_config.py      # shared example-config and output helpers
    pytest_fixtures.py # shared pytest fixture implementations
    test_scope.py      # test-scope validation and query helper
  scripts/
    testing/           # VS Code / DevKit task helpers

examples/<category>/<example>/
  tests/test-scope.yaml      # enabled tests and test model download sources
```

Generated e2e artifacts are written under `sandbox-test` by default:

```text
sandbox-test/
  summary/
    cpp-e2e.log
    cpp-unit.log
    python-e2e.log
    python-unit.log
  python/<example>/<test>/
    command.txt
    config.yaml
    stdout.log
    stderr.log
    out/*
  cpp/<example>/<test>/
    command.txt
    config.yaml
    stdout.log
    stderr.log
    out/*
```

Each e2e run directory contains the generated config passed to the example, the
command that was run, captured logs, and any produced output artifacts. The
generated config is derived from the example's `src/common/config.yaml`, with only
the local test harness values patched in. Unit-test scratch files are not kept in
`sandbox-test`.

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

The selected interpreter must have `pytest` and `PyYAML` installed:

```bash
${PYTHON_TEST_BIN:-python3} -m pip install pytest PyYAML
```

For a persistent local override, set `PYTHON_TEST_BIN` in `tests/configs/.env.local`.

## VS Code From Neat Development Environment

Launch the VS Code tasks from the Neat Development Environment workspace under `/workspace`.
The task wrapper builds in the Neat Development Environment, then calls `tests/test.sh` on the
board-side workspace through `dk`.

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
  examples/object-detection/multi-stream-object-detector/tests/python/test_unit.py::TestMainEntrypoint::test_missing_config_file_fails_cleanly \
  -v
```

## Environment Variables

`tests/test.sh` reads these variables:

- `SIMANEAT_APPS_TEST_MODELS_DIR` (default: `${APPS_ROOT}/assets/models`)
- `SIMANEAT_APPS_TEST_SCOPE_FILE` (default: `${APPS_ROOT}/examples`)
- `SIMANEAT_APPS_TEST_INPUT_DIR` (default: `${APPS_ROOT}/assets/test_images`)
- `SIMANEAT_APPS_TEST_OUTPUT_DIR` (default: `${APPS_ROOT}/sandbox-test`)
- `SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE` (default: `${APPS_ROOT}/assets/test_images_classification/goldfish.jpeg`)
- `SIMANEAT_APPS_TEST_KEEP_OUTPUT` (`1` keeps e2e output dirs, default: `1`)
- `SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS` (`1` writes summary logs, default: `1`)
- `SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS` (`1` writes per-example command/stdout/stderr logs, default: `1`)
- `SIMANEAT_TEST_RTSP_H264_URL` (single RTSP H.264 stream URL)
- `SIMANEAT_TEST_RTSP_H264_URLS` (comma-separated RTSP H.264 URLs)
- `SIMANEAT_TEST_RTSP_MJPEG_URL` (single RTSP MJPEG stream URL)
- `SIMANEAT_TEST_RTSP_MJPEG_URLS` (comma-separated RTSP MJPEG URLs)
- `SIMANEAT_TEST_HTTP_MJPEG_URL` (single HTTP MJPEG stream URL)
- `SIMANEAT_TEST_HTTP_MJPEG_URLS` (comma-separated HTTP MJPEG URLs)
- `SIMANEAT_APPS_TEST_TIMEOUT_MS` (default: `180000`)
- `SIMANEAT_APPS_TEST_REQUIRE_E2E` (backward-compatible strict e2e env flag; prefer `--strict`)
- `SIMANEAT_APPS_TEST_LABELS_FILE` (optional labels file override)
- `SIMANEAT_APPS_TEST_INSIGHT_HOST` (default: `127.0.0.1`)
- `SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT` (optional Insight video port override)
- `SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT` (optional Insight metadata port override)
- `NEAT_APPS_SKIP_MODEL_DOWNLOAD` (`1` skips model download before e2e, default: `0`)
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
