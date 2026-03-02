# SiMa NEAT Apps Testing

This document covers test execution only. Build and NEAT install stay in `build.sh`.

## Scope

- `build.sh` is build-only.
- `tests/test.sh` is test-only.
- Run `build.sh` first, then run `tests/test.sh`.

## Quick Start

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

# Build first
./build.sh

# Run all tests
./tests/test.sh --all
```

## Auto-Setup Test Env

Use the helper script to export all common test variables in one step.

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"
source tests/scripts/testing/setup_test_env.sh
```

Then override only what you need:

```bash
export SIMANEAT_APPS_TEST_OUTPUT_DIR="${APPS_ROOT}/sandbox/test-output"
export SIMANEAT_APPS_TEST_KEEP_OUTPUT=1
export SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE="${APPS_ROOT}/assets/test_images_classification/goldfish.jpeg"
export SIMANEAT_APPS_TEST_RTSP_URL="rtsp://127.0.0.1:8555/stream0"
export SIMANEAT_APPS_TEST_RTSP_URLS="rtsp://127.0.0.1:8555/stream0,rtsp://127.0.0.1:8555/stream1"
export SIMANEAT_APPS_TEST_OPTIVIEW_VIDEO_PORT=19000
export SIMANEAT_APPS_TEST_OPTIVIEW_JSON_PORT=19100
```

## `tests/test.sh` Commands

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

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
```

## Run Individual Tests

Use native runners when you need one specific test.

### C++ (CTest)

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}/build"

# List registered C++ tests
ctest -N

# Run one C++ test by name regex
ctest -R "single-rtsp-object-detection-optiview\.optiview_json_e2e" --verbose
```

### Python (pytest)

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

# Run one Python test node
python3 -m pytest \
  examples/object-detection/multistream-rtsp-detection-pipeline/tests/python/test_unit.py::TestArgParsing::test_missing_model \
  -v
```

## Environment Variables

`tests/test.sh` reads these variables during e2e preflight:

- `SIMANEAT_APPS_TEST_MODELS_DIR` (default: `${APPS_ROOT}/assets/models`)
- `SIMANEAT_APPS_TEST_INPUT_DIR` (default: `${APPS_ROOT}/assets/test_images`)
- `SIMANEAT_APPS_TEST_OUTPUT_DIR` (default: `/tmp`; setup script default: `${APPS_ROOT}/sandbox/tests`)
  - C++ e2e output root: `${SIMANEAT_APPS_TEST_OUTPUT_DIR}/cpp`
  - Python e2e output root: `${SIMANEAT_APPS_TEST_OUTPUT_DIR}/python`
- `SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE` (default: `${APPS_ROOT}/assets/test_images_classification/goldfish.jpeg`)
- `SIMANEAT_APPS_TEST_KEEP_OUTPUT` (`1` keeps e2e output dirs, default: `0`)
- `SIMANEAT_APPS_TEST_MPK` (optional explicit model `.tar.gz` path)
- `SIMANEAT_APPS_TEST_RTSP_URL` (single RTSP stream URL)
- `SIMANEAT_APPS_TEST_RTSP_URLS` (comma-separated RTSP URLs for multistream tests)
- `SIMANEAT_APPS_TEST_TIMEOUT_MS` (default: `180000`)
- `SIMANEAT_APPS_TEST_REQUIRE_E2E` (`1` means missing e2e prerequisites fail instead of skip)

Example:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

export SIMANEAT_APPS_TEST_MODELS_DIR="${APPS_ROOT}/assets/models"
export SIMANEAT_APPS_TEST_INPUT_DIR="${APPS_ROOT}/assets/test_images"
export SIMANEAT_APPS_TEST_OUTPUT_DIR="${APPS_ROOT}/sandbox/tests"
export SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE="${APPS_ROOT}/assets/test_images_classification/goldfish.jpeg"
export SIMANEAT_APPS_TEST_KEEP_OUTPUT=0
export SIMANEAT_APPS_TEST_TIMEOUT_MS=180000
export SIMANEAT_APPS_TEST_REQUIRE_E2E=1
```

## RTSP E2E Prerequisites

RTSP e2e tests require live reachable RTSP streams at test time:

- `single-rtsp-object-detection-optiview` (C++/Python)
- `multistream-rtsp-detection-pipeline` (C++/Python)
- `live-rtsp-depth-estimation` (C++/Python)

Use one of the following setups.

### Option A (Recommended): One process serving multiple streams

Terminal 1:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"
python3 utils/rtsp/rtsp_multi_file_server.py assets/videos/neat-video-1.mp4 --streams 2 --port 8554 --width 1280 --height 720 --fps 30
```

Terminal 2:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

export SIMANEAT_APPS_TEST_RTSP_URL="rtsp://127.0.0.1:8554/stream0"
export SIMANEAT_APPS_TEST_RTSP_URLS="rtsp://127.0.0.1:8554/stream0,rtsp://127.0.0.1:8554/stream1"

# Optional reachability check
ffprobe rtsp://127.0.0.1:8554/stream0
ffprobe rtsp://127.0.0.1:8554/stream1

./tests/test.sh --all
```

### Option B: Docker RTSP server + ffmpeg publishers

Terminal 1:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"
./utils/rtsp/run_rtsp_server.sh --host-port 8555 --detach
```

Terminal 2:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"
./utils/rtsp/stream_cam.sh --video-path assets/videos/neat-video-1.mp4 --rtsp-host 127.0.0.1 --rtsp-port 8555 --rtsp-path stream0
```

Terminal 3:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"
./utils/rtsp/stream_cam.sh --video-path assets/videos/neat-video-2.mp4 --rtsp-host 127.0.0.1 --rtsp-port 8555 --rtsp-path stream1
```

Terminal 4:

```bash
export APPS_ROOT=/path/to/sima-neat/apps
cd "${APPS_ROOT}"

export SIMANEAT_APPS_TEST_RTSP_URL="rtsp://127.0.0.1:8555/stream0"
export SIMANEAT_APPS_TEST_RTSP_URLS="rtsp://127.0.0.1:8555/stream0,rtsp://127.0.0.1:8555/stream1"
./tests/test.sh --all
```

## Two-Stage CI

- Stage 1 (eLxr runner): `./build.sh` and package artifacts.
- Stage 2 (board/devkit runner): set `SIMANEAT_APPS_TEST_*` and run `./tests/test.sh`.
- Keep build and test as separate stages.
