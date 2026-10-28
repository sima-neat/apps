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
SIMANEAT_TEST_RTSP_H265_URL=rtsp://<host>:<port>/<stream>
SIMANEAT_TEST_RTSP_H265_URLS=rtsp://<host>:<port>/<stream0>,rtsp://<host>:<port>/<stream1>
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

A suite runs with the first model listed under `e2e.<language>.models`. Examples
that advertise several model variants list the ones worth exercising under
`e2e.<language>.variants`; with `SIMANEAT_APPS_TEST_MODEL_VARIANTS=1`, which
`nightly-e2e.yml` sets, those are downloaded too and each e2e test runs once per
model, named after it. Pull request runs leave the variable unset and stay at one
model per suite.

A `source: model-registry` model names the registry model (`name`), the Models
repository branch (`ref`) and the catalog spec (`spec`), and is installed with
`sima-cli neat install models/<name>@<ref>:<spec>`. Adding `variant` instead
downloads only that build variant's latest artifact with `sima-cli models download
--id <name> --variant <variant> --branch <ref>`; such a model must use
`spec: latest`. Either way, `file` is the flat filename the E2E tests read from
the models directory.

The README `Model` row remains customer-facing metadata. Test selection and
test model downloads are controlled by `examples/*/*/tests/test-scope.yaml`, so large
or blocked examples can stay documented without blocking CI. If a test is
enabled in the scope file but the matching Python or C++ test file is missing,
`tests/test.sh` fails before running tests. If a test is disabled, it is skipped
even if a test file exists.

### Adaptive CI fleet

After the build, Vulcan CI queries the existing GitHub App's read-only runner
inventory in the resolved Modalix runner group. It selects up to six idle RTSP
runners and three idle regular runners matching the hardware/environment labels
(and `elxr_version` when pinned). If a pool is entirely busy, it queues one shard
for that pool. Offline registrations never count as available capacity. Discovery
does not provision, reconfigure, reserve, or interrupt devices; the GitHub queue
owns assignment, so availability can change after planning.

Every enabled example/language/unit-or-e2e suite is assigned exactly once, and
the harness self-tests run once. E2E scopes declare `requires_rtsp: true` or
`false` under each language. Missing metadata conservatively requires RTSP.
All unit tests use regular runners. Regular selectors include an existing label
combination that excludes every RTSP registration in the inventory; removing
the `rtsp` label alone would not exclude RTSP runners. Discovery fails clearly
if no such combination exists. Use a dedicated regular label/group in that case.

The optional repository variable `APPS_UNREADY_RUNNERS` is a JSON array of runner
names temporarily excluded from Apps admission (for example, while model access
is being provisioned). The planner also rejects any selector that could match an
excluded registration, even when the proposed candidate has a different name.
Both pools preserve candidate labels in the safety check and emitted selector,
so a dedicated ready-runner label can distinguish healthy devices from excluded
registrations. GitHub cannot express negative runner-name filters. Clear exclusions
only after the affected devices have the required runtime and Developer Portal
model access.
Missing protected-model access fails explicitly instead of starting an interactive
device-login flow in CI. Existing cached sessions and refresh tokens are reused.

The planner balances suites using `tests/ci-durations.json`, seeded from the
recorded successful baseline run. Unknown suites receive a conservative default
duration. Nightly variants and all model selections remain in each generated
scope. Each shard uses the existing harness with `--strict`, keeping suites
serial within a DevKit. Strict E2E stream checks apply only to selected suites
that require streams. `SIMANEAT_APPS_TEST_RUN_HARNESS=0` suppresses the harness
self-tests only for CI shards that do not own them; ordinary local runs retain
the default behavior.

Each shard uploads a result receipt, timing/cache summary, and test logs, even
on failure. `Test Apps Runtime` requires every shard to succeed and verifies
their commit identities and complete suite coverage before packaging. Existing
contract-test, artifact-publication, portal-publication, and nightly-no-publish
gates remain required. Matrix fail-fast cancellation is disabled.

CI caches URL-based model archives under
`/media/nvme/kerrigan/neat-apps-model-cache/v1`, outside the cleaned checkout. NVMe must
be mounted. Cache keys include the acquisition descriptor, Model Zoo version,
and a remote ETag (or Last-Modified plus Content-Length). Every cache hit verifies
the cached SHA-256 before copying the archive into the isolated test runtime.
Misses use the normal scoped downloader; cache publication uses a lock and atomic
file replacement. Registry/modelzoo acquisition and URLs without a usable remote
validator bypass persistent reuse, so mutable registry `latest` and variant
requests resolve on every run. Neither CI cleanup nor package installation
removes the persistent cache.

The cache defaults to a **32 GiB size limit** and **20 GiB free-space reserve**.
Override positive byte counts with `SIMANEAT_APPS_MODEL_CACHE_MAX_BYTES` and
`SIMANEAT_APPS_MODEL_CACHE_RESERVE_BYTES` in the shard environment. The reserve
is checked on both cache and runtime filesystems. Preparation is serialized per
cache; hits refresh LRU age, and cleanup evicts oldest entries under nonblocking
entry locks. Active entries, lock files, unrelated files, and symlinks are never
removed. Abandoned staging directories created by this implementation are
reclaimed only when their entry lock is available. Older unidentifiable staging
directories require operator inspection.

Before copying a cache hit or publishing a new entry, space checks include the
additional copy. Oversized entries or cache publications that cannot preserve
the reserve bypass persistent storage; the runtime copy remains usable. During
normal downloads, storage is checked every 200 ms and the downloader process
group is stopped if eviction cannot restore the reserve. This guards model
acquisition; unrelated writers can still consume space between checks. Shard
receipts/logs report filesystem capacity, free space, cache size/limit, reclaimed
bytes/entries, and downloaded model sizes. No files outside this cache are evicted.

Package installation does not include the test harness or download test models.
Vulcan overlays an ephemeral internal test bundle and lets `tests/test.sh`
download the models selected by the test scope. Only the customer runtime is
published after the activated tests pass.

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
  harness/
    test_output_assertions.py  # harness self-tests; run by test.sh --unit
    test_config_cases.py       # self-tests for the shared config-test helpers
    test_fake_run.py           # self-tests for the scripted run
    test_model_variants.py     # self-tests for per-model e2e generation
  scripts/
    test_*_contract.py         # repository contract tests; need a source checkout,
                               # run with: pytest -c tests/pytest.ini tests/scripts
                               # (CI: the Repository Contract Tests job)
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
    python-harness.log
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

The selected interpreter must have `pytest`, `PyYAML`, NumPy and OpenCV
installed. NumPy and OpenCV are what the e2e output assertions and the harness
self-tests decode saved frames with; this is the same set CI installs:

```bash
${PYTHON_TEST_BIN:-python3} -m pip install pytest PyYAML "opencv-python-headless<4.12"
```

Without NumPy or OpenCV the harness self-tests under `tests/harness` skip with
a message naming the missing package, and `--strict` turns that skip into a
failure.

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

- `SIMANEAT_APPS_TEST_MODELS_DIR` (default: `${APPS_ROOT}/models`)
- `SIMANEAT_APPS_TEST_SCOPE_FILE` (default: `${APPS_ROOT}/examples`)
- `SIMANEAT_APPS_TEST_INPUT_DIR` (default: `${APPS_ROOT}/assets/datasets-test/coco`)
- `SIMANEAT_APPS_TEST_OUTPUT_DIR` (default: `${APPS_ROOT}/sandbox-test`)
- `SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE` (default: `${APPS_ROOT}/assets/datasets-test/imagenet/goldfish.jpeg`)
- `SIMANEAT_APPS_TEST_KEEP_OUTPUT` (`1` keeps e2e output dirs, default: `1`)
- `SIMANEAT_APPS_TEST_WRITE_SUMMARY_LOGS` (`1` writes summary logs, default: `1`)
- `SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS` (`1` writes per-example command/stdout/stderr logs, default: `1`)
- `SIMANEAT_TEST_RTSP_H264_URL` (single RTSP H.264 stream URL)
- `SIMANEAT_TEST_RTSP_H264_URLS` (comma-separated RTSP H.264 URLs)
- `SIMANEAT_TEST_RTSP_H265_URL` (single RTSP H.265 stream URL)
- `SIMANEAT_TEST_RTSP_H265_URLS` (comma-separated RTSP H.265 URLs)
- `SIMANEAT_TEST_RTSP_MJPEG_URL` (single RTSP MJPEG stream URL)
- `SIMANEAT_TEST_RTSP_MJPEG_URLS` (comma-separated RTSP MJPEG URLs)
- `SIMANEAT_TEST_HTTP_MJPEG_URL` (single HTTP MJPEG stream URL)
- `SIMANEAT_TEST_HTTP_MJPEG_URLS` (comma-separated HTTP MJPEG URLs)
- `SIMANEAT_APPS_TEST_TIMEOUT_MS` (default: `180000`)
- `SIMANEAT_APPS_HIGH_DENSITY_INITIAL_PROGRESS_TIMEOUT_MS` (high-density receiver warmup progress timeout, default: `90000`)
- `SIMANEAT_APPS_HIGH_DENSITY_STREAM_PROGRESS_TIMEOUT_MS` (high-density receiver measurement progress timeout, default: `30000`)
- `SIMANEAT_APPS_TEST_REQUIRE_E2E` (backward-compatible strict e2e env flag; prefer `--strict`)
- `SIMANEAT_APPS_TEST_LABELS_FILE` (optional labels file override)
- `SIMANEAT_APPS_TEST_INSIGHT_HOST` (default: `127.0.0.1`)
- `SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT` (optional Insight video port override)
- `SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT` (optional Insight metadata port override)
- `NEAT_APPS_SKIP_MODEL_DOWNLOAD` (`1` skips model download before e2e, default: `0`)
- `PYTHON_TEST_BIN` (optional Python interpreter override)

## Streaming E2E Prerequisites

Streaming e2e tests require live reachable sources at test time:

- `single-stream-object-detector` (C++/Python)
- `single-stream-instance-segmenter` (C++/Python)
- `multi-stream-object-detector` (C++/Python)
- `multi-stream-tracker` (C++/Python)

Use [Insight](https://developer.sima.ai/software/tools/insight/) to install
videos directly from its catalog or through YouTube support. Start the required
streams and set their source URLs in the corresponding test environment
variables. Strict single-stream coverage uses H.264 RTSP, MJPEG RTSP, and HTTP
MJPEG URLs.

## Two-Stage Vulcan CI

- Stage 1 (eLxr runner): `./build.sh --all --clean` validates the Apps and portal
  builds, then creates a runtime candidate and an ephemeral test bundle.
- Stage 2 (Modalix runner): overlays the test bundle, runs
  `./tests/test.sh --all --strict`, and publishes only the runtime candidate after
  every activated test passes.
- Alongside both, an Ubuntu job runs the repository contract tests under
  `tests/scripts` against the source checkout, because they read `build.sh`, the
  workflows and the portal sources, which the test bundle does not carry.
  Publishing waits for it too.

## Nightly Integration

At 03:00 UTC each day, `nightly-e2e.yml` dispatches Vulcan CI on the latest
`develop` with `nightly=true`, then waits and reports its build/test result.
The workflow can also be started manually with a source branch to rebuild;
`develop` is the default. There is no scheduled `main` matrix entry.

It runs the same clean SDK build, package validation, repository contract tests,
and installed-runtime `./tests/test.sh --all --strict` as push CI. The `snap`
Core policy selects the latest published Core `develop` artifact at build time;
the Modalix tests install the exact Core version recorded in the new package.
`SIMANEAT_APPS_TEST_MODEL_VARIANTS=1` adds each enabled suite's model variants.

Nightly mode skips release packaging, Vulcan publication, latest promotion,
and portal/search deployment. Build archives remain ephemeral GitHub Actions
artifacts used to transfer the candidate to the Modalix job. The nightly run
summary links to the full Vulcan build and test logs. Runtime test logs and
resolved Apps/Core metadata are retained for seven days, including on failure.
Nightly and publishing runs use separate concurrency groups so they do not
cancel each other.
