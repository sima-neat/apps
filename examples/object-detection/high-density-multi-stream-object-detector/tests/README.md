# High-density detector E2E tests

The E2E tests run C++ and Python separately with H.264 and H.265. For each codec, they select one 1280×720 source at 30 FPS or 29.97 FPS, verified with OpenCV, from `SIMANEAT_TEST_RTSP_H264_URLS` or `SIMANEAT_TEST_RTSP_H265_URLS` and open it 16 times. Configure each source with the matching codec, no B-frames, and a one-second-or-shorter keyframe interval; the tests trust this configuration instead of inspecting codec and B-frame metadata. Use footage containing objects recognized by the configured model; every stream must produce at least one detection. This tests 16 application streams with one unique publisher per codec.

After all streams process 100 warm-up frames each, each test measures 5000 successful metadata sends and requires aggregate throughput above 450 FPS, with each stream exceeding 28.125 FPS over the same interval. The test receiver drains metadata throughout the run, validates object-detection metadata and frame identity on all 16 channels, and reconciles unique received frames against every successful send; it does not measure browser rendering. Video publishing remains enabled. The application reports elapsed time, aggregate FPS, per-stream counts, and send failures in a `[measurement]` JSON line. `HIGH_DENSITY_DETECTOR_MEASURE_FRAMES` enables this bounded measurement for test runs; unset or zero preserves the normal continuous run.

From the repository root on Modalix, use an explicit local environment file containing the codec URLs and an absolute shared model directory. For a focused run, supply a complete local scope with this example enabled and other examples disabled. The current scope validator requires entries for every example:

```bash
export SIMANEAT_APPS_TEST_SCOPE_FILE="$(mktemp /tmp/high-density-scope.XXXXXX)"
python3 - <<'PYTHON'
import os
from pathlib import Path
import yaml
from tests.utils.test_scope import discover_scope

scope = discover_scope(Path.cwd() / "examples", Path.cwd())
selected = "object-detection/high-density-multi-stream-object-detector"
assert selected in scope["examples"]
for name, entry in scope["examples"].items():
    if name != selected:
        entry["unit"] = {"cpp": False, "python": False}
        entry["e2e"] = {
            language: {"enabled": False, "models": []}
            for language in ("cpp", "python")
        }
Path(os.environ["SIMANEAT_APPS_TEST_SCOPE_FILE"]).write_text(
    yaml.safe_dump(scope, sort_keys=False)
)
PYTHON
python3 tests/utils/test_scope.py --scope-file "$SIMANEAT_APPS_TEST_SCOPE_FILE" validate
export SIMANEAT_APPS_TEST_MODELS_DIR=/path/to/models
./tests/test.sh --e2e --cpp --strict --config /path/to/test.env
./tests/test.sh --e2e --python --strict --config /path/to/test.env
```


## High-density decoder qualification

The same C++ and Python E2E tests can use the shipped 720p10 profile. These test
settings do not add application or Core API options. Omitted settings keep the
registered 16-stream, 720p30, 5000-frame test unchanged.

| Environment variable (`SIMANEAT_APPS_TEST_HD_` prefix) | Default | Meaning |
| --- | --- | --- |
| `STREAMS` | 16 | Independent application inputs, from 1 through the application's 80-stream ceiling |
| `SOURCE_FPS` | 30 | Required source FPS, verified before running; 1–120 |
| `MEASURE_FRAMES` | 5000 | Total successful metadata sends after every stream warms up for 100 frames |
| `PROFILE` | `config.yaml` | Shipped `config.yaml`, `config-24x720p20fps.yaml` or `config-48x720p10fps.yaml` |
| `DECODER_BUFFERS` | Selected profile | Optional positive external output count |
| `INPUT_BUFFERS` | Selected profile | Optional positive compressed input count |

For example, from the Apps root on a qualified board, retain the complete local
scope generated above and configure both RTSP sources and the selected model:

```bash
export SIMANEAT_APPS_TEST_HD_PROFILE=config-48x720p10fps.yaml
export SIMANEAT_APPS_TEST_HD_STREAMS=48
export SIMANEAT_APPS_TEST_HD_SOURCE_FPS=10
export SIMANEAT_APPS_TEST_HD_MEASURE_FRAMES=14400
./tests/test.sh --e2e --strict
```

This measures approximately 30 seconds at the offered rate after warm-up, not a
fixed wall-clock duration. For five minutes at N streams, use N × 10 × 300
measured frames and a test timeout that also covers warm-up and startup. Repeat
the complete invocation for startup-repeatability checks. Run C++ and Python
sequentially with `--cpp` and `--python` when collecting separate resource data.
For 10-FPS sources every stream must exceed 9.5 FPS; the historical 30-FPS gate
remains 28.125 FPS per stream. Every stream must produce object detections,
strictly increasing timestamps and unique frame identities, with no metadata
send failures. Received unique metadata counts must match successful sends.

For high-density qualification with repeated URLs, configure the test publisher
with an independent media pipeline per client. With GStreamer RTSP Server, use
`RTSPMediaFactory.set_shared(False)`. In the 48-stream full-application control,
a shared factory produced SDP/SETUP timeouts while the independent-pipeline
factory passed with the same application artifacts, buffer counts and timeouts.
Software-only fanout also passed with the shared factory, so this is a source and
full-graph startup interaction, not evidence of a decoder capacity limit. Keep
publisher sharing settings in the result record; do not increase decoder buffers
or application timeouts to hide the difference.

Each test repeats one verified RTSP source URL across the inputs. This exercises
independent decoder/inference branches, but does not establish fairness between
different cameras. The application uses realtime queue policies; matching sent
and received metadata does not prove every encoded input frame was retained.
Video publishing stays enabled; this harness drains metadata and does not
qualify browser rendering or received video pixels. The app requires positive
explicit decoder counts, so automatic Core decoder sizing must be qualified
through Core's tests rather than an undocumented Apps sentinel.

For supervised hardware qualification only, `SIMANEAT_APPS_TEST_TIMEOUT_MS=0`
disables the application's test subprocess timeout in both languages. The
operator must monitor the run, request graceful cancellation if necessary, and
verify hardware/resource cleanup before another workload. An elapsed timeout is
not evidence that DMA buffers are safe to reuse. Ordinary E2E runs retain their
existing bounded timeout.
