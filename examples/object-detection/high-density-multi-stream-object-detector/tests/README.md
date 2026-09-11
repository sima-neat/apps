# High-density detector E2E tests

The E2E tests run C++ and Python separately with H.264 and H.265. For each codec, they select one verified 1280×720 source at 30 FPS or 29.97 FPS without B-frames from `SIMANEAT_TEST_RTSP_H264_URLS` or `SIMANEAT_TEST_RTSP_H265_URLS` and open it 16 times. Keep sources at a one-second-or-shorter keyframe interval. Use footage containing objects recognized by the configured model; every stream must produce at least one detection. This tests 16 application streams with one unique publisher per codec.

After all streams process 100 warm-up frames each, each test measures 5000 successful metadata sends and requires aggregate throughput above 450 FPS, with each stream exceeding 28.125 FPS over the same interval. The test receiver drains metadata throughout the run, validates object-detection metadata and frame identity on all 16 channels, and reconciles unique received frames against every successful send; it does not measure browser rendering. Video publishing remains enabled. The application reports elapsed time, aggregate FPS, per-stream counts, and send failures in a `[measurement]` JSON line. `HIGH_DENSITY_DETECTOR_MEASURE_FRAMES` enables this bounded measurement for test runs; unset or zero preserves the normal continuous run.

Install `ffprobe` on the test host, provided by the `ffmpeg` package on Debian. CI installs this prerequisite before running the tests.

From the repository root on Modalix, use an explicit local environment file containing the codec URLs and an absolute shared model directory. For a focused run, supply a complete local scope with this example enabled and other examples disabled. The current scope validator requires entries for every example:

```bash
export SIMANEAT_APPS_TEST_SCOPE_FILE=/path/to/local-scope.yaml
export SIMANEAT_APPS_TEST_MODELS_DIR=/path/to/models
./tests/test.sh --e2e --cpp --strict --config /path/to/test.env
./tests/test.sh --e2e --python --strict --config /path/to/test.env
```
