# High-density detector E2E tests

The E2E tests run C++ and Python separately with H.264 and H.265. For each codec, they select one 1280×720 source at 30 FPS or 29.97 FPS, verified with OpenCV, from `SIMANEAT_TEST_RTSP_H264_URLS` or `SIMANEAT_TEST_RTSP_H265_URLS` and open it 16 times. Configure each source with the matching codec, no B-frames, and a one-second-or-shorter keyframe interval; the tests trust this configuration instead of inspecting codec and B-frame metadata. Use footage containing objects recognized by the configured model; every stream must produce at least one detection. This tests 16 application streams with one unique publisher per codec.

The test receiver, rather than the application, owns warmup and measurement. After observing 100 unique metadata frames from every stream, it measures exactly 5000 additional unique received frames over one common interval. Aggregate throughput must exceed 450 FPS, and every stream must exceed 28.125 FPS over that same interval. This received-metadata rate differs from the former application-reported successful-send rate: its interval starts and stops on unique messages accepted by the receiver, so nonblocking sends that are dropped before delivery are not counted. The receiver validates the object-detection schema, channel, stream and frame identity, PTS, and RTP timestamp on every message; every stream must also produce at least one useful detection. Video publishing remains enabled, but the test does not measure browser rendering.

The initial per-stream progress deadline defaults to 90000 ms and the ongoing measured-stream deadline defaults to 30000 ms. Override them with `SIMANEAT_APPS_HIGH_DENSITY_INITIAL_PROGRESS_TIMEOUT_MS` and `SIMANEAT_APPS_HIGH_DENSITY_STREAM_PROGRESS_TIMEOUT_MS`. `SIMANEAT_APPS_TEST_TIMEOUT_MS` remains the total test timeout (default 180000 ms). Missing-stream failures report both warmup and measured counts. After the receiver reaches its target, it sends `SIGINT` to the continuously running application and requires a clean exit within five seconds; a forced kill fails the test.

From the repository root on Modalix, use an explicit local environment file containing the codec URLs and an absolute shared model directory. For a focused run, supply a complete local scope with this example enabled and other examples disabled. The current scope validator requires entries for every example:

```bash
export SIMANEAT_APPS_TEST_SCOPE_FILE=/path/to/local-scope.yaml
export SIMANEAT_APPS_TEST_MODELS_DIR=/path/to/models
./tests/test.sh --e2e --cpp --strict --config /path/to/test.env
./tests/test.sh --e2e --python --strict --config /path/to/test.env
```
