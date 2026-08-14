# Manual End-to-End Testing Guide

Every command below was run and verified on a Modalix DevKit with Insight. It uses
`tools/gen_test_config.sh` to fill in the placeholders the shipped
`src/common/config.yaml` leaves blank (Insight host, RTSP URLs, model paths), so
you can run immediately.

Adjust two IPs for your setup:
- **`DEVKIT`** — the board that runs the app (this guide: `sima@192.168.135.72`).
- **`HOST`** — the machine running Insight + the RTSP server (this guide:
  `192.168.131.68`). The DevKit must be able to reach it.

---

## 0. What you need

| Piece | Where | Notes |
|---|---|---|
| App binary | DevKit `/tmp/adaptive-build/.../adaptive-resolution-object-detector` | build in Step 1 |
| 3 tier models | `assets/models/yolo26n-{320,640,960}-det-int8-mla_tess-b1.tar.gz` | already built; else `tools/build_yolo26_tiers.sh` |
| Insight + RTSP | `HOST:9900` (API), `HOST:8554` (RTSP), UDP `9000+`/`9100+` | video/metadata ingest |
| Shared FS | `/workspace/tudu_jaagrit/apps` identical on host+DevKit | edit here, run on DevKit |
| Your videos | any `.mp4` | Insight re-encodes to H.264 |

---

## 1. Build (only if you changed source; `/tmp` is wiped on reboot)

```bash
ssh sima@192.168.135.72        # pw: edgeai
cd /workspace/tudu_jaagrit/apps
cmake -S . -B /tmp/adaptive-build -DBUILD_TESTING=ON -DSIMANEAT_APPS_BUILD_CPP=ON
cmake --build /tmp/adaptive-build -j"$(nproc)" \
  --target app__workspace_tudu_jaagrit_apps_examples_object_detection_adaptive_resolution_object_detector_src_cpp_adaptive_resolution_object_detector \
           adaptive-resolution-object-detector_unit_test
BIN=/tmp/adaptive-build/examples/object-detection/adaptive-resolution-object-detector/adaptive-resolution-object-detector
```
`$BIN` only lives in the current shell — re-`export` it in each new SSH session, or use the full path.

Python parity: `pip install -r src/python/requirements.txt`, then
`python3 src/python/main.py --config <cfg>` (same flags).

---

## 2. RTSP sources

**Already have your own RTSP streams?** Skip this section — pass their full URLs
straight to the generator in Step 3 (it uses any `SRC` containing `://` verbatim):
```bash
bash $GEN /tmp/mytest.yaml 192.168.131.68 - \
  rtsp://192.168.131.68:8555/busy rtsp://192.168.131.68:8555/simple
```
The app auto-negotiates each stream's fps (high/variable-rate sources are fine).

**Otherwise, host YOUR videos via Insight (run on HOST):**

```bash
B=https://192.168.131.68:9900
curl -k -F "file=@/path/to/busy_scene.mp4"   $B/api/upload/media      # many/small objects
curl -k -F "file=@/path/to/simple_scene.mp4" $B/api/upload/media      # few/large objects
curl -k -H 'Content-Type: application/json' -d '{"index":1,"file":"busy_scene.mp4","transport":"rtsp"}'   $B/api/mediasrc/assign
curl -k -H 'Content-Type: application/json' -d '{"index":2,"file":"simple_scene.mp4","transport":"rtsp"}' $B/api/mediasrc/assign
curl -k -H 'Content-Type: application/json' -d '{"count":2}' $B/api/mediasrc/start-bulk
curl -k $B/api/mediasrc     # confirm state=playing
```
The DevKit connects to `rtsp://192.168.131.68:8554/src1`, `/src2`, … (`src<index>`).
**Insight re-encodes uploads to 30 fps**; for true source-fps tests host a raw RTSP
(`ffmpeg -re -i your60fps.mp4 -c copy -f rtsp rtsp://…`) or a camera.

---

## 3. Generate a runnable config

```bash
cd /workspace/tudu_jaagrit/apps
GEN=examples/object-detection/adaptive-resolution-object-detector/tools/gen_test_config.sh
# gen_test_config.sh OUT INSIGHT_HOST RTSP_HOST SRC...
bash $GEN /tmp/mytest.yaml 192.168.131.68 192.168.131.68 src1 src2
```
Tune behaviour with env vars (defaults in parens): `RES(320,640,960)`, `HYST(15)`,
`BUDGET(12)`, `MINPX(24)`, `CONF_LOW(0.40)`, `DENSITY(20)`, `MIN_SCORE(0.30)`,
`FPS(0)`, `FRAMES(0)`, `MAX_STREAMS(8)`, `VIDEO(true)`, `PROFILE(true)`,
`DEBUG_DIR(/tmp/adaptive_out)`, `SAVE_EVERY(20)`.

---

## 4. Run

```bash
BIN=/tmp/adaptive-build/examples/object-detection/adaptive-resolution-object-detector/adaptive-resolution-object-detector
$BIN --config /tmp/mytest.yaml --validate-config-only        # quick sanity (no streams)
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 $BIN --config /tmp/mytest.yaml   # Ctrl-C to stop
```

---

## 5. See results (4 ways)

1. **Insight viewer** (visual overlays) — open the URL this prints, in a browser:
   ```bash
   curl -k "https://192.168.131.68:9900/api/viewer-url?src=0,1"
   ```
2. **Is data arriving?**
   ```bash
   curl -k "https://192.168.131.68:9900/api/ingest/stats?all=1" | python3 -m json.tool
   ```
   Healthy stream: `media.idr_count` rising, `metadata.messages_received` rising,
   `metadata.invalid_json: 0`, `remote_addr` = the DevKit.
3. **Debug frames** — set `DEBUG_DIR`/`SAVE_EVERY`; annotated JPEGs with a
   `cam-id · tier · streams` banner land in `DEBUG_DIR` on the DevKit; `scp` them back.
4. **Logs** — `[stream …] tier X -> Y`, `[config] reload: N stream(s)`,
   `[profile stream=… output_fps=… avg_pull_ms=… avg_boxes=…]`, and the pipeline
   dump's `model-path=…yolo26n-<size>…` (proves which tier archive is loaded).

---

## 6. Functionality test matrix (all verified)

| # | Function | Command / action | Expected |
|---|---|---|---|
| 1 | Detect + Insight | `bash $GEN /tmp/c.yaml 192.168.131.68 192.168.131.68 src1 src2` → run | boxes in viewer; `ingest/stats` shows video+metadata, `invalid_json:0` |
| 2 | **Easy scene → tier down** | run a **few-large-object** video, `BUDGET=100` | logs `tier 640 -> 320`; loads `yolo26n-320` archive |
| 3 | **Hard scene → tier up** | run a **busy/small-object** video, or force it: `MINPX=400 BUDGET=100 bash $GEN … src1` | `tier 640 -> 960`; loads `yolo26n-960` archive |
| 4 | Different real model per tier | any run with a switch | `grep model-path <log>` shows different `yolo26n-320/640/960` ELFs |
| 5 | Small-object → up | video with distant objects; keep `MINPX=24` | tier steps up when smallest object < 24 px |
| 6 | Low-confidence → up | blurry/hard video; `CONF_LOW=0.40` | tier steps up when a box < 0.40 |
| 7 | Density → up | your **too-many-objects** video; `DENSITY=20` | crowded frames push tier up |
| 8 | **Shared budget cap** | `MINPX=400 BUDGET=8 bash $GEN … src1 src2 src3 src4` → run | content wants 960 but stays low; **no `-> 960`** in log |
| 9 | Hysteresis (anti-thrash) | alternating video; compare `HYST=5` vs `HYST=30` | low = frequent `tier ->` lines; high = few |
| 10 | **Runtime add** | run with 2 srcs; after ~25s `cp` a config with a 3rd src over it | `[config] reload: 3`; `[stream cam-3] channel=2` |
| 11 | **Runtime remove** | then `cp` a 2-src config back | `[config] reload: 2`; `[stream cam-3] removed (channel 2 released)` |
| 12 | max_streams | `MAX_STREAMS=2 bash $GEN /tmp/m.yaml … src1 src2 src3` → `--validate-config-only` | exit 1, `streams count exceeds streams.max_streams` |
| 13 | 8 concurrent | 8 srcs, `MAX_STREAMS=8` | all 8 init (channels 0–7), no crash |
| 14 | Metadata-only | `VIDEO=false bash $GEN …` | log `video=disabled`; only metadata in `ingest/stats` |
| 15 | fps cap | `FPS=10 bash $GEN …` | `[profile] output_fps ≈ 10` |
| 16 | frame limit | `FRAMES=200 bash $GEN …` | each stream stops at 200, clean exit |
| 17 | Invalid reload ignored | while running, `echo 'streams: []' >> the config` | log `ignoring invalid config reload`; streams keep running |
| 18 | Bare-list form | hand-write `streams:` as a plain URL list | auto ids cam-1, cam-2 |
| 19 | Unit tests | `cd /tmp/adaptive-build && APPS_ROOT=/workspace/tudu_jaagrit/apps ctest -R adaptive-resolution -L unit --output-on-failure` | policy/budget/hysteresis/CLI pass, no hardware |

### Runtime add/remove — the reliable recipe
```bash
GEN=examples/object-detection/adaptive-resolution-object-detector/tools/gen_test_config.sh
bash $GEN /tmp/a.yaml 192.168.131.68 192.168.131.68 src1 src2          # 2 streams
bash $GEN /tmp/b.yaml 192.168.131.68 192.168.131.68 src1 src2 src3     # +cam-3
cp /tmp/a.yaml /tmp/live.yaml
$BIN --config /tmp/live.yaml &                 # start
sleep 25                                       # let both streams come up first!
cp /tmp/b.yaml /tmp/live.yaml ; sleep 18       # ADD cam-3
cp /tmp/a.yaml /tmp/live.yaml ; sleep 12       # REMOVE cam-3
kill -INT %1
```
Give init time before the first edit, and space edits by more than
`config_watch_seconds` (1 s) — otherwise a change can be missed.

---

## 7. Mapping YOUR videos → tests

- **Too-many-objects video** → #3, #5, #7 (drives tier *up* to 960; then #8: add streams and watch budget force it back down).
- **Few/large-object video** → #2 (drives tier *down* to 320). *Note: a highway with a handful of big cars is an "easy" scene and correctly goes to 320 — you need genuinely small/dense objects to reach 960.*
- **Alternating dense↔sparse** → #9 (hysteresis — the key stability knob; tune `HYST` and count `tier ->` lines).
- **Different-fps videos** → #15 via `FPS`; for true source fps, host raw RTSP (Insight normalises uploads to 30 fps).

**Calibrate one expectation:** with a single 30 fps source, wall-clock FPS is
source-capped (~30) at every tier, so 320 and 960 *look* the same. The compute
difference is real (0.32M / 0.67M / 1.24M MLA cycles) and shows up as **headroom** —
lower tiers let more streams keep up. Test throughput impact with many streams
(#13) + budget (#8), not one stream.

---

## 8. Cleanup & troubleshooting

```bash
curl -k -H 'Content-Type: application/json' -d '{}' https://192.168.131.68:9900/api/mediasrc/stop-all
ssh sima@192.168.135.72 'pkill -INT -f adaptive-resolution-object-detector'
```
- **`failed to probe RTSP`** → source not `playing` (`/api/mediasrc`) or wrong host/URL.
- **One dense stream starves the others** (app seems to hang) → a stream at tier 960 monopolises the MLA. Keep `BUDGET` at the default (12) so streams share; raise it only to *demonstrate* a single stream reaching 960.
- **High/variable-fps sources** (e.g. a stream reporting 150/500 fps) are handled — the app no longer pins the decoder fps. Use `FPS=15`/`FPS=30` to cap processing to something sane if a source pushes frames very fast.
- **A tier switch reconnects RTSP (~1 s stall)** — normal; switches are rate-limited to ≥2.5 s and serialized across streams so many streams never rebuild at once.
- **Aggressive settings** (`HYST` very low + forced churn across many streams) push the MLA runtime hard; keep `HYST ≥ 15` (default). The app rate-limits, serialises, settles between rebuilds, and self-heals transient MLA errors, but sane hysteresis avoids the churn entirely.
- **No overlay in viewer** but `ingest/stats` shows data → open the URL from `/api/viewer-url` (the browser DataChannel must connect).
- **A very dense stream shows detection numbers in the console but no boxes in Insight** (while a sparse stream renders fine) → its per-frame metadata packet exceeds one UDP datagram (~1500-byte MTU) and Insight drops it (it does not reassemble fragments — verified: the viewer clears + redraws per message). Check `/api/ingest/stats` — the dense channel shows `active: false` with a low `messages_received`. The metadata is **compact** (label/confidence/bbox only), so up to **~20 detections** deliver reliably. For denser scenes cap detections: `MAX_DET=20` on the generator (or `inference.max_detections: 20`). To draw *all* boxes in a huge crowd you'd burn them into the video stream instead of the metadata overlay.
- **DevKit flaky after heavy runs** → `bash /usr/bin/fix_devkit_runtime.sh` or reboot.
