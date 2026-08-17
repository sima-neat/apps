# YOLO26 Batch-4 Detector

## Metadata
| Field | Value |
| --- | --- |
| Category | object-detection |
| Difficulty | Advanced |
| Tags | object-detection, yolo26, batching, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | yolo26-batch4-detector |
| Model | yolo26m-det-int8-b4 |

## Concept
Four RTSP streams hosted by Insight are decoded, one frame is taken from each, and all four are submitted to the MLA as a single `[4, 640, 640, 3]` batch — one dispatch instead of four. The model returns the six YOLO26 heads, which are decoded on the CPU (A65) per batch lane, so each stream keeps its own detections. The analysed frame is published to Insight with its detections beside it as metadata.

The example exercises batched inference across several streams, per-lane attribution of results, and a pipeline shaped so that ingest overlaps inference rather than queueing behind it.

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix DevKit.
- A Modalix DevKit with the Neat runtime installed and `simaai-appcomplex` running.
- Model artifacts are user-managed. Download the default model as described below, or set `model.path` to another compatible batch-4 six-head package.
- Four reachable RTSP H.264 sources, from Insight or from cameras.
- A runtime containing the multi-output batch layout fix described below.

### Runtime requirement
Older runtimes misaddress multi-output tensors when `batch_size > 1`, producing incorrect results without reporting an error. Use a Neat runtime that includes the batched multi-output OFM layout fix. If lane 0 looks correct while lanes 1–3 contain implausible detections, update the runtime before debugging the application.

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host RTSP streams, receive video from `VideoSender`, receive detection metadata from `MetadataSender`, and render the overlays in the browser.

In the Neat Development Environment, install the sample video assets:

```bash
sima-cli install assets/multi-video-sources
```

Then create the inputs:
1. Run `neat` in the Neat Development Environment and open the reported `Insight Web UI`.
2. In Insight, open `RTSP Source`.
3. Use the sample videos or upload your own.
4. Start four streams and copy their RTSP URLs.
5. Copy the Insight host and its `videoUDP` and `metadataUDP` base ports into the config. The defaults are 9000 and 9100.

Stream N is published on `video_port_base + N` and `metadata_port_base + N`, as Insight channel N. The application prints the resulting map on startup.

When the application runs on a DevKit and Insight runs in the Neat Development Environment, the RTSP host is the SDK host address, not `127.0.0.1`. No port mapping is needed: the DevKit reaches the RTSP port on that host directly.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

Use the Model Zoo release version wherever `<modelzoo-version>` appears. It can differ from the installed platform version.

Default model: `yolo26m-det-int8-b4.tar.gz`.

Download the default model:

```bash
mkdir -p models
cd models

sima-cli download https://docs.sima.ai/pkg_downloads/SDK<modelzoo-version>/models/modalix/yolo26-detection/yolo26m-det-int8-b4.tar.gz

cd ..
```

The command stores the model under `models/` as a bundle-local convention. `model.path` can point to any readable model package path.

The application requires a batch-4 YOLO26 package with six separate detection heads. It validates this contract at startup and rejects batch-1 or incompatible packages.

## Configure

Edit `examples/object-detection/yolo26-batch4-detector/src/common/config.yaml`. The checked-in file ships placeholders and the application rejects them, so no run can accidentally use someone else's addresses.

```yaml
model:
  path: <model-path>                                     # Example: models/yolo26m-det-int8-b4.tar.gz

streams:
  - <first-rtsp-url-copied-from-insight>
  - <second-rtsp-url-copied-from-insight>
  - <third-rtsp-url-copied-from-insight>
  - <fourth-rtsp-url-copied-from-insight>

inference:
  frames: 0                                              # 0 runs continuously.
  score_threshold: 0.35                                  # int8 score granularity is ~0.06.
  max_detections: 100

output:
  insight:
    host: <insight-host-ip>
    video_port_base: <videoUDP start port from neat>
    metadata_port_base: <metadataUDP start port from neat>
```

Fewer than four streams is allowed; the batch is filled by repeating the last frame, so the MLA still runs one dispatch.

## Run

### Validate Config Only

Quick smoke test that opens no streams and touches no hardware.

```bash
python3 examples/object-detection/yolo26-batch4-detector/src/python/main.py \
  --config examples/object-detection/yolo26-batch4-detector/src/common/config.yaml \
  --validate-config-only
```

### Python

On the Modalix DevKit, from the installed bundle root:

```bash
source ~/pyneat/bin/activate
pip install -r examples/object-detection/yolo26-batch4-detector/src/python/requirements.txt
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 python3 \
  examples/object-detection/yolo26-batch4-detector/src/python/main.py \
  --config examples/object-detection/yolo26-batch4-detector/src/common/config.yaml
```

### C++

```bash
SIMA_GST_RUN_INPUT_TIMEOUT_MS=120000 \
  ./examples/object-detection/yolo26-batch4-detector/src/cpp/pre-built/yolo26-batch4-detector \
  --config examples/object-detection/yolo26-batch4-detector/src/common/config.yaml
```

Open the Insight Web UI to watch the four channels with detection overlays.

## How It Works

1. Each RTSP source is split before decode. Encoded H.264 is forwarded directly to Insight while the second branch is decoded to NV12.
2. Worker threads take the latest frame from each stream, letterbox it to 640×640 RGB, and write it into one lane of a reusable `[4, 640, 640, 3]` tensor.
3. One synchronous `model.run(...)` call submits all four lanes. Batch preparation for the next dispatch continues in parallel.
4. The six raw YOLO26 heads are decoded on the CPU per lane and mapped back to source-frame coordinates.
5. Detection metadata carries the analysed frame's `_insight.rtp_timestamp`; C++ and Python use the same video and metadata topology.

CPU decode is required because the current model-managed `YoloV26` box decode only returns lane 0 for this batched six-head package. YOLO26 is end-to-end, so the application ranks detections but does not run NMS.

The forwarded video can run at 30 FPS while inference produces fewer results per second. Insight therefore has intervening frames without detection metadata, so boxes may appear to flicker. Exact-frame alignment is intentional: repeating old boxes on newer frames makes them visibly trail moving objects.

## Tests

Unit tests cover config loading and validation, the shape-based head mapping, and the decode maths, and need no hardware:

```bash
./tests/test.sh --unit
```

End-to-end coverage downloads the batch-4 model declared in `tests/test-scope.yaml` and runs when the test environment provides four live RTSP sources plus the required Modalix services.

## Debugging Notes

- The checked-in `config.yaml` ships placeholder URLs and an Insight host. The application refuses to start until they are replaced, rather than failing later with a connection error.
- `output.debug_dir` plus `output.save_every` write annotated JPEGs per stream without changing the Insight output contract — useful when Insight is not reachable.
- `runtime.profile` prints throughput and a phase breakdown every `runtime.profile_interval` dispatches.
- `method DESCRIBE failed: 404 Not Found` means the RTSP path exists as a naming scheme but Insight has no media source assigned to it. Assign the sources in the Insight UI first.
- If `pyneat` fails to import with `libsima_neat.so.2: cannot open shared object file`, the DevKit has a `pyneat` wheel that does not match the installed `sima-neat` package. Install the wheel that matches the runtime.
- If lanes 1–3 look incorrect while lane 0 is sensible, update to a runtime containing the multi-output batch fix.

## Source Files

- Test scope: `tests/test-scope.yaml`
- Python source: `src/python/main.py`
- C++ source: `src/cpp/main.cpp`
- Python tests: `tests/python/test_unit.py`
- Python e2e test: `tests/python/test_e2e.py`
- C++ tests: `tests/cpp/test_unit.cpp`
- C++ e2e test: `tests/cpp/test_e2e.cpp`
- Shared assets: `src/common/config.yaml`, `src/common/coco_label.txt`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
