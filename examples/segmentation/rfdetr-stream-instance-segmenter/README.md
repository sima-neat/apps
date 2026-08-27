# RF-DETR Stream Instance Segmenter

## Metadata

| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Advanced |
| Tags | segmentation, rfdetr, instance-segmentation, rtsp, insight |
| Languages | C++, Python |
| Status | stable |
| Binary Name | rfdetr-stream-instance-segmenter |
| Model | rfdetr-seg-432-base |

## Concept

Segments objects in one RTSP H.264 stream with RF-DETR-Seg and sends synchronized H.264 video and mask metadata to Insight.

The model ships as a three-stage split so it fits the Modalix pipeline: an INT8 backbone (MLA), a small top-k gather step (A65, in-process TVM), and a BF16 transformer and segmentation head (MLA). The application owns the handoff between stages itself, one decoded frame at a time, instead of expressing the model as a single graph node.

Metadata is sent as `type: "segmentation"` with one `data.segments[]` entry per instance, the same wire format the other segmentation examples in this category use. Each of the 200 query masks is a 108x108 grid over the model's stretched (non-letterboxed) 432x432 input, so a detection's silhouette is mapped back through that plain per-axis scale, upscaled to the detection rectangle, and only then thresholded. The silhouette is sent as `mask_format: "polygon"` in frame pixels. Each frame stays inside a 32 KB payload budget; over it, the lowest-confidence segments are dropped and counted in the run summary.

## Preview

![RF-DETR stream instance segmenter preview](../../../portal/assets/examples/segmentation/rfdetr-stream-instance-segmenter/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- An RTSP H.264 source and an [Insight](https://developer.sima.ai/software/tools/insight/) URL reachable from the target.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/segmentation/rfdetr-stream-instance-segmenter
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

The default model is `rfdetr-seg-432-base`, a three-stage split package (backbone, top-k gather, transformer+seg-head). It is not yet in the Model Zoo, so it ships as a direct artifact: a single archive containing all three extracted stage bundles.

| Model | Role | Source |
| --- | --- | --- |
| `rfdetr-seg-432-base` | Default | Direct artifact |

```bash
mkdir -p models
cd models
sima-cli download <model-archive-url>
tar -xzf <model-archive-name>.tar.gz
cd ..
```

> TODO (before merging): host `rfdetr_seg_432_base.tar.gz` at a stable direct-download URL (for example a GitHub Releases asset or an internal artifact store) and fill in `<model-archive-url>` above. The archive must extract to one folder containing the three stage subfolders: `rfdetr_seg_432_simplified_backbone_before_topk_base_mpk/`, `rfdetr_seg_432_simplified_topk_to_gather_base_mpk/`, and `rfdetr_seg_432_simplified_transformer_after_gather_base_mpk/`.

Set `model.path` in the config to the extracted folder (the one containing the three subfolders above), not to the archive itself.

## Prepare Insight

[Insight](https://developer.sima.ai/software/tools/insight/) can host the input stream and render segmentation metadata. Install videos directly from the Insight catalog or through Insight's YouTube support.

In the Insight Web UI, start the required RTSP H.264 stream and copy its source URL. Decoded frames are re-encoded as H.264 for Insight output.

## Configure

Open `${APP_DIR}/src/common/config.yaml`. Set `model.path`, `source.url`, and the Insight host and video and metadata ports.

Set `output.save_dir` only if you also want sampled annotated images.

## Run

### C++

```bash
./${APP_DIR}/src/cpp/pre-built/rfdetr-stream-instance-segmenter \
  --config ${APP_DIR}/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r ${APP_DIR}/src/python/requirements.txt
python3 ${APP_DIR}/src/python/main.py \
  --config ${APP_DIR}/src/common/config.yaml
```

## Troubleshooting

- Verify `model.path` and `source.url` if startup fails.
- Verify stream reachability if the first frame times out.
- Verify the Insight host and UDP ports if no output arrives.
- Set `output.save_dir` and `output.save_every` to save sampled frames.
- This example runs the three model stages and postprocessing on a single thread per frame, trading some throughput for a readable, linear pipeline. Raise `runtime.profile` to see per-stage timings if a stream falls behind.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- Python source: `src/python/main.py`
- Shared config and labels: `src/common/`

The packaged C++ source is an implementation reference. Run the executable under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
