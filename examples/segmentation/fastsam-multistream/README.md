# FastSAM Multistream

## Metadata

| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Advanced |
| Tags | segmentation, sam, fastsam, mobileclip, clip, text-prompt, multistream, rtsp, insight |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | fastsam-multistream |
| Model | FastSAM-x + MobileCLIP2-S0 |

## Concept

`fastsam-multistream` runs FastSAM-x over as many as four RTSP streams on one
Modalix MLA. MobileCLIP2-S0 compares every candidate mask with one free-text
prompt and publishes the selected polygon as `segmentation` metadata.

Each stream uses a separate Insight channel for H.264 video and metadata. The
C++ implementation is the throughput path and reuses precomputed prompt
features. The Python implementation encodes the prompt at startup and is the
readable reference.

## Preview

The prompt `"the black labrador"` matched across four Insight channels:

![FastSAM multistream preview](../../../portal/assets/examples/segmentation/fastsam-multistream/image.png)

## Prerequisites

- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/))
  on a supported Modalix or DevKit target.
- Insight with one to four reachable H.264 RTSP sources.

## Install Apps

Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model

This example uses four direct artifacts from the Model Zoo release. That
release can differ from the installed platform version.

| Artifact | Role |
| --- | --- |
| `FastSAM-x_quant_mpk.tar.gz` | FastSAM-x segmentation |
| `MobileCLIP2-S0_image_encoder_reparam_mpk.tar.gz` | MobileCLIP image encoder |
| `MobileCLIP2-S0_text_mpk.tar.gz` | MobileCLIP text encoder |
| `MobileCLIP2-S0_text_host_consts.npz` | Host-side text embeddings and projection |

```bash
export MODELZOO_VERSION="2.1.2"
mkdir -p models/fastsam-mobileclip
cd models/fastsam-mobileclip
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/FastSAM-x_quant_mpk.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/MobileCLIP2-S0_image_encoder_reparam_mpk.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/MobileCLIP2-S0_text_mpk.tar.gz"
sima-cli download "https://docs.sima.ai/pkg_downloads/SDK${MODELZOO_VERSION}/models/modalix/MobileCLIP2-S0_text_host_consts.npz"
cd ../..
```

The host constants are about 98 MiB and remain in the user-managed
`models/` directory; they are not part of the Apps runtime.

## Configure

Edit
`examples/segmentation/fastsam-multistream/src/common/config.yaml`.
The default model paths match the commands above.

Start Insight and inspect its active endpoints:

```bash
neat --json
```

In the Insight Web UI, open **RTSP Source**, assign a video to each source, and
start the streams. Use the reported RTSP URLs and externally reachable
`videoUDP` and `metadataUDP` port ranges:

```yaml
source:
  rtsp_urls:
    - rtsp://<insight-host>:<rtsp-port>/src0
    - rtsp://<insight-host>:<rtsp-port>/src1
    - rtsp://<insight-host>:<rtsp-port>/src2
    - rtsp://<insight-host>:<rtsp-port>/src3

prompt:
  text: "the black labrador"

output:
  insight:
    host: <insight-host>
    video_port_base: <videoUDP-start-port>
    metadata_port_base: <metadataUDP-start-port>
```

The C++-only CLIP interval/tracking and profiling settings are marked in the
shared config. Both implementations use the same models, prompt, decode
thresholds, score threshold, and Insight schema.

## Run

### C++

```bash
./examples/segmentation/fastsam-multistream/src/cpp/pre-built/fastsam-multistream \
  examples/segmentation/fastsam-multistream/src/common/config.yaml
```

### Python

```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/fastsam-multistream/src/python/requirements.txt
python3 examples/segmentation/fastsam-multistream/src/python/main.py \
  examples/segmentation/fastsam-multistream/src/common/config.yaml
```

Stop with `Ctrl+C`, or set `runtime.frames` for a fixed-length run. Open the
Insight Web UI and select channels 0-3 to view the video and segmentation
polygons.

### Change the C++ prompt

The C++ runtime reads precomputed text features. After changing
`prompt.text`, regenerate them on the board:

```bash
python3 examples/segmentation/fastsam-multistream/src/python/precompute_text_features.py \
  examples/segmentation/fastsam-multistream/src/common/config.yaml
```

The command writes `clip.text_features_path` and its matching
`.prompt.txt` file. No C++ rebuild is required.

## Troubleshooting

- Confirm all four files exist under `models/fastsam-mobileclip/`.
- Confirm `neat --json` reports Insight as running and use its current mapped
  RTSP, video, and metadata ports rather than assuming defaults.
- Confirm the board can reach the Insight host and every configured RTSP URL.
- Use `runtime.frames: 60` for a bounded smoke run.

## Source Files

- C++ reference source: `src/cpp/main.cpp`
- C++ pipeline: `src/cpp/pipeline.cpp`
- Python entrypoint: `src/python/main.py`
- Prompt utility: `src/python/precompute_text_features.py`
- Shared runtime files: `src/common/`
- C++ and Python tests: `tests/`

The packaged C++ source is an implementation reference. Run the executable
under `src/cpp/pre-built/`; the installed bundle does not include CMake files.

## Development From Source

To modify, compile, or test this example, use the
[Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
