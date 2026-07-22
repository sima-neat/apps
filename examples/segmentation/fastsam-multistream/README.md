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
| Model | FastSAM-x (+ MobileCLIP2-S0) |

## Concept
`fastsam-multistream` matches a MobileCLIP **text prompt** against FastSAM masks across several RTSP cameras on one Modalix MLA:

- decode each RTSP stream into frames
- run FastSAM segmentation on every stream
- score each mask against the prompt with MobileCLIP and keep the best match
- send H.264 video plus `segmentation` metadata to Insight, per stream

One natural-language query (e.g. `"the black labrador"`) selects a mask in every stream. The C++ app is tuned for throughput; the Python app runs FastSAM and MobileCLIP on every frame.

## Preview
The prompt `"the black labrador"` matched across four Insight channels:

![FastSAM multistream preview](../../../assets/portal/segmentation/fastsam-multistream/image.png)

## Insight Setup
[Neat Insight](https://developer.sima.ai/software/tools/insight/) can host RTSP sources, receive video from `VideoSender`, receive segmentation metadata from `MetadataSender`, and show rendered overlays plus runtime metrics in the browser.

Install the sample videos, which Insight can stream as RTSP sources:

```bash
sima-cli install assets/multi-video-sources
```

1. Run `neat`, open the reported `Insight Web UI`, and open `RTSP Source`.
2. Assign a sample or your own video to each source, start the streams, and copy the RTSP URLs into `source.rtsp_urls`.
3. Set `output.insight.host`, `video_port_base`, and `metadata_port_base` from the reported `videoUDP` and `metadataUDP` ranges.

## Supported Models
This example uses two custom compiled models rather than a modelzoo package:

- **FastSAM-x** — the segmentation model (on-device YOLO26-seg decode).
- **MobileCLIP2-S0** — the image and text encoders that score each mask against the prompt.

These are not distributed through `sima-cli modelzoo`; they come from the FastSAM and MobileCLIP model-preparation pipelines. Store the compiled `_mpk.tar.gz` packages under `assets/models/` and point `model.path` and `clip.*` at readable package paths:

```yaml
model:
  path: assets/models/fastsam/FastSAM-x_quant_mpk.tar.gz
clip:
  image_encoder_path: assets/models/mobileclip/MobileCLIP2-S0_image_encoder_reparam_mpk.tar.gz
  text_encoder_path:  assets/models/mobileclip/MobileCLIP2-S0_text_mpk.tar.gz
  text_host_consts:   assets/models/mobileclip/MobileCLIP2-S0_text_host_consts.npz
```

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- One or more reachable RTSP sources created in Insight or provided by your cameras. The C++ app caps at four streams; the Python app is uncapped, and MLA throughput (~8 fps aggregate at four streams) is the practical limit.
- FastSAM-x and MobileCLIP2-S0 compiled for Modalix, referenced from `src/common/config.yaml`.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) with the [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/) installed for setup and compilation.

Clone and build the apps repo inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Configure
Edit `examples/segmentation/fastsam-multistream/src/common/config.yaml`.

```yaml
model:
  path: <fastsam-model-path>                       # FastSAM-x package.

clip:
  image_encoder_path: <clip-image-encoder-path>    # MobileCLIP2-S0 image tower.
  text_encoder_path:  <clip-text-encoder-path>     # MobileCLIP2-S0 text tower.
  text_host_consts:   <clip-text-host-consts>      # MobileCLIP2-S0 text host constants.

source:
  rtsp_urls:                                       # RTSP stream URLs, one per camera.
    - <rtsp-url-1>
    - <rtsp-url-2>

prompt:
  text: "the black labrador"                       # The query matched across every stream.

output:
  insight:
    host: <insight-host-ip>                        # Host running Insight.
    video_port_base: 9000                          # Stream N video    = base + N.
    metadata_port_base: 9100                       # Stream N metadata = base + N.
```

See the comments in `config.yaml` for CLIP tuning, decode thresholds, and C++-only runtime options.

## Run
### C++
The C++ binary takes the config path as a positional argument:

```bash
./build/examples/segmentation/fastsam-multistream/fastsam-multistream \
  examples/segmentation/fastsam-multistream/src/common/config.yaml
```

### Python
The Python script takes the config path as a positional argument:

```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/fastsam-multistream/src/python/requirements.txt
python3 examples/segmentation/fastsam-multistream/src/python/main.py \
  examples/segmentation/fastsam-multistream/src/common/config.yaml
```

Stop with `Ctrl+C`, or set `runtime.frames` for a fixed-length run.

### Text prompt (C++)
The C++ app reads precomputed text features instead of running the text encoder. Regenerate them whenever you change `prompt.text` (run on the board, where pyneat and the MLA are available):

```bash
python3 examples/segmentation/fastsam-multistream/src/tools/precompute_text_features.py \
  examples/segmentation/fastsam-multistream/src/common/config.yaml
# writes clip.text_features_path (default src/common/text_features.npy)
```

## Source Files
- C++ source: `src/cpp/main.cpp`
- C++ helpers: `src/cpp/pipeline.cpp`, `src/cpp/fastsam.cpp`, `src/cpp/image_encoder.cpp`, `src/cpp/config.cpp`
- Python source: `src/python/main.py`
- Python helpers: `src/python/fastsam.py`, `src/python/clip.py`, `src/python/config.py`, `src/python/tokenizer.py`
- Text-feature tool (C++ app): `src/tools/precompute_text_features.py`
- Shared config: `src/common/config.yaml`
- Text features (C++ app): `src/common/text_features.npy`
