# Semantic Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | segmentation |
| Difficulty | Intermediate |
| Tags | segmentation |
| Status | experimental |
| Binary Name | semantic-overlay |
| Model | fcn_hrnet48 |

## Concept
Semantic segmentation overlay for image folders using FCN-HRNet output tensors.

## Preview
Snippet from a pipeline run:

![Semantic segmentation overlay preview](../../../assets/portal/segmentation/semantic-overlay/image.png)

## Supported Models
Also works with: `fcn_hrnet18`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v 2.0.0 get fcn_hrnet48 && cd ../..`

## Prerequisites
- Installed Neat SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo -v 2.0.0 get fcn_hrnet48 && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for image files.
- Output files are segmentation overlays.
- Runtime settings live in `common/config.yaml`.
- Every pixel receives a class label via per-pixel argmax, but the overlay intentionally leaves class `0`/background untinted so the original image remains visible in background regions.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/segmentation/semantic-overlay/semantic-overlay [--config <path>]`
- Required arguments:
  None. Defaults to `common/config.yaml`.
- Optional arguments:
  `--config <path>`

### Python
- Invocation:
  `python examples/segmentation/semantic-overlay/python/main.py [--config <path>]`
- Required arguments:
  None. Defaults to `common/config.yaml`.
- Optional arguments:
  `--config <path>`

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/segmentation/semantic-overlay/semantic-overlay
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/segmentation/semantic-overlay
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/semantic-overlay
```

## Run
### C++
```bash
./build/examples/segmentation/semantic-overlay/semantic-overlay \
  --config examples/segmentation/semantic-overlay/common/config.yaml
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/segmentation/semantic-overlay/python/requirements.txt
python examples/segmentation/semantic-overlay/python/main.py \
  --config examples/segmentation/semantic-overlay/common/config.yaml
```

## Debugging Notes
- If output is blank, verify label-map parsing and output tensor shape in logs.
- Validate image decode for all files in input folder.
- Ensure output directory is writable.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
- Shared config: `common/config.yaml`
