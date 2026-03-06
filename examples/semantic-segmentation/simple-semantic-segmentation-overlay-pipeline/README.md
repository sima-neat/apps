# Simple Semantic Segmentation Overlay Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | semantic-segmentation |
| Difficulty | Intermediate |
| Tags | semantic-segmentation |
| Status | experimental |
| Binary Name | simple-semantic-segmentation-overlay-pipeline |
| Model | fcn_hrnet48 |

## Concept
Semantic segmentation overlay for image folders using FCN-HRNet output tensors.

## Supported Models
Also works with: `fcn_hrnet18`

Download any variant into `assets/models/`:
- `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get fcn_hrnet48 && cd ../..`

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get fcn_hrnet48 && cd ../..`

## Important Behavior
- Model path is positional and required.
- Input directory is scanned for image files.
- Output files are segmentation overlays.
- Every pixel receives a class label via per-pixel argmax, but the overlay intentionally leaves class `0`/background untinted so the original image remains visible in background regions.

## Command-Line Options
### C++
- Invocation:
  `./build/examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/simple-semantic-segmentation-overlay-pipeline <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

### Python
- Invocation:
  `python examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/python/main.py <model.tar.gz> <input_dir> <output_dir>`
- Required arguments:
  `<model.tar.gz> <input_dir> <output_dir>`
- Optional arguments:
  None.

## Build
### Build From The Apps Repo
```bash
cd <apps-repo-root>
./build.sh
```

Binary output:
```bash
./build/examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/simple-semantic-segmentation-overlay-pipeline
```

### Build This Example Directly With CMake
```bash
cd <apps-repo-root>/examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline
cmake -S cpp -B build
cmake --build build -j
```

Binary output:
```bash
./build/simple-semantic-segmentation-overlay-pipeline
```

## Run
### C++
```bash
./build/examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/simple-semantic-segmentation-overlay-pipeline \
  assets/models/fcn_hrnet48_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/python/requirements.txt
python examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/python/main.py \
  assets/models/fcn_hrnet48_mpk.tar.gz <input_dir> <output_dir>
```

## Debugging Notes
- If output is blank, verify label-map parsing and output tensor shape in logs.
- Validate image decode for all files in input folder.
- Ensure output directory is writable.

## Source Files
- C++ source: `cpp/main.cpp`
- Python source: `python/main.py`
