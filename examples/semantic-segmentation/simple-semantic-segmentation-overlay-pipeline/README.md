# FCN-HRNet Semantic Segmentation Overlay

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
Minimal semantic segmentation overlay using an FCN-HRNet model. Processes images from a folder and writes segmentation overlays to an output directory.

## Supported Models
Also works with: `fcn_hrnet18`

Download any variant: `sima-cli modelzoo get fcn_hrnet18`

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `sima-cli modelzoo get fcn_hrnet48`

## Run
### C++
```bash
./build/examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/simple-semantic-segmentation-overlay-pipeline models/fcn_hrnet48_mpk.tar.gz <input_dir> <output_dir>
```

### Python
```bash
source ~/pyneat/.venv/bin/activate
pip install -r examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/requirements.txt
python examples/semantic-segmentation/simple-semantic-segmentation-overlay-pipeline/main.py models/fcn_hrnet48_mpk.tar.gz <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
