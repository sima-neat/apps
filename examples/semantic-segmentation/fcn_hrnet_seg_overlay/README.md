# FCN-HRNet Semantic Segmentation Overlay

## Metadata
| Field | Value |
| --- | --- |
| Category | semantic-segmentation |
| Difficulty | Intermediate |
| Tags | semantic-segmentation |
| Status | experimental |
| Binary Name | fcn_hrnet_seg_overlay |

## Concept
Minimal semantic segmentation overlay using an FCN-HRNet model. Processes images from a folder and writes segmentation overlays to an output directory.

## Prerequisites
- Compiled FCN-HRNet MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/semantic-segmentation/fcn_hrnet_seg_overlay/fcn_hrnet_seg_overlay <model.tar.gz> <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
