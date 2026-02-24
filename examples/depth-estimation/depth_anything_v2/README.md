# Depth Anything V2

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation |
| Status | experimental |
| Binary Name | depth_anything_v2 |
| Model | depth_anything_v2_vits |

## Concept
Minimal Depth Anything V2 pipeline: loads a depth model and infers depth maps for every image in a folder.

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `sima-cli modelzoo get depth_anything_v2_vits`

## Run
### C++
```bash
./build/examples/depth-estimation/depth_anything_v2/depth_anything_v2 models/depth_anything_v2_vits_mpk.tar.gz <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
