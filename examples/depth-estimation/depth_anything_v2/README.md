# Depth Anything V2

## Metadata
| Field | Value |
| --- | --- |
| Category | depth-estimation |
| Difficulty | Intermediate |
| Tags | depth-estimation |
| Status | experimental |
| Binary Name | depth_anything_v2 |

## Concept
Minimal Depth Anything V2 pipeline: loads a depth model and infers depth maps for every image in a folder.

## Prerequisites
- Compiled Depth Anything V2 MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/depth-estimation/depth_anything_v2/depth_anything_v2 <model.tar.gz> <input_dir> <output_dir>
```

## Source Files
- C++: `main.cpp`
