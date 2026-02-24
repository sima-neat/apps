# ResNet50 ModelMPK Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Status | experimental |
| Binary Name | modelmpk_resnet50 |
| Model | resnet_50 |

## Concept
ResNet50 classification using the ModelMPK API with OpenCV-based preprocessing. Demonstrates tensor extraction from inference samples.

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `sima-cli modelzoo get resnet_50`

## Run
### C++
```bash
./build/examples/classification/modelmpk_resnet50/modelmpk_resnet50 models/resnet_50_mpk.tar.gz <image>
```

## Source Files
- C++: `main.cpp`
