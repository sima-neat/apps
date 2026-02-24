# ResNet50 ModelMPK Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Status | experimental |
| Binary Name | modelmpk_resnet50 |

## Concept
ResNet50 classification using the ModelMPK API with OpenCV-based preprocessing. Demonstrates tensor extraction from inference samples.

## Prerequisites
- Compiled ResNet50 MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/classification/modelmpk_resnet50/modelmpk_resnet50 <model.tar.gz> <image>
```

## Source Files
- C++: `main.cpp`
