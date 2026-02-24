# ResNet50 Model Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Status | experimental |
| Binary Name | model_resnet50 |
| Model | resnet_50 |

## Concept
Minimal Model API usage with a ResNet50 MPK. Loads a compiled model package, runs synchronous inference, and prints top-1 predictions.

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `sima-cli modelzoo get resnet_50`

## Run
### C++
```bash
./build/examples/classification/model_resnet50/model_resnet50 --model models/resnet_50_mpk.tar.gz
```

## Source Files
- C++: `main.cpp`
