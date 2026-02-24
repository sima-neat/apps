# ResNet50 Model Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Status | experimental |
| Binary Name | model_resnet50 |

## Concept
Minimal Model API usage with a ResNet50 MPK. Loads a compiled model package, runs synchronous inference, and prints top-1 predictions.

## Prerequisites
- Compiled ResNet50 MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/classification/model_resnet50/model_resnet50 --model /path/to/resnet_50_mpk.tar.gz
```

## Source Files
- C++: `main.cpp`
