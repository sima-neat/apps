# ResNet50 PipelineSession Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Intermediate |
| Tags | classification, session, pipeline |
| Status | experimental |
| Binary Name | pipelinesession_resnet50 |

## Concept
ResNet50 classification using Session with ImageInputGroup and Model nodes. Demonstrates the pipeline-based inference pattern.

## Prerequisites
- Compiled ResNet50 MPK (`.tar.gz`)
- Installed NEAT SDK

## Run
### C++
```bash
./build/examples/classification/pipelinesession_resnet50/pipelinesession_resnet50 <model.tar.gz> <image_dir>
```

## Source Files
- C++: `main.cpp`
