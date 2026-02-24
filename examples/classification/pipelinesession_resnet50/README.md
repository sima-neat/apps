# ResNet50 PipelineSession Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Intermediate |
| Tags | classification, session, pipeline |
| Status | experimental |
| Binary Name | pipelinesession_resnet50 |
| Model | resnet_50 |

## Concept
ResNet50 classification using Session with ImageInputGroup and Model nodes. Demonstrates the pipeline-based inference pattern.

## Prerequisites
- Installed NEAT SDK
- Model downloaded: `./scripts/download_models.sh` (or `sima-cli modelzoo get resnet_50`)

## Run
### C++
```bash
./build/examples/classification/pipelinesession_resnet50/pipelinesession_resnet50 models/resnet_50_mpk.tar.gz <image_dir>
```

## Source Files
- C++: `main.cpp`
