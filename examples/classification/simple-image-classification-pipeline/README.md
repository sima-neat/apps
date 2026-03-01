# ResNet50 Model Example

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | classification, model, mpk |
| Status | experimental |
| Binary Name | simple-image-classification-pipeline |
| Model | resnet_50 |

## Concept
Minimal Model API usage with a ResNet50 MPK. Loads a compiled model package, runs synchronous inference, and prints top-1 predictions.

## Prerequisites
- Installed NEAT SDK
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get resnet_50 && cd ../..`
- Pass model explicitly at runtime with `--model <path/to/resnet_50_mpk.tar.gz>`

## Run
### C++
```bash
./build/examples/classification/simple-image-classification-pipeline/simple-image-classification-pipeline --model assets/models/resnet_50_mpk.tar.gz
```

### Python
```bash
source <venv-path>/bin/activate
pip install -r examples/classification/simple-image-classification-pipeline/requirements.txt
python examples/classification/simple-image-classification-pipeline/main.py --model assets/models/resnet_50_mpk.tar.gz
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
