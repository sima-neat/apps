# <Example Name>

## Metadata
| Field | Value |
| --- | --- |
| Category | <classification / object-detection / semantic-segmentation / instance-segmentation / depth-estimation / throughput> |
| Difficulty | <Beginner / Intermediate / Advanced> |
| Tags | <comma-separated tags> |
| Status | <experimental / stable> |
| Binary Name | <cmake_target_name> |
| Model | <default_model_name> |

## Concept
<1-2 paragraphs: What does this example demonstrate? What NEAT features does it exercise?>

## Supported Models
<!-- Optional: include this section only if the example works with multiple model variants -->
Also works with: `<model_variant_1>`, `<model_variant_2>`

Download any variant into `assets/models/`: `sima-cli modelzoo get <model_variant_1>`

## Prerequisites
- Installed NEAT SDK
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- Download command: `mkdir -p assets/models && cd assets/models && sima-cli modelzoo get <default_model_name> && cd ../..`

## Run
### C++
```bash
./build/examples/<category>/<name>/<binary> assets/models/<default_model_name>_mpk.tar.gz [args]
```
### Python
<!-- Optional: include only if a Python implementation exists -->
```bash
python3 examples/<category>/<name>/<script>.py [args]
```

## Source Files
- C++: `main.cpp`
- Python: `main.py`
