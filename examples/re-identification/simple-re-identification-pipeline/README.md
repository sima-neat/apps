# Simple Re-Identification Pipeline

## Metadata
| Field | Value |
| --- | --- |
| Category | re-identification |
| Difficulty | Beginner |
| Tags | re-identification, embedding, similarity, model, mpk |
| Status | experimental |
| Binary Name | simple-re-identification-pipeline |
| Model | reid |

## Concept
Minimal pyneat Model API usage for person re-identification. The example loads
a folder of person-crop images, extracts feature embeddings via an on-device
re-identification model, and computes pairwise cosine similarity scores. Results
are written to `similarity.txt` in the output directory.

## Input / Output
- **Input**: directory of person-crop images (JPEG/PNG/BMP)
- **Model input**: `128 × 256` RGB images (`input_max_depth = 3`)
- **Output**: `output_dir/similarity.txt` with pairwise similarity scores

## Supported Models
Primary model: `reid`

Download into `assets/models/`:
```
mkdir -p assets/models && cd assets/models && sima-cli modelzoo get reid && cd ../..
```

## Prerequisites
- Installed NEAT SDK.
- Model artifacts are user-managed and should be downloaded into `assets/models/`.
- At least two images in the input directory.

## Important Behavior
- All three positional arguments (`model`, `input_dir`, `output_dir`) are required.
- `input_max_depth` **must** be set to `3` when using `format = "RGB"` or `"BGR"`;
  omitting it causes `Session::build` to fail with
  `video depth does not match format`.
- At least 2 readable images are required; otherwise the pipeline exits with code 3.

## Command-Line Options
### Python
- Invocation:
  `python examples/re-identification/simple-re-identification-pipeline/python/main.py <model> <input_dir> <output_dir>`
- Required arguments:
  `model`, `input_dir`, `output_dir`

## Run
### Python
```bash
source ~/pyneat/bin/activate
pip install -r examples/re-identification/simple-re-identification-pipeline/python/requirements.txt
python examples/re-identification/simple-re-identification-pipeline/python/main.py \
  assets/models/reid_mpk.tar.gz \
  /path/to/person/crops \
  /tmp/reid_output
```

## Debugging Notes
- `Error: [misconfig.input_shape] … video depth does not match format` — verify
  that `opt.input_max_depth = 3` is set alongside `opt.format = "RGB"`.
- If an image cannot be read, it is skipped with a warning.
- If fewer than 2 images are loadable, the pipeline exits with code 3.

## Source Files
- Python source: `python/main.py`
