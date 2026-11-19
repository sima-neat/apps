# PreprocessOptions

## What it is

`PreprocessOptions` tells a `Model` what input your application sends and how
Neat should turn it into the tensor the model expects. A model here is a
compiled model package (an MPK, `*_mpk.tar.gz`) loaded with
`simaai::neat::Model`; see [model.md](model.md). Preprocessing covers resize,
color conversion, layout change, normalization (subtract a mean, divide by a
standard deviation), quantization (convert to the model's INT8 input) and
tessellation (rearrange the tensor into the tile layout of the MLA, the
board's machine-learning accelerator).

You set it as `Model::Options::preprocess` (Python `ModelOptions.preprocess`)
before you load the model. The options are an intent: you set only what you
need, and a planner fills in the rest from the package. The image steps run on
the EV74, the board's vision processor.

## When to use it

Set it when your application sends images: camera frames, decoded video or
files read with OpenCV. Then the model takes an image of any size up to a
limit, and Neat resizes, converts and normalizes it on the EV74, so your code
does not.

Leave it at its defaults when your application already produces the exact
tensor the model wants: the shape, type and value range that
`model.input_specs()` reports, for example a float32 `[1, 640, 640, 3]` tensor
for YOLO26n. The model then quantizes it and runs. This is also what you get
when you do not set `preprocess` at all.

A benchmark's source size and resize policy are set in `BenchmarkOptions`
([benchmark.md](benchmark.md)); how boxes are mapped back to the source
image is on [decoding.md](decoding.md).

## Minimal example

C++, from
[`yolo26-object-detector/src/cpp/main.cpp`](../../examples/object-detection/yolo26-object-detector/src/cpp/main.cpp)
(lines 248–252 and 258, de-indented):

```cpp
simaai::neat::Model::Options model_opt;
model_opt.preprocess.kind = simaai::neat::InputKind::Image;
model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
simaai::neat::Model model(cfg.model_path, model_opt);
```

Python, from
[`yolo26-object-detector/src/python/main.py`](../../examples/object-detection/yolo26-object-detector/src/python/main.py)
(lines 171–175 and 180, de-indented):

```python
opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Image
opt.preprocess.enable = pyneat.AutoFlag.On
opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
model = pyneat.Model(model_path, opt)
```

The example then sends each OpenCV frame as a BGR image tensor
(`pyneat.Tensor.from_numpy(..., image_format=pyneat.PixelFormat.BGR)`), and
Neat letterboxes it to 640 × 640 and scales pixels to 0–1.

Complete flow: [YOLO26 Object Detector](../../examples/object-detection/yolo26-object-detector/)

## Options

Fields of `PreprocessOptions` (Python `pyneat.PreprocessOptions`). The tables
under it explain the step structures (`ResizeSpec` and the rest) and the enums.
In Python, an unset optional field is `None`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Input kind | `kind` | `kind` | `InputKind::Auto` | Says whether you send images or ready model tensors. `Auto` lets the planner decide: an image route if you set any image step, otherwise the package's own route, which on YOLO26n and ResNet-50 takes a float32 tensor. | Set `Image` whenever you send pixels. Set `Tensor` only to state explicitly that you send model-shaped tensors. |
| Preprocessing switch (legacy) | `enable` | `enable` | `AutoFlag::Auto` | `On` forces the preprocessing stage on. `Off` turns off the image steps, but keeps the adapters the package requires (such as quantize) and does not make the route MLA-only. A non-empty `transforms` list overrides `Off`. | The image examples set `On` next to `kind = Image`; copy that. Do not use `Off` to skip preprocessing: send a model tensor with default options instead. |
| Maximum input width | `input_max_width` | `input_max_width` | `0`: 1920 px | The widest image the route accepts. It is a capacity, not a resize target. | Your source is wider than 1920 px, such as a 4K camera, or you want to declare the exact camera size, as the depth and anomaly examples do. A wider frame is rejected; see Pitfalls. |
| Maximum input height | `input_max_height` | `input_max_height` | `0`: 1080 px | The tallest image the route accepts. | Same cases as `input_max_width`, for a source taller than 1080 px. |
| Maximum input channels | `input_max_depth` | `input_max_depth` | `0`: taken from the image format (3 for RGB or BGR) | Channel capacity of the input. | Some examples set `3` explicitly next to the width and height. Leave it otherwise. |
| Resize | `resize` | `resize` | `ResizeSpec{}`: `Auto`, mode `Letterbox`, `pad_value` 114 | How the image is fitted to the model's input size. See [`ResizeSpec`](#resize-resizespec). | Set it when the model was trained with a specific fit: letterbox for YOLO, stretch for some depth models, center crop for ImageNet classifiers. |
| Color conversion | `color_convert` | `color_convert` | `ColorConvertSpec{}`: `Auto` | The pixel format you send. See [`ColorConvertSpec`](#color-colorconvertspec). | Always set `input_format` on an image route: `BGR` for OpenCV frames, `RGB` for PIL or converted frames, `NV12` for decoded video or a camera. |
| Layout change | `layout_convert` | `layout_convert` | `LayoutConvertSpec{}`: `Auto`, empty `perm` | Reorders tensor axes, for example height-width-channel to channel-height-width. See [`LayoutConvertSpec`](#layout-layoutconvertspec). | Rarely. No shipped example sets it. |
| Normalization | `normalize` | `normalize` | `NormalizeSpec{}`: `Auto`, mean `{0, 0, 0}`, stddev `{1, 1, 1}`, no explicit stats | Per-channel `(pixel / 255 − mean) / stddev` on the EV74. See [`NormalizeSpec`](#normalize-normalizespec). | Set it, or `preset`, for every image route: without it the model receives unnormalized 0–255 values; see Pitfalls. Set your own mean and stddev for a model trained with other statistics, as the FastFlow example does. |
| Quantization | `quantize` | `quantize` | `QuantizeSpec{}`: `Auto` | Conversion to the model's INT8 input. The package's scale and zero point replace any values you set. | Leave it. Setting `enable` selects a tensor route, and image-only options then fail. |
| Tessellation | `tessellate` | `tessellate` | `TessellateSpec{}`: `Auto`, empty `slice_shape` | Rearranges the tensor into the MLA's tile layout. The package's tile shape replaces yours. | Leave it, for the same reason as `quantize`. |
| Ordered steps | `transforms` | `transforms` | Empty (`[]`) | A list of `Transform` entries, each with a `type` (`TransformType`) and the matching step structure. When the list is not empty, it replaces the step fields above, and the plan warns if both were set. | Rarely. Use the step fields instead. Crop and antialiased resize are rejected in this list. |
| Normalization preset | `preset` | `preset` | `NormalizePreset::None` | Fills normalization (and, for `COCO_YOLO`, the resize) with a known setup. See the [`NormalizePreset` values](#normalizepreset-values). | `ImageNet` for classifiers trained on ImageNet, such as ResNet-50. `COCO_YOLO` for YOLO detectors. |
| Source multiplier | `rescale_factor` | `rescale_factor` | Not set (`None`) | Multiplies source pixels before the resize. Setting it selects fixed source arithmetic: `normalize` then subtracts and divides after the resize, with no implied `/ 255`. | You must match a training pipeline exactly, for example a BF16 model whose reference preprocessing is `pixel * (1/255)` followed by mean and stddev. Image route only. |
| Quantization step | `quantization_step` | `quantization_step` | Not set (`None`) | The model's original INT8 step (a positive float). Core checks it against the package's quantizer and fails if they disagree. | With fixed source arithmetic on an INT8 model. Leave it unset otherwise. |
| YUV conversion order | `color_conversion_placement` | `color_conversion_placement` | `PreprocColorConversionPlacement::AfterResize` | For YUV input (NV12, I420), whether YUV-to-RGB conversion happens after the resize (default) or before it. Other color inputs reject `BeforeResize`. | You feed NV12 from a camera and must match a reference that converts to RGB before resizing. |
| Float rounding | `floating_conversion` | `floating_conversion` | `PreprocFloatingConversion::Legacy` | How values are rounded to BF16: `Legacy` truncates, `NearestEven` rounds to nearest. | A BF16 model whose accuracy check against a PyTorch reference is off by rounding. Selects fixed source arithmetic. |

The last four rows are fixed source arithmetic (see Core's
[Fixed source arithmetic](https://developer.sima.ai/software/reference/nodes/preproc#fixed-source-arithmetic)).
Any of them, a typed `resize.interpolation`, `Crop` or `shorter_side` needs an
image route with preprocessing on, no `transforms` list, and `quantize` and
`tessellate` left at `Auto`; otherwise the `Model` constructor throws.

### Resize: `ResizeSpec`

Python `pyneat.ResizeSpec`.

| Field | Default | What it controls | When to set it |
|---|---|---|---|
| `enable` | `Auto` | `On` forces the resize. `Auto` turns it on when you set a size, `Crop`, a typed interpolation or the `COCO_YOLO` preset. | Set `On` together with `mode`, so the mode is applied; see Pitfalls. |
| `width`, `height` | `0`: taken from the model input | The size the image is resized to. | Leave `0` so the planner uses the model's input size. The examples that set it (depth estimator, MIPI multi-model) set the model's own size. |
| `mode` | `ResizeMode::Letterbox` | How the image fits the model input: see the [`ResizeMode` values](#resizemode-values). | Match the training pipeline: `Letterbox` for YOLO, `Stretch` for TensorFlow SSD models (MIPI multi-model example) and the depth estimator, `Crop` with `shorter_side` for the ImageNet "resize 256, crop 224" recipe. |
| `pad_value` | `114` | Fill value of letterbox padding (114 is mid-gray, the YOLO convention). | A model trained with another pad, such as `0` (black). |
| `scaling_type` | `"BILINEAR"` | Legacy string for the sampler: `BILINEAR`, `NEAREST_NEIGHBOUR`, `BICUBIC`, `INTERAREA`, `NO_SCALING`. | Prefer `interpolation` in new code. The depth estimator sets `"BICUBIC"`. |
| `interpolation` | Not set (`None`): `scaling_type` applies | Typed sampler: `ResizeInterpolation::Nearest`, `Bilinear`, `Bicubic`, `BoxAverage`, `BilinearAntialiased`. A value that conflicts with a non-default `scaling_type` is rejected. | A classifier whose reference preprocessing resizes with Pillow or torchvision, which antialias. |
| `shorter_side` | Not set (`None`) | With `Crop` only: resize the shorter side to this many pixels (1–4096) before the center crop. | The classic ImageNet "resize 256, crop 224" pipeline. |
| `placement` | `LetterboxPlacement::Center` | Where a letterbox puts the image: see the [`LetterboxPlacement` values](#letterboxplacement-values). | Segment Anything models need `TopLeft`. |

### Color: `ColorConvertSpec`

Python `pyneat.ColorConvertSpec`.

| Field | Default | What it controls | When to set it |
|---|---|---|---|
| `enable` | `Auto` | Turns color conversion on. `Auto` turns it on when you set a format. | Leave it. |
| `input_format` | `PreprocessColorFormat::Auto`: RGB | The pixel format you send. It must match the format the input tensor is tagged with, or `run` fails with `Tensor image format mismatch`. | Every image route: `BGR` for OpenCV, `NV12` for video and cameras. |
| `output_format` | `Auto`: what the model takes | The format handed to the model. | The MIPI multi-model example sets `RGB` explicitly after NV12 input. Usually leave it. |

### Layout: `LayoutConvertSpec`

Python `pyneat.LayoutConvertSpec`. `perm` lists the source axis for each output
axis, for example `{2, 0, 1}` turns height-width-channel into
channel-height-width. Setting `perm` turns the step on; `enable = On` with an
empty `perm`, a negative index or a repeated index throws. No shipped example
sets it.

### Normalize: `NormalizeSpec`

Python `pyneat.NormalizeSpec`.

| Field | Default | What it controls | When to set it |
|---|---|---|---|
| `enable` | `Auto` | `Auto` turns normalization on only when `has_explicit_stats` is true or a preset is set. | Leave it, or set `On`. |
| `mean` | `{0, 0, 0}` | Per-channel value subtracted after scaling pixels to 0–1. Python takes a list of three floats. | Your model's training statistics. |
| `stddev` | `{1, 1, 1}` | Per-channel divisor. Must be greater than 0. | Your model's training statistics. |
| `has_explicit_stats` | `false` | Marks `mean` and `stddev` as yours. Without it, `Auto` leaves normalization off and a preset overwrites `mean` and `stddev`. | Set it to `true` whenever you set `mean` or `stddev`; see Pitfalls. |

### Quantize and tessellate: `QuantizeSpec`, `TessellateSpec`

Python `pyneat.QuantizeSpec` and `pyneat.TessellateSpec`. They describe the
last steps before the MLA: INT8 conversion (`scale`, `zero_point`,
`output_dtype`) and the tile layout (`slice_shape`). The package owns these
values: on load, Neat fills them from the package and replaces anything you
set. An application leaves both at `Auto`.

### What "auto" decides

With every field at its default, Neat keeps only the steps the package
declares. For the YOLO26n and ResNet-50 packages that is just quantize, so
the model wants a float32 tensor that you resized and normalized yourself:
`[1, 640, 640, 3]` for YOLO26n and `[1, 224, 224, 3]` for ResNet-50. The plan
reports `graph_family = Disabled`, and `info().pre_kernels` lists `quantize`.

With `kind = Image`, Neat adds an EV74 image stage (`graph_family =
Preproc`). Then:

- the image may have any size up to `input_max_width` × `input_max_height`,
  and `input_specs()` reports `[-1, -1, 3]` UInt8;
- it is always fitted to the model's input size; when you leave `resize`
  unset, ResNet-50 on a DevKit was letterboxed with pad `0`;
- color conversion runs from `input_format` (RGB if unset);
- normalization runs only if you set statistics or a preset;
- quantize and tessellate follow the package.

To see what was decided, read the plan after loading:

| Question | C++ | Python |
|---|---|---|
| What must I send? | `model.input_specs()` | `model.input_specs()` |
| Was an image stage added, and which steps are on? | `model.resolved_preprocess_plan()`: `graph_family`, `effective.resize`, `effective.normalize`, `warnings` | `model.preprocess_plan()`, same fields |
| One-line text of the plan | `plan.to_debug_string()` | `repr(plan)` or `plan.to_debug_string()` |
| What does the package itself do before the MLA? | `model.preprocess_requirements()`, `model.info().pre_kernels` | same names |

These methods belong to `Model`; see [model.md](model.md). The plan's
`ingress_contracts` describe the package's tensor input even on an image route;
use `input_specs()` to know what to send.

### `InputKind` values

| C++ | Python | Meaning |
|---|---|---|
| `InputKind::Auto` | `pyneat.InputKind.Auto` | The planner decides: image if any image step is set, otherwise the package's route. |
| `InputKind::Image` | `pyneat.InputKind.Image` | You send decoded images (UInt8 pixels in a tagged format). Neat adds the EV74 image stage. |
| `InputKind::Tensor` | `pyneat.InputKind.Tensor` | You send tensors already shaped like the model input. Neat only adapts them (quantize, tessellate). |

### `NormalizePreset` values

| C++ | Python | Meaning |
|---|---|---|
| `NormalizePreset::None` | `getattr(pyneat.NormalizePreset, "None")`: `None` is a Python keyword, so `pyneat.NormalizePreset.None` is a syntax error | No preset; use the `normalize` fields. |
| `NormalizePreset::ImageNet` | `pyneat.NormalizePreset.ImageNet` | Turns normalization on with mean `{0.485, 0.456, 0.406}` and stddev `{0.229, 0.224, 0.225}`, unless you set explicit statistics. Does not change the resize. |
| `NormalizePreset::COCO_YOLO` | `pyneat.NormalizePreset.COCO_YOLO` | Turns the resize on and forces `Letterbox` with pad 114, and turns normalization on with mean `{0, 0, 0}` and stddev `{1, 1, 1}` (pixels scaled to 0–1). |

### `PreprocessColorFormat` values

| C++ | Python | Meaning |
|---|---|---|
| `PreprocessColorFormat::Auto` | `pyneat.PreprocessColorFormat.Auto` | Not specified. As input, it means RGB; as output, what the model takes. |
| `RGB` | `pyneat.PreprocessColorFormat.RGB` | Packed 8-bit RGB. |
| `BGR` | `pyneat.PreprocessColorFormat.BGR` | Packed 8-bit BGR, OpenCV's order. |
| `GRAY8` | `pyneat.PreprocessColorFormat.GRAY8` | 8-bit grayscale, one channel. |
| `NV12` | `pyneat.PreprocessColorFormat.NV12` | YUV 4:2:0, a Y plane and an interleaved UV plane. What video decoders and cameras produce. |
| `I420` | `pyneat.PreprocessColorFormat.I420` | YUV 4:2:0 with three separate planes. |
| `NV16` | `pyneat.PreprocessColorFormat.NV16` | YUV 4:2:2, a Y plane and an interleaved UV plane. Some cameras produce it. |

### `ResizeMode` values

`ResizeMode` is declared in `include/nodes/sima/PreprocMath.h`.

| C++ | Python | Meaning |
|---|---|---|
| `ResizeMode::Stretch` | `pyneat.ResizeMode.Stretch` | Scales width and height separately to the model size; the aspect ratio is not kept. |
| `ResizeMode::Letterbox` | `pyneat.ResizeMode.Letterbox` | Scales to fit, keeping the aspect ratio, and pads the rest with `pad_value`. |
| `ResizeMode::Crop` | `pyneat.ResizeMode.Crop` | Scales until the image covers the model size, then cuts out the center. No padding. |

### `AutoFlag` values

| C++ | Python | Meaning |
|---|---|---|
| `AutoFlag::Auto` | `pyneat.AutoFlag.Auto` | The planner decides from the package and the other options. |
| `AutoFlag::On` | `pyneat.AutoFlag.On` | Force the step on. |
| `AutoFlag::Off` | `pyneat.AutoFlag.Off` | Force the step off. |

### `LetterboxPlacement` values

| C++ | Python | Meaning |
|---|---|---|
| `LetterboxPlacement::Center` | `pyneat.LetterboxPlacement.Center` | Padding split evenly on both sides (YOLO convention). |
| `LetterboxPlacement::TopLeft` | `pyneat.LetterboxPlacement.TopLeft` | Image at the top-left corner, all padding right and bottom, as Segment Anything expects. Needs an enabled `Letterbox` resize. |

### Other enums used by the fields

| Enum (Python name the same) | Values | Used by |
|---|---|---|
| `ResizeInterpolation` | `Nearest`, `Bilinear`, `Bicubic`, `BoxAverage`, `BilinearAntialiased` | `resize.interpolation` |
| `PreprocColorConversionPlacement` | `AfterResize`, `BeforeResize` | `color_conversion_placement` |
| `PreprocFloatingConversion` | `Legacy` (BF16 truncation), `NearestEven` | `floating_conversion` |
| `TransformType` | `Resize`, `ColorConvert`, `LayoutConvert`, `Normalize`, `Quantize`, `Tessellate` | `Transform::type` in `transforms` |

## Language differences

- In Python, an image must reach the model as a tagged tensor:
  `pyneat.Tensor.from_numpy(frame, copy=True,
  image_format=pyneat.PixelFormat.BGR)`. A plain NumPy array on an image route
  fails with `[misconfig.input_shape] ... image-mode Tensor input requires
  explicit image format metadata`. In C++, `from_cv_mat` tags a `cv::Mat` as
  BGR unless you pass another format.
- Python `Model` has `preprocess_plan()` as another name for
  `resolved_preprocess_plan()`, and returns the plan's types from
  `pyneat.advanced` (for example `pyneat.advanced.PreprocessGraphFamily`).
- Optional fields (`rescale_factor`, `quantization_step`,
  `resize.interpolation`, `resize.shorter_side`) are `std::optional` in C++ and
  `None` when unset in Python.

## Pitfalls

- **No normalization gives a wrong answer, not an error.** With
  `kind = Image` and no preset or statistics, the model gets unnormalized
  0–255 pixel values. On a DevKit, ResNet-50 classified a Labrador photo as
  class 253 (basenji, softmax 0.32) instead of 208 (Labrador retriever, 0.83
  with `preset = ImageNet` and otherwise the same options). Fix: set a
  preset or explicit statistics, and check
  `preprocess_plan().effective.normalize.enable` is `On`.
- **Mean and stddev alone do not turn normalization on.** With
  `normalize.enable` at `Auto`, the planner turns normalization on only when
  `normalize.has_explicit_stats` is true or a preset is set. Setting only
  `mean` and `stddev` leaves normalization off, with the symptom above. Fix:
  set `has_explicit_stats = true`, as the FastFlow example does.
- **Swapped colors are silent when the tag is wrong.** Pixels in BGR order
  tagged as RGB are converted as if they were RGB. On a DevKit, ResNet-50 with
  `preset = ImageNet` then answered class 178 (Weimaraner, 0.32) for the
  Labrador. YOLO26n's boxes barely changed, so a detector can hide the
  mistake. Fix: tag the tensor with the order you really have (`cv2.imread`
  gives BGR) and set `input_format` to the same value.
- **A tag that disagrees with `input_format` fails.** A tensor tagged BGR sent
  to a model built with `input_format = RGB` fails in `run` with
  `[misconfig.input_shape] Run::push: simaai::neat::Tensor image format
  mismatch`. Neat does not convert for you.
- **Images larger than the capacity fail.** With `input_max_width` and
  `input_max_height` unset, a 2560 × 1440 frame fails with
  `[misconfig.input_capacity] Run::push: input width 2560 exceeds configured
  capacity 1920`. Other sizes within the capacity work: a
  700 × 577 and a 1280 × 720 frame ran on the same ResNet-50 model. Fix: set
  `input_max_width` and `input_max_height` to your largest source.
- **An unset resize letterboxes with black padding.** With `kind = Image` and
  `resize` left at `Auto`, the image is still fitted to the model, but with
  pad `0`, not the `pad_value` default of 114. On ResNet-50 the scores matched
  an explicit `Letterbox` with `pad_value = 0` exactly, and differed from
  `Stretch` and from `Letterbox` with pad 114. Fix: set `resize.enable = On`
  with the `mode` and `pad_value` your model was trained with.
- **`COCO_YOLO` overrides your resize mode.** The preset sets `Letterbox` and
  pad 114 even if you set `resize.mode = Stretch`; the plan shows `Letterbox`
  and gives no warning. Fix: for another resize mode, leave `preset` at
  `None`, set `resize.enable = On` and `resize.mode`, and set the statistics
  yourself (mean 0 and stddev 1 with `has_explicit_stats = true` is what
  `COCO_YOLO` uses). This fix does not work together with YOLO box decoding.
  On a DevKit with Core 0.6.2, a `Stretch` resize with `decode_type =
  YoloV26` failed when the pipeline was built, with `contract compilation
  failed … BoxDecode kernel recipe: resize: explicit resize does not match
  profile geometry`, also with `boxdecode_resize_mode = Stretch`. Without a
  `decode_type`, the same route ran and returned the raw outputs.
- **The wrong input kind fails clearly.** On the default tensor route, a UInt8
  image fails with `[misconfig.input_shape] Run::push: tensor format does not
  match dtype`. On an image route, a float32 model tensor fails with
  `image-mode Tensor input requires explicit image format metadata`. Check
  `input_specs()` before building inputs.
- **`quantize` and `tessellate` values you set are replaced.** The package's
  scale, zero point and tile shape overwrite them on load. Setting their
  `enable` selects a tensor route, and fixed image options then throw
  `fixed image preprocessing requires the Preproc image route`.

## Used in

- [YOLO26 Object Detector](../../examples/object-detection/yolo26-object-detector/): BGR images, `COCO_YOLO`
- [Single-Stream Object Detector](../../examples/object-detection/single-stream-object-detector/): NV12 or NV16 video frames, input capacity set to the frame size
- [Depth Estimator](../../examples/depth-estimation/depth-estimator/): `Stretch` with an explicit size and `"BICUBIC"`
- [PatchCore Anomaly Detector](../../examples/anomaly-detection/patchcore-anomaly-detector/): `ImageNet` preset, letterbox for a camera
- [FastFlow Anomaly Detector](../../examples/anomaly-detection/fastflow-anomaly-detector/): explicit mean and stddev
- [Image Classification Explorer](../../examples/classification/image-classification-explorer/): `ImageNet` preset with input capacity per model
- [MIPI Multi-Model](../../examples/benchmarking/mipi-multi-model/) (Python): NV12 camera input; `Letterbox` for YOLO26, `Stretch` with mean and stddev 0.5 for SSD
- Other detection, pose, segmentation and tracking examples set `preprocess` too; list them with `git grep -n "preprocess\." -- examples`.

## Source

- C++ header: `include/model/PreprocessPlan.h`, `PreprocessOptions`,
  `ResizeSpec`, `ColorConvertSpec`, `LayoutConvertSpec`, `NormalizeSpec`,
  `QuantizeSpec`, `TessellateSpec`, `Transform` and their enums;
  `include/nodes/sima/PreprocMath.h`, `ResizeMode`, `ResizeInterpolation`,
  `PreprocColorConversionPlacement`, `PreprocFloatingConversion`
- Python binding: `python/src/module.cpp`, `PreprocessOptions`, `ResizeSpec`,
  `ColorConvertSpec`, `LayoutConvertSpec`, `NormalizeSpec`, `QuantizeSpec`,
  `TessellateSpec`, `Transform` and the enums `AutoFlag`, `InputKind`,
  `ResizeMode`, `LetterboxPlacement`, `PreprocessColorFormat`,
  `NormalizePreset`, `TransformType`, `ResizeInterpolation`,
  `PreprocColorConversionPlacement`, `PreprocFloatingConversion`
- Core commit: `6ed16525`
