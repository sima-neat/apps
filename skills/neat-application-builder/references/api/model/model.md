# Model

## What it is

`simaai::neat::Model` (Python `pyneat.Model`) loads a compiled model package
and runs it on the board. The package is an MPK, a `*_mpk.tar.gz` archive made
by the SiMa model compiler. It holds the program for the MLA, the board's
machine-learning accelerator, and a contract that lists the tensors the model
takes and returns and the stages it needs before and after the MLA.

Loading does two things. It unpacks the archive into a folder that belongs to
the running process. Then it plans the model's route: the chain of stages that
turns your input into the model's output. A typical route is quantize (convert
float input to the INT8 numbers the MLA reads), MLA inference, then dequantize
(convert the MLA's INT8 output back to float). `Model::Options` changes the
route, for example to add image preprocessing or box decoding. Loading does not
start a pipeline; that happens the first time you run the model.

## When to use it

Every application that runs a compiled model starts with `Model`. After loading,
pick one of three ways to run it:

| Situation | Use |
|---|---|
| A few inputs, a test, or a folder of images handled one at a time | `model.run(inputs)` |
| A loop over many frames, or you need queue settings, several frames in flight, or an explicit close | `model.build(...)`, which returns a `Model::Runner` |
| The model is one stage of a pipeline with a camera or RTSP input, video output or other models | `model.graph(...)` inside a `Graph`, Neat's pipeline of connected nodes; see [routes.md](routes.md) |

`run` is not slow per call: it builds one internal runner on the first call and
reuses it. Its limits are that it always uses default route and run options,
the first call fixes the input shape, and the runner lives until the `Model` is
destroyed. Switch to `build` as soon as you need control over any of these.

To measure how fast the model runs on its own, use `Model::benchmark`; see
[benchmark.md](benchmark.md).

## Minimal example

Both blocks come from the PatchCore example's video-file mode. The first frame
seeds `build`, so the pipeline is planned for that input, and the loop then
calls `runner.run` once per frame. `image_model_options()` and
`make_image_model()` fill `Options::preprocess` for RGB image input (see
[preprocessing.md](preprocessing.md)), and `model_input_tensor` letterboxes a
frame (scales it to fit the model input and pads the rest) into an RGB tensor.

C++, from
[`patchcore-anomaly-detector/src/cpp/main.cpp`](https://github.com/sima-neat/apps/blob/main/examples/anomaly-detection/patchcore-anomaly-detector/src/cpp/main.cpp)
(lines 623–624, 638, 674, de-indented):

```cpp
simaai::neat::Model model(cfg.model_path, image_model_options());
auto runner = model.build(simaai::neat::TensorList{model_input_tensor(frame)});
auto tensors = runner.run(simaai::neat::TensorList{model_input_tensor(frame)});
runner.close();
```

Python, from
[`patchcore-anomaly-detector/src/python/main.py`](https://github.com/sima-neat/apps/blob/main/examples/anomaly-detection/patchcore-anomaly-detector/src/python/main.py)
(lines 542–545, 560, 594, de-indented):

```python
model = make_image_model(cfg.model_path)
runner = model.build(
    [model_input_tensor(frame)], route_options=pyneat.ModelRouteOptions(), run_options=pyneat.RunOptions()
)
outputs = runner.run([model_input_tensor(frame)])
runner.close()
```

For single calibration images the same example skips the runner and calls
`model.run(simaai::neat::TensorList{model_input_tensor(bgr)})` in C++
(line 319) and `model.run([model_input_tensor(bgr)])` in Python (line 279).

Complete flow: [PatchCore Anomaly Detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector)

## Options

`Model::Options` (Python `pyneat.ModelOptions`) holds every option. Pass it to
the constructor, `Model(path, options)`. The `Model` keeps its own copy, so
changing the struct after loading has no effect.

The box-decoding fields (`decode_type`, `decode_type_option`,
`score_threshold`, `nms_iou_threshold`, `top_k`, `superpoint`, `num_classes`,
`yolox_seg_pose`, `boxdecode_original_width`, `boxdecode_original_height`,
`boxdecode_resize_mode` and `multimask_output`) are on
[decoding.md](decoding.md#options). This table has the others.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Preprocessing | `preprocess` | `preprocess` | Empty `PreprocessOptions`: no image preprocessing; the route keeps only the adapters the MPK requires, such as quantize | Describes the input you send (for example a camera frame) and how Neat turns it into the model's input tensor: resize, color conversion, normalization. | See [preprocessing.md](preprocessing.md). |
| Class-map decode | `semantic_segmentation` | `semantic_segmentation` | Off: `enable = false`, `output_width = output_height = 0` (the model input size), `align_corners = false` | Replaces the dequantize or cast postprocess with a decode on the EV74 (the board's vision processor). It resizes the class scores to the output size and returns one class index per pixel: an `[N, H, W]` class map, UINT8 for up to 256 classes, INT32 above. | A SegFormer-style segmentation model whose raw output is per-class scores: enable it to get a class map instead of scores. Set `output_width` and `output_height` both to the camera frame size to label every source pixel. The constructor throws `std::invalid_argument` when the route cannot do it, for example when `decode_type` is set or only one dimension is set. |
| Upstream element name | `upstream_name` | `upstream_name` | `"decoder"` (a video decoder) | Name of the pipeline element the route expects in front of its first stage. | Leave it for `run` and `build`. It matters when you join the model's route to another element by name in a `Graph`; see [routes.md](routes.md), where `RouteOptions::upstream_name` sets it per route. |
| Name suffix | `name_suffix` | `name_suffix` | `""` | Text appended to every pipeline element name the model generates. | You load the same model twice in one process, for example once per camera, and want element names and diagnostics that do not collide. `RouteOptions::name_suffix` does the same per route. |
| Keep unpacked files | `cleanup_extracted_model_data` | `cleanup_extracted_model_data` | `true` | `true` deletes the folder the archive was unpacked into when the process exits. `false` keeps it. | Set `false` while debugging, to read the unpacked contract (`etc/*_mpk.json`) and stage configuration files after the run. Leave `true` in production; see [What loading does](#what-loading-does). |
| Diagnostics | `verbose` | `verbose` | `VerboseOptions{}`: level `Production`, progress messages on, every topic off | Which messages loading and route planning print. Presets are `VerboseOptions::quiet()`, `production()`, `debug_plugins()` and `debug_all()` (Python `pyneat.VerboseOptions.quiet()` and so on). Topics such as `planner`, `plugins` and `gstreamer` can be switched on one at a time. | `quiet()` for a headless service whose log should hold only your own messages, as the High-Density Multi-Stream Object Detector does by default. `planner = true` when the route is not what you expected: it prints the MPK stages in order and the adapters the planner chose. It covers loading only; see Pitfalls. |
| Stop early | `inference_terminal` | `inference_terminal` | Full route: `mla_only = false`, every stop point unset | Ends the route before its last stage. `mla_only` stops after the MLA. `last_stage_index`, `last_stage_name`, `last_plugin_id` and `last_processor` stop after a stage picked by position, name, plugin or processor. Python type: `pyneat.InferenceTerminalPolicy`. | You want the raw MLA output, to debug a postprocess or to feed a next stage that needs only the MLA result. With `mla_only` on a DevKit, ResNet-50 returned one 1008-byte UINT8 buffer instead of 1000 float scores. |
| Stage placement (older form) | `processcvu` | `processcvu` | `pre_run_target = "AUTO"`, `post_run_target = "AUTO"`, `async = true` (Python `async_`) | Where the model's pre- and postprocessing kernels run: `"EV74"`, `"A65"` (the Arm CPU) or `"AUTO"`. `AUTO` runs a stage on the EV74 whenever the EV74 can run it. Only the conversion steps after the MLA, such as dequantize (`Dequant`) and `DetessDequant`, go to the A65 when the A65 can run them. `async = true` submits these stages through the asynchronous path, which favors throughput. `async = false` set here has no effect; see Pitfalls. | Prefer `advanced_execution` in new code. The SuperPoint example sets `post_run_target = "A65"` together with its A65 decoder profile. To force the synchronous path, for example when you suspect the asynchronous one of a problem, set `advanced_execution.preprocess_async = false` instead. |
| MLA execution (older form) | `processmla` | `processmla` | `async = true` (Python `async_`), `output_pool_buffers = 0` (the runtime default of 4), `defer_output_invalidate = true` | `async = true` submits MLA work through the asynchronous path. `async = false` set here has no effect; see Pitfalls. `output_pool_buffers` sets how many MLA output buffers exist. `defer_output_invalidate` is kept for compatibility; `false` set here is ignored. | Raise `output_pool_buffers` when your code holds several model outputs at once, such as the tensors of previous frames, and the MLA has to wait for a free buffer. Prefer `advanced_execution` in new code. |
| Prepared runner (retired) | `prepared_runner` | not available | Empty `mode`, `ring_depth = 0`, `profile = false`, empty `dequant_flags` | Kept only so older C++ code still compiles. Retired modes fail when the pipeline is built. | Never. Leave the default. |
| Internal queue depth (older form) | `async_queue_depth` | not available | `0`: the framework default | Depth of the queues Neat inserts between model stages. | Prefer `advanced_execution.internal_queue_depth`, which is also the Python way to set it. |
| Execution controls | `advanced_execution` | `advanced_execution` | Every field unset: no effect | Gathers the execution settings of the four rows above under plain names; see the table below. A field you set overrides the matching older field, even when you set it to `false`. | The preferred way to change placement, concurrency or buffering. Change one field at a time and compare with the default route, as you would for any tuning. |

`advanced_execution` (`AdvancedExecutionOptions`, Python
`pyneat.AdvancedExecutionOptions`) has these fields. Every field is optional,
and Python shows an unset field as `None`.

| Field | Replaces | What it sets |
|---|---|---|
| `preprocess_target` | `processcvu.pre_run_target` | Where model-managed preprocessing runs: `"AUTO"`, `"A65"` or `"EV74"`. A placement the stage does not support is an error; Neat's own image preprocessing runs only on the EV74. |
| `postprocess_target` | `processcvu.post_run_target` | Where postprocessing adapters such as dequantize run. It does not move box decoding, which runs on the A65. |
| `preprocess_async` | `processcvu.async` (Python `async_`) | Asynchronous pre- and postprocessing stages. |
| `inference_async` | `processmla.async` (Python `async_`) | Asynchronous MLA submissions. |
| `inference_output_buffers` | `processmla.output_pool_buffers` | Number of MLA output buffers. |
| `defer_output_cache_sync` | `processmla.defer_output_invalidate` | Leave unset; `false` is rejected. |
| `prepared_runner` | `prepared_runner` | Leave unset; retired. |
| `internal_queue_depth` | `async_queue_depth` | Depth of the queues between model stages. |

The same execution settings exist on `Model::RouteOptions`. Values given there
when you call `build` or `graph` win over `Model::Options` for that runner or
route; see [routes.md](routes.md).

### What loading does

| C++ | Python | Same as |
|---|---|---|
| `Model(const std::string& model_path)` | `pyneat.Model(model_path)` | Default `Options` |
| `Model(const std::string& model_path, const Options& opt)` | `pyneat.Model(model_path, options)` | — |

The constructor:

1. Checks the archive and unpacks it. Unpacked models live in a
   `proc_run_<random>` folder for the process, in the first writable place
   among: an NVMe drive, `/data/simaai/coprocessing/models/`,
   `/tmp/simaai/coprocessing/models` and `./tmp/model_extract`. Loading the
   same archive again in one process reuses the unpacked copy.
2. Reads the contract and plans the route from it and your `Options`.
3. Prints `Model loaded: <name> (package storage: <kind>, runtime package path:
   <folder>)` unless `verbose` is `quiet()`.

Most failures throw `NeatError` (Python `pyneat.NeatError`, with an
`error_code` attribute). Invalid `semantic_segmentation` settings and retired
execution settings throw `std::invalid_argument` (Python `ValueError`). A
`Model` can be moved but not copied.

### Running: `run`, `build` and the runner

`Model::run` pushes inputs and waits for the result. A `Sample` is Neat's
container for one input or output together with its metadata, such as
timestamps; most applications pass tensors instead.

| C++ | Python | Returns |
|---|---|---|
| `run(const TensorList& inputs, int timeout_ms = -1)` | `run(input, timeout_ms=-1, copy=False)` with a list of `Tensor` or NumPy arrays | `TensorList` (Python: list of `pyneat.Tensor`) |
| `run(const Sample& inputs, int timeout_ms = -1)` | `run(input, ...)` with a list of `pyneat.Sample` | `Sample` |
| `run(const std::vector<cv::Mat>& inputs, int timeout_ms = -1)` (OpenCV builds) | not available | `TensorList` |

The list holds one entry per model input, not a batch. `timeout_ms = -1` waits
forever. On a DevKit, one `[1, 640, 640, 3]` float32 NumPy array into YOLO26n
returned six float32 tensors, the raw detection heads, because `decode_type`
was not set. One `[1, 224, 224, 3]` array into ResNet-50 returned one
`[1, 1000]` float32 tensor of class scores.

`Model::build` returns a `Runner` that you keep for the whole loop:

| C++ | Python | What it plans the pipeline for |
|---|---|---|
| `build()` | `build()` | Placeholder inputs made from the model's input description |
| `build(const RouteOptions& opt)` | `build_with_route_options(options)` | Placeholder inputs, with route options |
| `build(const RunOptions& run_opt)` | not available | Placeholder inputs, with run options |
| `build(const RouteOptions& opt, const RunOptions& run_opt)` | not available | Placeholder inputs, with both |
| `build(const TensorList& inputs, opt = {}, run_opt = {})` | `build(input, route_options=ModelRouteOptions(), run_options=RunOptions(), copy=False)` | Your inputs: their shape and pixel format are checked and fixed at build time |
| `build(const Sample& inputs, opt = {}, run_opt = {})` | Same function, with a list of `Sample` | Your inputs, with their per-buffer metadata |
| `build(const std::vector<cv::Mat>& inputs, opt = {}, run_opt = {})` (OpenCV builds) | not available | Your images |

Seed `build` with a real input when the route takes images: pass a frame of the
size you will send, as the examples do. `RouteOptions` belongs to
[routes.md](routes.md). `RunOptions`, such as the queue depth and what happens
when the queue is full, belongs to the `run/` pages.

The runner methods an application uses:

| C++ | Python | What it does |
|---|---|---|
| `run(inputs, timeout_ms = -1)` | `run(input, timeout_ms=-1, copy=False, layout=None, image_format=None)` | Pushes one input and waits for its result. Same input types as `Model::run`. |
| `push(inputs)` | `push(input, ...)`, `push_tensors(inputs)`, `push_samples(inputs)` | Submits an input without waiting for the result. Waits for queue space and returns `false` if the input is rejected. |
| `try_push(inputs)` | `try_push_tensors(inputs)`, `try_push_samples(inputs)` | Submits without waiting for queue space; a full queue follows `RunOptions::overflow_policy`. |
| `pull(timeout_ms = -1)` | `pull(timeout_ms=-1)` | Returns the next result as a `Sample`. Empty when the timeout passed or the runner closed. |
| `close_input()` | `close_input()` | Tells the pipeline no more input is coming, so `pull` can drain the results still in flight. |
| `close()` | `close()` | Ends the input and releases the pipeline. The runner cannot be used afterwards. |
| `operator bool` | `bool(runner)` | `true` while the runner can push and pull. |
| `start_measurement(...)` | `start_measurement(...)` | Measures your own push and pull loop; see the `outputs/` pages. |

Use `runner.run` when each input should wait for its own result. Use `push` and
`pull` when you want to keep several frames in flight.

Setup cost on a DevKit, indicative only: loading took a fraction of a second,
the first `model.run` and a `build` each took a few hundred milliseconds, and
later `run` calls took milliseconds. Load once and build once, outside the frame
loop.

### What the model tells you

These calls ask a loaded model about itself. All of them except
`preprocess_requirements()` appear in the examples.

| C++ | Python | Returns | Use it to |
|---|---|---|---|
| `input_specs()` | `input_specs()` | One `TensorSpec` per input (Python `pyneat.TensorConstraint`): `shape`, `dtypes`, `rank`, `image_format` | Make inputs of the right shape and type. On a DevKit, YOLO26n wants `[1, 640, 640, 3]` and ResNet-50 `[1, 224, 224, 3]`, both float32, in height-width-channel order with a batch axis of 1. The batch axis can be left out: `[224, 224, 3]` also worked. |
| `output_specs()` | `output_specs()` | One `TensorSpec` per output | Read output sizes before the first run, as FastFlow does for its anomaly-map side. |
| `compiled_batch_size()` | `compiled_batch_size()` | The batch size the model was compiled for; `1` when not batched | Size the batch axis of your inputs. |
| `info()` | `info()`, and `summary()` as text | `ModelInfo`: model name, contract path, which adapter stages the model needs and the planner chose, output topology, planner warnings | Check what the planner did when the output is not what you expected. |
| `preprocess_requirements()` | `preprocess_requirements()` | `PreprocessRequirements`: whether the route preprocesses, whether it quantizes or tessellates (rearranges the tensor into the MLA's tile layout), expected input format, and the shape and type handed to the MLA | Repeat the model's preprocessing in your own code when you feed inputs Neat cannot preprocess for you. |
| `resolved_preprocess_plan()` | `resolved_preprocess_plan()` or `preprocess_plan()` | The full preprocessing plan | Read the size the MLA receives, as the SuperPoint example does; see [preprocessing.md](preprocessing.md). |

`Model` also has `metadata()`, `execution_plan(...)`, and stage helpers such as
`fragment(...)` for debugging and for building your own `Graph`; see
[routes.md](routes.md).

## Language differences

- Python takes inputs as a list or tuple, one entry per model input. An entry
  can be a `pyneat.Tensor` or a NumPy array; `copy=True` copies the array
  instead of sharing its memory. A single `Tensor` or array that is not in a
  list is rejected. C++ takes a `TensorList`, a `Sample`, or, in OpenCV builds,
  a `std::vector<cv::Mat>`.
- Python has no `build` form that takes `RunOptions` without an input. To pass
  `RunOptions` in Python, seed `build` with an input. Python spells
  `build(RouteOptions)` as `build_with_route_options(options)`.
- Python splits the runner's `push` and `try_push` overloads by input type:
  `push_tensors`, `push_samples`, `try_push_tensors` and `try_push_samples`.
  `push` and `run` also take `layout` and `image_format` to describe a NumPy
  array.
- Python accepts a `str` or an `os.PathLike` model path.
- Python has `summary()`, a text version of `info()`, and `preprocess_plan()`,
  another name for `resolved_preprocess_plan()`. C++ has
  `input_appsrc_options_list()` for models with several inputs.
- `run`, `push` and `pull` release the GIL in Python, so other Python threads
  keep running while they wait. `build` holds it.
- Failures throw `NeatError` in C++ and `pyneat.NeatError` in Python, which
  has an `error_code` attribute. Some input checks throw `std::runtime_error`,
  which is `RuntimeError` in Python.
- Python names `InferenceTerminalPolicy` and `PreprocessRequirements` without
  the `Model` prefix the naming rules predict.
- A retired execution setting, such as `defer_output_cache_sync = false`,
  throws `std::invalid_argument` in C++ and `ValueError` in Python.

## Pitfalls

- **A bare array is not a list.** `model.run(array)` in Python throws
  `RuntimeError: expected list/tuple of Tensor or DLPack-compatible inputs`,
  and a bare `Tensor` throws `Model.run expects TensorList; pass [tensor]
  instead of a single Tensor`. Fix: `model.run([array])`.
- **Input shape and type must match `input_specs()`.** On a DevKit, a
  `[1, 100, 100, 3]` array into ResNet-50 failed with `NeatError`
  `[misconfig.media_caps] Two connected pipeline stages require incompatible
  media caps`. The message does not mention the shape, so check
  `input_specs()` first. A `uint8` array where the spec says float32 failed
  only when it ran, with `[runtime.kernel_binding]`. Two inputs into a
  one-input model fail with `[misconfig.input_shape] Model::build: expected
  exactly 1 ingress tensor inputs (pixel_values), got 2`.
- **The first `run` fixes the input shape.** After a successful
  `model.run`, a call with another shape failed with
  `[runtime.element_failed] InputStream::try_push_message: tensor changes are
  not supported`. A failed call drops the internal runner, and the next call
  builds a new one. Fix: send one size, resizing before the call, or use one
  `Model` per input size.
- **A closed runner stays closed.** After `runner.close()`, `bool(runner)` is
  `False` and `runner.run(...)` throws `RuntimeError: [runtime.pull] Run::push:
  stream is closed`. Build a new runner instead.
- **Path errors appear at load time, not at run time.** A missing file throws
  `NeatError` `[io.parse] ... archive path does not exist or is not a regular
  file`, and a file that is not a `.tar.gz` throws `[io.parse] ... failed to
  decompress archive`.
- **The default timeout waits forever.** `timeout_ms = -1` blocks until a
  result arrives. Pass a timeout, such as the 2000 ms the Core docs use, when a
  stuck pipeline must not hang your application.
- **`verbose` does not silence the build.** With
  `verbose = VerboseOptions::quiet()` the `Model loaded` line disappears, but
  `build` and the first `run` still print `[1/4] Initializing runtime...` to
  `[4/4] Graph ready`. Those lines follow `RouteOptions::verbose`, which only
  C++ has.
- **Kept folders are never deleted for you.** With
  `cleanup_extracted_model_data = false` the process folder stays after exit,
  with a `.sima_modelpack_keep` marker, until you delete it. Setting `false` on
  one `Model` keeps the folder for every model in the process. On the
  DevKit used here the folder was under `/tmp`, a memory-backed file system,
  so it is also gone after a reboot.
- **The older `async = false` switches have no effect.** When a runner or
  route is built, the route copies `processcvu.async` and `processmla.async`
  only when they are `true`, and the route's own default is `true`. On a
  DevKit, `processcvu.async_ = False` and `processmla.async_ = False` on
  `ModelOptions` still gave `async=true` on every element in
  `describe_backend()`. Fix: set `advanced_execution.preprocess_async = false`
  and `inference_async = false`, which gave `async=false`.
- **An unknown placement target is silently `AUTO`.**
  `advanced_execution.preprocess_target = "GPU"` built without an error; Core
  treats every unknown value as `"AUTO"`. Fix: write exactly `"AUTO"`,
  `"A65"` or `"EV74"`.
- **A target the stage cannot use fails at build.** On a YOLO26n image route,
  `preprocess_target = "A65"` on the route threw `ValueError: processcvu stage
  cannot run on explicitly requested target A65 (source=processcvu_pre).
  Supported target(s): EV74.` Neat's image preprocessing runs only on the
  EV74.
- **`defer_output_cache_sync = false` is rejected.** It threw `ValueError:
  processmla.defer_output_invalidate=false is retired (including
  advanced_execution.defer_output_cache_sync=false)`. Leave it unset.

## Used in

- One-shot `run`:
  [FastFlow Anomaly Detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/fastflow-anomaly-detector),
  [Image Classification Explorer](https://github.com/sima-neat/apps/tree/main/examples/classification/image-classification-explorer),
  [PatchCore Anomaly Detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector)
  (calibration)
- `build` and a runner:
  [Depth Estimator](https://github.com/sima-neat/apps/tree/main/examples/depth-estimation/depth-estimator),
  [PatchCore Anomaly Detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector)
  (video file),
  [YOLO26 Object Detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/yolo26-object-detector),
  [SSD-MobileNet Object Detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/ssd-mobilenet-object-detector),
  [DETR Object Detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/detr-object-detector),
  [PCB Defect Detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/pcb-defect-detector)
- `Model` inside a `Graph`: most streaming examples; see [routes.md](routes.md).

## Source

- C++ header: `include/model/Model.h`, `Model`, `Model::Options`,
  `Model::SemanticSegmentationOptions`, `Model::InferenceTerminalPolicy`,
  `Model::PreprocessRequirements`, `Model::ModelInfo` and `Model::Runner`;
  `include/pipeline/GraphOptions.h`, `VerboseOptions`, `ProcessCvuOptions`,
  `ProcessMlaOptions`, `PreparedRunnerOptions` and `AdvancedExecutionOptions`;
  `include/pipeline/TensorSpec.h`, `TensorConstraint`
- Python binding: `python/src/module.cpp`, `ModelOptions`,
  `ModelSemanticSegmentationOptions`, `InferenceTerminalPolicy`,
  `VerboseOptions`, `ProcessCvuOptions`, `ProcessMlaOptions`,
  `AdvancedExecutionOptions`, `Model`, `ModelRunner`, `TensorConstraint`,
  `ModelInfo` and `PreprocessRequirements`
- Core commit: `6ed16525`
