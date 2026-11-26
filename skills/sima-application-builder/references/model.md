# Model

`Model` loads a compiled model archive and assembles its route: preprocessing, inference, and
box decoding. Use that route as it is. A route rebuilt by hand loses the adapters the archive
requires. `include/model/Model.h` and Core's
`docs/develop-apps/development-workflow/model.mdx` define every option named here.

## Get a compiled model

When the user names a model, look for a compiled package in the
[Model Browser](https://developer.sima.ai/models/) that matches the variant and target.

- `sima-cli models download --id <model-id> --variant <variant>` (sima-cli 2.1.20 or newer)
  reads the `main` branch and writes `models/<model-id>/<variant>/<model-id>_<variant>_default.zip`.
- Unpack it with `unzip -o <zip> -d models/<model-id>/<variant>` and load the
  `<model-id>_<variant>_mpk.tar.gz` inside, not the ZIP.
- `sima-cli models list --branch main --json` lists what `main` publishes. Use only the variants
  the Model Browser offers. A model missing from `main` has not been promoted yet.

## Construct and configure

- Construct `Model(path, options)`. `input_specs()`, `output_specs()`, `metadata()`, and
  `info()` report the input and output contract.
- **Decoded camera or RTSP frames:** set `preprocess.kind = InputKind::Image`,
  `preprocess.enable = AutoFlag::On`, `preprocess.color_convert.input_format =
  PreprocessColorFormat::NV12`, and the normalization `preprocess.preset` the model was trained
  with (`COCO_YOLO` for YOLO, `ImageNet` for RF-DETR). The default `enable` is `Auto`, and both
  reference examples set `On`. The model's preprocessing then resizes and converts each NV12
  frame on the EV74, so the decoder can hand it frames at the stream's own size.
- **Images the application decodes itself,** such as an OpenCV `cv::Mat` or a NumPy array:
  set `preprocess.kind = InputKind::Image`, `preprocess.color_convert.input_format` to the
  image's pixel order (`BGR` for OpenCV), and the normalization the model was trained with,
  through `preprocess.preset` (`COCO_YOLO` for YOLO detection and pose models, `ImageNet` for
  ImageNet-trained models) or explicit `preprocess.normalize` values. Neat never takes
  normalization from the model package, so without either setting the MLA receives raw pixel
  values. That gives no error, only lower accuracy, so confirm that
  `resolved_preprocess_plan().effective.normalize.enable` is `On`.
- **A model with one gray input channel** (SuperPoint) takes no NV12: pull the NV12 frames and
  run their Y plane as `GRAY8` through a `Model::Runner`.
- When frames reach a model through `VideoConvert`, or `SimaDecode` with `raw_output = false`,
  the model failed on a DevKit with `GstSimaMeta needs nonnegative frame-id and timestamp`.
  Convert on the host and run a `Model::Runner`.
- `Stretch` applies only with `resize.enable = AutoFlag::On`, a target size, or an
  `interpolation`. Otherwise the image is letterboxed with black padding. `Crop` turns resize
  on by itself.
- **Tensors that already match `input_specs()`:** leave preprocessing unset. The archive's
  Cast, Quant, and Tess adapters stay automatic, and `preprocess.enable = Off` does not select
  MLA-only execution. Core's `model.mdx` covers raw tensor input.
- Before adding your own preprocessing or postprocessing, check what the route already runs
  with `execution_plan(...)`. The `sima-application-optimizer` skill describes reading it.

## Detection settings

- Detection routes take `decode_type`, `score_threshold`, `nms_iou_threshold`, and `top_k` from
  `Model::Options`. Core's `docs/reference/boxdecode_decode_types.md` defines each value.
  - `score_threshold`: the lowest score a box needs. Zero keeps zero-score boxes. The
    high-density example uses `0.30`.
  - `top_k`: the most boxes per frame. Zero builds only when the model package supplies a cap.
    The high-density example uses `50`.
  - `nms_iou_threshold`: zero keeps the value the model package carries. When the package
    carries none, `YoloV26` skips NMS and the other decoders suppress every overlapping box. The
    high-density example sets `0.60` for `YoloV26` explicitly. Set an explicit value, such as
    `0.45`, for any decoder that runs NMS.
- `decode_type_option` (`BoxDecodeTypeOption` in `pipeline/BoxDecodeType.h`) sets how the head
  tensors are arranged and whether class scores are logits or probabilities. The default,
  `Auto`, decides the score form only when the package's tensor names or score activation
  declare it. When neither does, the scores can be read in the wrong form. The values (Python:
  `pyneat.BoxDecodeTypeOption.<name>`):

  | Value | Head layout and score form it forces |
  |---|---|
  | `Auto` | Taken from the package's tensor contract (default) |
  | `PackedPerHead` | One tensor per head holds box, objectness, and classes |
  | `InterleavedByHead` | Tensors alternate by head: box and class of head 1, then head 2 |
  | `GroupedByRole` | All box tensors, then all class tensors |
  | `Split3Interleaved` | Three split tensors, alternating by head |
  | `Split3Grouped` | Three split tensors, grouped by role |
  | `InterleavedByHeadProbability` | `InterleavedByHead`, class scores already probabilities |
  | `InterleavedByHeadLogit` | `InterleavedByHead`, class scores as logits (before sigmoid) |
  | `GroupedByRoleProbability` | `GroupedByRole`, class scores already probabilities |
  | `GroupedByRoleLogit` | `GroupedByRole`, class scores as logits (before sigmoid) |

  - A decoded score above 1 means the decoder read logits as probabilities. Set the matching
    `...Logit` option, for example `GroupedByRoleLogit` for a YOLOv8 pose package with separate
    box, score, and keypoint heads.
  - Do not clamp scores into [0, 1]. The score threshold is still compared with logits, so a
    clamp only hides the wrong option and drops correct detections.
- Pose decoders: choose by the box heads in `output_specs()`. 64-channel DFL box heads mean
  `YoloV8Pose`, and 4-channel l/t/r/b box heads mean `YoloV26Pose`. `YoloV26Pose` on a YOLOv8
  package returns valid-looking scores and wrong boxes.
- Oriented boxes: `YoloV26Obb` requires `nms_iou_threshold = 0` and a positive `top_k`.
  `YoloV8Obb`, used for YOLOv8 and YOLO11 oriented boxes, runs NMS and needs a nonzero
  threshold. The `object-detection/multi-stream-obb-detector` example switches both values with
  its model family.
- `BoxDecodeType::Ssd` selects the supported prepared SSD recipes. It does not infer a decoder
  from tensor shapes.

## Read outputs

- Outputs carry the model package's logical shape (for example `[1, 1000]` for a classifier).
  A Float32 output's format tag may be `EVXX_FLOAT32` instead of `FP32`, so decode by dtype and
  shape.
- A detection result is a `Sample` of kind `TensorSet` whose first tensor is the `BBOX`
  payload. The reference examples also accept a `Sample` of kind `Tensor`. Decode it with
  `decode_bbox_tensor(tensor, width, height, top_k, strict)`
  (`pyneat.detections.decode_bbox_tensor` in Python). Each `Box` has `x1`, `y1`, `x2`, `y2`,
  `score`, and `class_id`.
  - The boxes are already in the pixels of the frame the model received. `width` and `height`
    are that frame's size, and `decode_bbox_tensor` clamps boxes to it.
  - With `strict = true`, a malformed payload or more boxes than `top_k` raises an error, and
    zero-area boxes are kept. With `false`, the box count is capped at `top_k` and zero-area
    boxes are dropped. The high-density example passes `true` with its `top_k`.
- Class labels come from a label file the application loads, one name per line, in class-ID
  order. `labels[box.class_id]` names a box. The high-density example loads the 80 COCO
  classes this way.
  - When a model scores class IDs with gaps, such as the 91 COCO class IDs DETR models score,
    give every ID a line and write a placeholder such as `N/A` for an ID that names no class.
    The IDs then still index the file.
- Oriented-box decoders emit `BBOX_OBB_V1`. [Application output](application-output.md)
  covers drawing them.
- `decode_superpoint` returns keypoints in model-input pixels until sima-neat/core#1328 is
  fixed. Scale them to the source frame with the width and height of
  `Model::resolved_preprocess_plan().mla_contract`. This holds only after a `Stretch` resize
  (`resize.mode = Stretch` with `resize.enable = On`). A letterbox needs its padding removed
  first.
- **One read per result:** in C++, copy float values with `copy_dense_bytes_tight_to(...)`.
  `data_ptr<T>()` requires mapped CPU storage and throws `tensor storage is not mappable` on
  some outputs, such as the RF-DETR transformer's `Owned` results. In Python,
  `to_numpy(copy=False)` reads a CPU result in place. `copy_payload_bytes()` makes another
  copy.
- `Tensor.from_numpy` rejects a read-only array, such as a NumPy view over the bytes from
  `copy_payload_bytes()`, with `BufferError: Cannot export readonly array`. Pass a writable
  array, for example a `.copy()` of only the plane you need.

## Run the model directly

- `model.run(inputs)` is the shortest synchronous path.
- `model.build(seed, Model::RouteOptions{}, run_options)` returns a `Model::Runner` for many
  pushes and pulls. In Python, call `model.build(seed, run_options=...)`. The `seed` holds
  tensors shaped by `input_specs()`, and the build runs it once. `push(...)` waits while the
  input queue is full. `pull(timeout_ms)` returns an empty `Sample` on a timeout or after the
  runner closes, which is false in Python. `close_input()` sends end-of-stream into the input
  and releases a push that is waiting. `close()` releases the runner.
- To use the model inside a larger `Graph`, add or connect `model.graph()` as one stage, as
  [graph topologies](graph-topologies.md) describes.
