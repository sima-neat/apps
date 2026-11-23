# BoxDecodeType

## What it is

A detection model does not output boxes. It outputs raw head tensors: grids
of box distances and class scores, laid out the way the model family was
exported. Box decoding turns them into a list of detections, each with a box,
a score and a class, in the pixels of the image you sent.

In Neat, `Model::Options::decode_type` (Python `pyneat.ModelOptions`) picks a
`BoxDecodeType`, the decoder for one model family such as YOLO26, YOLOv8, SSD
or DETR. The `Model` then adds a BoxDecode stage after the MLA (the board's
machine-learning accelerator) to its route, the chain of stages it runs. On a
DevKit this stage ran on the A65, the board's Arm CPU. The route then returns
one packed byte tensor instead of the raw heads. The `decode_*` helper
functions read that tensor in your application and give you boxes, keypoints,
masks or feature points.

## When to use it

Set `decode_type` whenever a detection, pose, segmentation, oriented-box or
SuperPoint model has a matching `BoxDecodeType` and your application needs
boxes or points. Then read the result with the helper for that output, for
example `decode_bbox_tensor` for boxes or `decode_pose` for keypoints. The
helpers do not decode raw heads: they only read what the BoxDecode stage
produced. To decode raw heads you already hold, C++ has
`stages::BoxDecode(sample, model, options)` and `stages::BoxDecodeResults(...)`,
which run the same stage outside the route; Python does not bind them, and no
shipped example uses them.

Leave `decode_type` unset when no `BoxDecodeType` fits your model, or when you
want the raw tensors for your own postprocessing. The Face Recognizer does
this for SCRFD, whose anchors and landmarks it decodes in its own code. The
route then returns the model's output tensors as they leave the MLA stage
(after dequantize).

The same options work when the model runs inside a `Graph`. A graph can also
place a standalone BoxDecode node (`nodes::SimaBoxDecode`); see
[routes.md](routes.md).

## Minimal example

Both blocks come from the `model_options` function of the PCB Defect
Detector, which runs a YOLO26 model. The lines before them in that function
fill the image preprocessing (see [preprocessing.md](preprocessing.md)), and
the function returns the options that the example passes to the `Model`
constructor ([model.md](model.md)).

C++, from
[`pcb-defect-detector/src/cpp/main.cpp`](../../../examples/object-detection/pcb-defect-detector/src/cpp/main.cpp)
(lines 71–76, de-indented):

```cpp
options.decode_type = neat::BoxDecodeType::YoloV26;
options.score_threshold = static_cast<float>(config.number("inference.min_score"));
options.nms_iou_threshold = static_cast<float>(config.number("inference.nms_iou"));
options.top_k = max_detections;
// Loading fails when the label file's class count differs from the model's.
options.num_classes = static_cast<int>(labels.size());
```

Python, from
[`pcb-defect-detector/src/python/main.py`](../../../examples/object-detection/pcb-defect-detector/src/python/main.py)
(lines 53–58, de-indented):

```python
options.decode_type = pyneat.BoxDecodeType.YoloV26
options.score_threshold = config.number("inference.min_score")
options.nms_iou_threshold = config.number("inference.nms_iou")
options.top_k = max_detections
# Loading fails when the label file's class count differs from the model's.
options.num_classes = len(labels)
```

The example reads each result in its `detect_defects` function: it calls
`decode_bbox_tensor` (Python `pyneat.detections.decode_bbox_tensor`) on the one
output tensor with the image's width and height, `max_detections` as
`expected_topk`, and `strict` on, which gives a list of `Box` records in the
image's pixels.

On a DevKit, the YOLO26n package with no `decode_type` returned six float32
tensors: three `[1, H, W, 4]` box heads and three
`[1, H, W, 80]` class heads. With `YoloV26`, `top_k = 100` and
`score_threshold = 0.25`, the same image returned one 2404-byte UInt8 tensor
tagged `BBOX`, which held two boxes.

Complete flow: [PCB Defect Detector](../../../examples/object-detection/pcb-defect-detector/)

## Options

These are the decoding fields of `Model::Options` (Python
`pyneat.ModelOptions`); [model.md](model.md) has the others. Pass them to the
`Model` constructor. The score threshold, NMS overlap and detection cap have
no effect while `decode_type` is `Unspecified` (see Pitfalls).

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Decoder | `decode_type` | `decode_type` | `BoxDecodeType::Unspecified`: no BoxDecode stage; the route returns raw head tensors | Selects the decoder for the model family and adds the BoxDecode stage. See [`BoxDecodeType` values](#boxdecodetype-values). | Set it for every detection, pose, segmentation, oriented-box or SuperPoint model you want decoded. Pick the value for the family and export of your model, as the model's documentation names it. |
| Head layout | `decode_type_option` | `decode_type_option` | `BoxDecodeTypeOption::Auto`: the package's contract decides | Forces one arrangement of the head tensors when the decoder cannot tell it apart. See [`BoxDecodeTypeOption` values](#boxdecodetypeoption-values). | Leave `Auto` for a model package from the registry. Core's decode-types guide names one case: a YOLOv6 export with grouped `[bbox, class_prob]` outputs needs `GroupedByRoleProbability` and `num_classes`. |
| Score threshold | `score_threshold` | `score_threshold` | `0.0`: no score filter | Drops detections whose score is below this value. `0` keeps zero-score detections; NMS and `top_k` still apply. For `SuperPoint`, `0` takes the profile's default, for example 0.1 for `A65V1`. | Set it to the confidence below which a detection is noise for your application: the YOLO26 Object Detector uses 0.40, the PCB Defect Detector 0.25, the SSD-MobileNet Object Detector 0.55. Raise it when a live display shows false boxes. Lower it only when you rank or filter the boxes yourself. |
| NMS overlap | `nms_iou_threshold` | `nms_iou_threshold` | `0.0`. For YOLO26 (`YoloV26`, `YoloV26Pose`, `YoloV26Seg`) this turns NMS off. It does not turn NMS off for the other detectors: YOLOv5 to v10, YOLOX, EfficientDet, SSD and `YoloV8Obb` always run NMS, and at `0` it removes every box that overlaps a higher-scoring box of the same class, even slightly. With the default, these decoders silently lose objects that overlap (see Pitfalls). | Non-maximum suppression (NMS), the step that removes duplicate boxes, removes a box that overlaps a higher-scoring box of the same class by more than this IoU (intersection over union, 0–1). | Set 0.45 to 0.6 for every decoder that runs NMS, as the YOLO26 examples do. Raise it when real objects overlap, such as a crowd, so their boxes are kept. Lower it when the same object gets two boxes. Leave `0` for `YoloV26Obb`, `Detr` and `DetrFocal`, which apply no NMS. |
| Detection cap | `top_k` | `top_k` | `0`: unset, and box routes fail on it (see Pitfalls) | The most detections one frame can return. It also sizes the output tensor: 4 + 24 × `top_k` bytes for plain boxes. | Always set it on a box route, to the most objects you expect in one frame: 50 in the YOLO26 Object Detector, 300 in the PCB Defect Detector for many small defects. For `SuperPoint`, `0` takes the profile default (600 for `A65V1`). |
| SuperPoint settings | `superpoint` | `superpoint` | `SuperPointOptions{}`: profile `Auto`, `nms_radius = -1`, `border_margin = -1`, Float32 descriptors, `FeaturePointsV1` output | Settings of the `SuperPoint` decoder. See [SuperPoint and YOLOX settings](#superpoint-and-yolox-settings). | Only with `decode_type = SuperPoint`. The SuperPoint Feature Extractor sets the `A65V1` profile, `FeaturePointsV1` output and Float32 descriptors explicitly. |
| Class count | `num_classes` | `num_classes` | `0`: taken from the model package | The number of classes the detection heads produce. | Set it when the package cannot give the right count. The Multi-Stream Multi-Model example sets `1` for YOLO pose, whose package declares 80 detector classes for single-class pose heads. For every YOLO26 type, `YoloV5` and `YoloXSegPose`, a value that differs from the heads fails, so leave `0` there unless you want that check. Other types print a warning and use your value. |
| YOLOX keypoint classes | `yolox_seg_pose` | `yolox_seg_pose` | `YoloXSegPoseOptions{}`: empty `pose_classes`, every class has keypoints | Settings of the `YoloXSegPose` decoder. See [SuperPoint and YOLOX settings](#superpoint-and-yolox-settings). | Only with `YoloXSegPose`, for a model whose classes mix objects with and without keypoints. |
| Source width (deprecated) | `boxdecode_original_width` | `boxdecode_original_width` | `0`: use the size the route's preprocessing records | Width, in pixels, of the image the boxes are mapped back to. Set it with `boxdecode_original_height`. | Leave it `0` on an image route, which knows the frame size (see Pitfalls). Set it on a tensor route, where you preprocess the image yourself and Neat cannot know the source size; without it the first run fails. |
| Source height (deprecated) | `boxdecode_original_height` | `boxdecode_original_height` | `0` | Height of the image the boxes are mapped back to. | Same cases as the source width. |
| Decoder resize policy | `boxdecode_resize_mode` | `boxdecode_resize_mode` | Not set (`None`): the route's own resize policy | The `ResizeMode` (see [preprocessing.md](preprocessing.md)) the decoder assumes when it maps boxes back. It cannot contradict a resize the route performs. | On a tensor route, set the mode you used to fit the image to the model, such as `Letterbox`. On a DevKit tensor route with the source size set and this unset, boxes matched an explicit `Letterbox`. |
| Segment Anything mask choice | `multimask_output` | `multimask_output` | `true` / `True` | For the Segment Anything decoders (`MobileSam`, `EdgeTamSeg`, `EdgeTamTrack`, `EfficientSam`, `EfficientSam3`): `true` keeps the candidate mask with the best predicted IoU, `false` the single-mask answer. | `false` for a box prompt or several points, `true` for one click. Other decode types, `EfficientSam`, and three-mask `EfficientSam3` decoders reject `false`. |

`semantic_segmentation`, the class-map decode for segmentation models such as
SegFormer, is not a box decoder; see [model.md](model.md).

### `BoxDecodeType` values

Python uses the same names: `pyneat.BoxDecodeType.YoloV26` and so on. The
"Route output" column names the tag on the output tensor and the helper that
reads it (see [Reading the result](#reading-the-result-the-decode-helpers)).

| Value | Model family | Route output |
|---|---|---|
| `Unspecified` | No decoder (default) | The model's raw output tensors |
| `Yolo` | YOLO exports whose graph already decodes boxes to center, width and height in model pixels, such as YOLOv6 packs with in-graph box decoding | `BBOX`: `decode_bbox` |
| `YoloV5` | YOLOv5 detection: exactly three packed heads (P3, P4, P5) with the standard YOLOv5 anchors. Custom anchor tables are not supported | `BBOX`: `decode_bbox` |
| `YoloV5Seg` | YOLOv5 instance segmentation | `BBOX_SEGMENTATION`: `decode_segmentation` |
| `YoloV7` | YOLOv7 detection | `BBOX`: `decode_bbox` |
| `YoloV7Seg` | YOLOv7 instance segmentation | `BBOX_SEGMENTATION`: `decode_segmentation` |
| `YoloV8` | YOLOv8 detection | `BBOX`: `decode_bbox` |
| `YoloV8Seg` | YOLOv8 instance segmentation | `BBOX_SEGMENTATION`: `decode_segmentation` |
| `YoloV8Pose` | YOLOv8 pose estimation | `BBOX_POSE`: `decode_pose` |
| `YoloV9` | YOLOv9 detection | `BBOX`: `decode_bbox` |
| `YoloV9Seg` | YOLOv9 instance segmentation | `BBOX_SEGMENTATION`: `decode_segmentation` |
| `YoloV10` | YOLOv10 detection | `BBOX`: `decode_bbox` |
| `YoloV10Seg` | YOLOv10 instance segmentation | `BBOX_SEGMENTATION`: `decode_segmentation` |
| `Detr` | Classic DETR: box logits and class logits with a background class. Needs a `Stretch` resize and a positive `top_k`; no NMS | `BBOX`: `decode_bbox` |
| `EffDet` | EfficientDet detection | `BBOX`: `decode_bbox` |
| `RcnnStage1` | The region-proposal stage of a two-stage R-CNN | A proposal tensor for the second stage, not `BBOX`; do not read it with `decode_bbox_tensor` |
| `Centernet` | CenterNet detection | `BBOX`: `decode_bbox` |
| `YoloV26` | YOLO26 detection: box-distance heads (left, top, right, bottom) and class heads | `BBOX`: `decode_bbox` |
| `YoloV26Pose` | YOLO26 pose estimation | `BBOX_POSE`: `decode_pose` |
| `YoloV26Seg` | YOLO26 instance segmentation | `BBOX_SEGMENTATION`: `decode_segmentation` |
| `YoloV6` | YOLOv6 with raw box-distance heads and class-logit heads | `BBOX`: `decode_bbox` |
| `YoloX` | YOLOX with separate box, objectness and class heads | `BBOX`: `decode_bbox` |
| `Ssd` | Four prepared SSD models only: SSD300, SSD-Mobile-300, SSD-Mobile-320 and SSDlite-Mobile-320. Needs a `Stretch` resize; any other SSD fails | `BBOX`: `decode_bbox` |
| `SuperPoint` | SuperPoint keypoints and descriptors | `FEATURE_POINTS_V1`: `decode_superpoint` |
| `YoloV26Obb` | YOLO26 oriented (rotated) boxes. No NMS; leave `nms_iou_threshold` at 0 | `BBOX_OBB_V1`: `decode_obb` |
| `YoloXSegPose` | A YOLOX export with box, mask and keypoint heads together | Boxes, masks and keypoints in one tensor: `decode_segmentation_pose` |
| `YoloV8Obb` | YOLOv8 and YOLO11 oriented boxes, with rotated NMS | `BBOX_OBB_V1`: `decode_obb` |
| `DetrFocal` | DETR-style query detectors with normalized boxes and sigmoid class scores, without a background class. Needs a `Stretch` resize; no NMS | `BBOX`: `decode_bbox` |
| `MobileSam` | MobileSAM prompt decoder. Its image encoder needs a top-left letterbox | One mask at the original image size: `decode_segmentation` |
| `EdgeTamSeg` | EdgeTAM mask decoder | One mask: `decode_segmentation` |
| `EfficientSam` | EfficientSAM prompt decoder | One mask: `decode_segmentation` |
| `EfficientSam3` | EfficientSAM3 prompt decoder, or its concept image model (up to `top_k` masks) | Masks: `decode_segmentation` |
| `EdgeTamTrack` | EdgeTAM tracking: as `EdgeTamSeg`, followed by the decoder's raw outputs for the memory encoder | The mask tensor first (`decode_segmentation`), then the raw tensors |

`Ssd`, `Detr` and `DetrFocal` need a `Stretch` resize. Every other box
decoder, including the YOLO types, `EffDet` and `Centernet`, needs a letterbox
resize; a `Stretch` resize fails when the pipeline is built.

Pick the value by the model's family and how it was exported, not by the
number or shape of its outputs. Core's
[decode-types reference](https://developer.sima.ai/software/reference/boxdecode_decode_types)
gives the exact head layout each value expects.

### `BoxDecodeTypeOption` values

Python: `pyneat.BoxDecodeTypeOption.<name>`. A model package from the registry
describes its head layout, so `Auto` is the right value for it. On a DevKit,
YOLO26n gave the same boxes with `PackedPerHead`, `InterleavedByHead` and
`GroupedByRoleProbability` as with `Auto`.

| Value | Head layout it forces |
|---|---|
| `Auto` | The decoder takes the layout from the package's tensor contract (default) |
| `PackedPerHead` | One tensor per head holds box, objectness and classes |
| `InterleavedByHead` | Tensors alternate by head: box and class of head 1, then head 2, and so on |
| `GroupedByRole` | Tensors grouped by role: all box tensors, then all class tensors |
| `Split3Interleaved` | Three split tensors, alternating by head |
| `Split3Grouped` | Three split tensors, grouped by role |
| `InterleavedByHeadProbability` | `InterleavedByHead`, with class scores that are already probabilities |
| `InterleavedByHeadLogit` | `InterleavedByHead`, with class scores as logits (before sigmoid) |
| `GroupedByRoleProbability` | `GroupedByRole`, with class scores as probabilities |
| `GroupedByRoleLogit` | `GroupedByRole`, with class scores as logits |

### SuperPoint and YOLOX settings

`superpoint` is a `SuperPointOptions` (Python `pyneat.SuperPointOptions`);
`yolox_seg_pose` is a `YoloXSegPoseOptions` (Python
`pyneat.YoloXSegPoseOptions`).

| Field | Default | What it sets |
|---|---|---|
| `superpoint.profile` | `SuperPointProfile::Auto`: the profile the package names, else `A65V1` | The numerical rules of the decoder: `LightGlueV1`, `MagicLeapDemoV1`, `A65V1`. `PaperBicubicV1` is rejected for now. |
| `superpoint.nms_radius` | `-1`: profile default (4) | Radius, in pixels, within which only the strongest keypoint is kept. SuperPoint uses this instead of `nms_iou_threshold`. |
| `superpoint.border_margin` | `-1`: profile default (0 for `A65V1`, 4 for the others) | Keypoints closer than this to the image edge are dropped. |
| `superpoint.descriptor_output_dtype` | `TensorDType::Float32` | Type of the returned descriptors. |
| `superpoint.output_format` | `SuperPointOutputFormat::FeaturePointsV1` | `LegacyA65InterleavedV0` keeps an older byte layout and needs Int8 descriptors. |
| `yolox_seg_pose.pose_classes` | Empty: every class has keypoints | Class IDs that have keypoints. Detections of other classes get all-zero keypoints. |

### Reading the result: the decode helpers

The helpers read the packed tensor a BoxDecode route returns, in your
application, on the host CPU. Each one fails on a tensor of another kind,
except `decode_bbox_tensor` with an untagged tensor and `parse_bbox_bytes`
with any bytes (see Pitfalls). Coordinates come out in the source image's
pixels as the route wrote them; `img_w` and `img_h` (Python `clamp_to`) only
clamp boxes to that rectangle.

| C++ | Python | Reads | Returns |
|---|---|---|---|
| `decode_bbox(tensors, img_w = 0, img_h = 0, top_k = 0, strict = false)` | `pyneat.decode_bbox(tensors, *, clamp_to=None, top_k=None, strict=False)` | `BBOX` tensors | One `[N, 6]` float32 tensor per input: `x1, y1, x2, y2, score, class_id` |
| `decode_bbox_tensor(tensor, img_w, img_h, expected_topk, strict)` | `pyneat.detections.decode_bbox_tensor(tensor, img_w, img_h, expected_topk=0, strict=False)` | One `BBOX` tensor | `BoxDecodeResult` |
| `parse_bbox_bytes(bytes, img_w, img_h, expected_topk, strict)` | `pyneat.detections.parse_bbox_bytes(data, img_w, img_h, expected_topk=0, strict=False)` | The raw bytes of a `BBOX` payload | A list of `Box` |
| `decode_pose(tensors, ...)`, `decode_pose_tensor(tensor, ...)` | `pyneat.decode_pose(tensors, *, clamp_to=None, top_k=None, strict=False)` | `BBOX_POSE` | `boxes` `[N, 6]` and `keypoints` `[N, 17, 3]` (`x, y, visibility`) |
| `decode_segmentation(tensors, ...)`, `decode_segmentation_tensor(tensor, ...)` | `pyneat.decode_segmentation(tensors, *, clamp_to=None, top_k=None, strict=False)` | `BBOX_SEGMENTATION`, Segment Anything masks | `boxes` `[N, 6]` and `masks` uint8 `[N, 160, 160]`, or `[N, H, W]` at the original size for Segment Anything |
| `decode_segmentation_pose(tensors, ...)`, `decode_segmentation_pose_tensor(tensor, ...)` | `pyneat.decode_segmentation_pose(tensors, *, clamp_to=None, top_k=None, strict=False)` | `YoloXSegPose` output | `boxes`, `masks` and `keypoints`; row `i` of each is the same detection |
| `decode_obb(tensors)`, `decode_obb_tensor(tensor)` | `pyneat.decode_obb(tensors)`, `pyneat.decode_obb_tensor(tensor)` | `BBOX_OBB_V1` | `geometry` `[N, 5]` (`cx, cy, width, height, angle` in radians), `scores` `[N]`, `class_ids` `[N]` int32 |
| `decode_superpoint(tensors)`, `decode_superpoint_tensor(tensor)` | `pyneat.decode_superpoint(tensors)` | `FEATURE_POINTS_V1` | `keypoints` `[N, 2]`, `scores` `[N]`, `descriptors` `[N, D]` |

- `top_k` and `expected_topk` cap the boxes the helper returns; `0` adds no
  cap. With `strict = true` a payload that holds more boxes than
  `expected_topk`, or a malformed payload, throws.
- Without `strict`, boxes that have zero width or height after clamping are
  dropped. With `strict`, they are kept.
- `read_detection_format(tensor)` (Python
  `pyneat.detections.read_detection_format`) returns the tag, and
  `detection_format_is_bbox(format)` (Python
  `pyneat.detections.format_is_bbox`) checks it.

### `Box` and `BoxDecodeResult`

`decode_bbox_tensor` returns a `BoxDecodeResult` (Python
`pyneat.detections.BoxDecodeResult`) with `boxes`, a list of `Box` (Python
`pyneat.detections.Box`), and `raw`, the payload bytes (Python `bytes`).

| Field | Type | Meaning |
|---|---|---|
| `x1`, `y1` | float | Top-left corner, in pixels |
| `x2`, `y2` | float | Bottom-right corner, in pixels |
| `score` | float | Confidence after NMS, 0–1 |
| `class_id` | int | Class index from the model's label list; `-1` means unset |

The coordinates are in the pixels of the image you sent, not of the model
input. On a DevKit, the same photo at 640 × 425 gave an elephant box of
(66, 116)–(332, 306), and at 1280 × 850 a box of (137, 237)–(664, 622). The
payload stores whole pixels, so the floats have no fractional part.

## Language differences

- Python splits the helpers. `decode_bbox`, `decode_pose`,
  `decode_segmentation`, `decode_segmentation_pose`, `decode_obb`,
  `decode_obb_tensor` and `decode_superpoint` are in `pyneat`. `Box`,
  `BoxDecodeResult`, `decode_bbox_tensor`, `parse_bbox_bytes`,
  `read_detection_format` and the `format_is_*` checks are in
  `pyneat.detections`. C++ `detection_format_is_bbox` is
  `pyneat.detections.format_is_bbox`, and so on for the other formats.
- Python has no `decode_pose_tensor`, `decode_segmentation_tensor`,
  `decode_segmentation_pose_tensor`, `decode_superpoint_tensor` or
  `boxes_to_tensor`. Use the list forms with a one-item list.
- Python has no `stages::BoxDecode`, `stages::BoxDecodeResults` or
  `stages::SuperPointResults`; `pyneat.stages` holds only `preproc`.
- The Python list helpers take the clamp size as one keyword-only tuple,
  `clamp_to=(width, height)`, and `top_k` as a keyword. C++ takes `img_w`,
  `img_h` and `top_k` as positional arguments.
- Errors map to Python types. A tensor whose tag is of another kind raises
  `TypeError` from `decode_bbox`, `decode_pose`, `decode_segmentation` and
  `decode_segmentation_pose`; a payload of the wrong size raises
  `RuntimeError`, and `decode_obb` raises `ValueError`. C++ throws
  `std::runtime_error`, or `std::invalid_argument` from `decode_obb`.
- In Python, `BoxDecodeResult.boxes` and `raw` are read-only, and `raw` is
  `bytes`.

## Pitfalls

- **Decode options without `decode_type` do nothing.** With `score_threshold`,
  `nms_iou_threshold` and `top_k` set but `decode_type` left `Unspecified`,
  YOLO26n on a DevKit returned the six raw head tensors, with no warning. Fix:
  set `decode_type`.
- **The default score threshold returns `top_k` boxes of noise.** With
  `score_threshold = 0` and `top_k = 100`, an image with two real objects
  returned 100 boxes; the third had score 0.035. Fix: set the threshold your
  application needs, for example 0.25.
- **The default NMS overlap drops overlapping objects outside YOLO26.** At
  `nms_iou_threshold = 0`, every decoder that runs NMS except YOLO26 removes
  any box that touches a higher-scoring box of the same class. On a DevKit,
  YOLO26n decoded with `YoloV6` (same head layout) and a score threshold of
  0.01 returned 7 boxes, none overlapping another of its class; at 0.5 it
  returned 10 boxes. `YoloV26` at `0` skipped NMS and returned 17 boxes, some
  of them duplicates. There is no error in either case. Fix: set 0.45 to 0.6.
- **`top_k = 0` fails on a box route.** The first `run` (or `build`) fails
  with `[misconfig.pipeline_shape] ... top_k is unresolved: set a positive
  output capacity`. Fix: set a positive `top_k`.
- **The source size overrides the real frame on an image route.** On a route
  that resizes images itself, `boxdecode_original_width` and
  `boxdecode_original_height` still replace the frame size the route records.
  On a DevKit, a 1280 × 850 frame with these set to 640 × 640 moved the
  elephant box from (137, 237)–(664, 622) to (68, 226)–(331, 418), with no
  error. Fix: leave both at `0` on an image route.
- **A tensor route needs the source size.** When you send a model-shaped
  tensor, the route has no record of the original image, and the first run
  fails with `boxdecode.geometry: required resize-mode metadata is missing`.
  Fix: set `boxdecode_original_width` and `boxdecode_original_height` to the
  source size, and `boxdecode_resize_mode` if you did not letterbox.
- **`img_w` and `img_h` clamp; they do not rescale.** The helpers expect
  boxes already in source pixels. Passing the wrong size cuts or drops boxes
  without an error. On a DevKit, a 1280 × 850 result read with 640 × 425
  returned one box cut to (137, 237)–(640, 425) and dropped the second. Fix:
  pass the size of the frame you sent, or `0` for no clamping.
- **`decode_bbox_tensor` reads an untagged tensor as boxes.** Given a raw
  head tensor, which has no tag, it returned an empty list without `strict`;
  with `strict` it threw `bbox header exceeds payload count`. An empty list
  therefore does not prove there were no objects. Fix: check the tag with `format_is_bbox`, or
  pass `strict = true`, as the PCB Defect Detector does. `pyneat.decode_bbox`
  raises `TypeError` on the same input.
- **A wrong `decode_type` fails at the first run, not at load.** On YOLO26n,
  `YoloV8`, `YoloX`, `YoloV26Pose`, `YoloV26Seg`, `YoloV26Obb`, `Ssd`, `Detr`
  and others loaded without error and then failed on the first `run` with a
  message about the head layout. `YoloV6`, whose heads have the same layout,
  gave the same boxes as `YoloV26`. A `num_classes` that does not match the
  heads also fails there: `num_classes mismatch: configured=1
  inferred_from_mpk=80`. Fix: run one frame right after loading, as the
  examples do with their seed image.
- **The decoder's resize policy cannot relabel the route's resize.** On a
  letterboxing image route, `boxdecode_resize_mode = Stretch` failed on the
  first run with `resize_mode_override='stretch' conflicts with the active
  preprocess resize mode 'letterbox'`. Fix: set the mode in `preprocess.resize`
  instead (see [preprocessing.md](preprocessing.md)).

## Used in

- `YoloV26`:
  [YOLO26 Object Detector](../../../examples/object-detection/yolo26-object-detector/),
  [PCB Defect Detector](../../../examples/object-detection/pcb-defect-detector/),
  [Single-Stream Object Detector](../../../examples/object-detection/single-stream-object-detector/),
  [Multi-Stream Object Detector](../../../examples/object-detection/multi-stream-object-detector/),
  [High-Density Multi-Stream Object Detector](../../../examples/object-detection/high-density-multi-stream-object-detector/),
  [Multi-Stream Tracker](../../../examples/tracking/multi-stream-tracker/),
  [YOLO26 Tiny Drone Tracker](../../../examples/tracking/yolo26-tiny-drone-tracker/),
  [Detection to VLM Assistant](../../../examples/genai/detection-to-vlm-assistant/),
  [Multi-Stream BlazePose 3D](../../../examples/pose-estimation/multi-stream-blazepose3d/)
  (person detector)
- `YoloV26`, `YoloV26Pose`, `YoloV26Seg` and `Ssd`, read with `decode_bbox`,
  `decode_pose` and `decode_segmentation`:
  [MIPI Multi-Model](../../../examples/benchmarking/mipi-multi-model/) (Python)
- `YoloV26Pose` and `YoloV8Pose` with `decode_pose`:
  [Multi-Stream Pose Estimator](../../../examples/pose-estimation/multi-stream-pose-estimator/),
  [Multi-Stream Multi-Model](../../../examples/multi-model/multi-stream-multi-model/)
- `YoloV26Seg` with `decode_segmentation`:
  [Single-Stream Instance Segmenter](../../../examples/segmentation/single-stream-instance-segmenter/),
  [Multi-Stream Multi-Model](../../../examples/multi-model/multi-stream-multi-model/)
- `YoloV26Obb` and `YoloV8Obb` with `decode_obb`:
  [Multi-Stream OBB Detector](../../../examples/object-detection/multi-stream-obb-detector/)
- `Ssd`: [SSD-MobileNet Object Detector](../../../examples/object-detection/ssd-mobilenet-object-detector/)
- `Detr`: [DETR Object Detector](../../../examples/object-detection/detr-object-detector/)
- `SuperPoint` with `decode_superpoint`:
  [SuperPoint Feature Extractor](../../../examples/feature-extraction/superpoint-feature-extractor/)
- Raw heads decoded in the application:
  [Face Recognizer](../../../examples/face-recognition/face-recognizer/) (SCRFD)

## Source

- C++ headers: `include/pipeline/BoxDecodeType.h`, `BoxDecodeType` and
  `BoxDecodeTypeOption`; `include/model/Model.h`, `Model::Options`;
  `include/pipeline/DetectionTypes.h`, `Box`, `BoxDecodeResult` and the
  `decode_*` helpers; `include/pipeline/FeatureTypes.h`, `decode_superpoint`;
  `include/pipeline/SuperPointTypes.h`, `SuperPointOptions`;
  `include/pipeline/YoloXSegPoseTypes.h`, `YoloXSegPoseOptions`
- Python binding: `python/src/module.cpp`, `BoxDecodeType`,
  `BoxDecodeTypeOption`, `SuperPointOptions`, `YoloXSegPoseOptions`,
  `ModelOptions`, `decode_bbox`, `decode_pose`, `decode_segmentation`,
  `decode_segmentation_pose`, `decode_obb`, `decode_obb_tensor`,
  `decode_superpoint` and the `detections` submodule
- Core commit: `6ed16525`
