"""Faster R-CNN object detection using two compiled NEAT model packages.

This example mirrors quantized_pipeline_merged.py from the Faster R-CNN demo:
  - backbone_rpn_head_640_640_mpk.tar.gz runs backbone + concat RPN
  - NumPy decodes RPN proposals and performs ROI Align
  - box_head_predictor_640_640_mpk.tar.gz runs box head + predictor
  - NumPy decodes final detections and writes annotated images
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VERBOSE = False
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_BACKBONE_RPN_MODEL = "assets/models/backbone_rpn_head_640_640_mpk.tar.gz"
DEFAULT_HEAD_PREDICTOR_MODEL = "assets/models/box_head_predictor_640_640_mpk.tar.gz"
PATCH_VERSION = "faster-rcnn-neat-2026-06-23-owned-output"

INFER_WIDTH = 640
INFER_HEIGHT = 640
INPUT_SHAPE = (INFER_HEIGHT, INFER_WIDTH)
PIXEL_MEAN = None
PIXEL_STD = None

ANCHOR_SIZES = [[32], [64], [128], [256], [512]]
ANCHOR_ASPECT_RATIOS = [0.5, 1.0, 2.0]
FPN_STRIDES = [4, 8, 16, 32, 64]
RPN_BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
RPN_NMS_THRESH = 0.7
RPN_PRE_NMS_TOPK = 1000
RPN_POST_NMS_TOPK = 1000
ROI_NUM_CLASSES = 80
ROI_BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
ROI_NMS_THRESH = 0.5
MAX_DETECTIONS = 100
FPN_LEVEL_HW = [(160, 160), (80, 80), (40, 40), (20, 20), (10, 10)]

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[faster-rcnn-debug] {msg}", flush=True)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class RunnerSpec:
    model_path: Path
    height: int
    width: int
    depth: int


@dataclass
class FasterRcnnRunners:
    backbone_rpn: Any
    head_predictor: Any

    def close(self) -> None:
        for runner in (self.backbone_rpn, self.head_predictor):
            try:
                runner.close()
            except Exception:
                logger.debug("runner.close() failed", exc_info=True)


def tensor_to_numpy(tensor: pyneat.Tensor) -> np.ndarray:
    # Prefer a zero-copy view for strided NEAT outputs. copy=True and
    # copy_dense_bytes_tight() can both route through Tensor::clone on some
    # runtime tensors, which fails for these model outputs. Once NumPy has a
    # strided view, np.array(..., copy=True) performs the dense copy safely.
    try:
        return np.array(tensor.to_numpy(copy=False), copy=True)
    except Exception as view_exc:
        dtype_map = {
            pyneat.TensorDType.UInt8: np.uint8,
            pyneat.TensorDType.Int8: np.int8,
            pyneat.TensorDType.UInt16: np.uint16,
            pyneat.TensorDType.Int16: np.int16,
            pyneat.TensorDType.Int32: np.int32,
            pyneat.TensorDType.Float32: np.float32,
            pyneat.TensorDType.Float64: np.float64,
        }
        np_dtype = dtype_map.get(tensor.dtype)
        if np_dtype is None:
            raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}") from view_exc

        shape = tuple(int(dim) for dim in tensor.shape)
        try:
            arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
        except Exception as dense_exc:
            raise RuntimeError(
                f"Failed to convert NEAT tensor to numpy; "
                f"shape={shape}, dtype={tensor.dtype}, "
                f"view_error={view_exc}, dense_error={dense_exc}"
            ) from dense_exc
        if shape:
            arr = arr.reshape(shape)
        return arr


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    for field in sample.fields:
        yield from iter_tensors(field)


def collect_tensors(sample: pyneat.Sample) -> list[pyneat.Tensor]:
    return list(iter_tensors(sample))


def tensor_from_hwc_f32(array: np.ndarray) -> pyneat.Tensor:
    if array.ndim != 3:
        raise ValueError(f"expected HWC tensor, got shape {array.shape}")
    arr = np.array(array, dtype=np.float32, order="C", copy=True)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return pyneat.Tensor.from_numpy(
        arr,
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )


def tensor_numpy_outputs(tensors: list[pyneat.Tensor], *, stage: str) -> list[np.ndarray]:
    outputs = []
    for idx, tensor in enumerate(tensors):
        try:
            outputs.append(tensor_to_numpy(tensor))
        except BaseException as exc:
            shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
            dtype = getattr(tensor, "dtype", "unknown")
            raise RuntimeError(
                f"{stage}: failed to convert output tensor {idx}: shape={shape}, dtype={dtype}: {exc}"
            ) from exc
    return outputs


def build_tensor_runner(spec: RunnerSpec) -> Any:
    _log(f"Loading model package {spec.model_path}")
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = spec.width
    opt.preprocess.input_max_height = spec.height
    opt.preprocess.input_max_depth = spec.depth

    model = pyneat.Model(str(spec.model_path), opt)
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    graph.add(pyneat.groups.mla(model))
    graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    graph.add(pyneat.nodes.output())

    run_options = pyneat.RunOptions()
    run_options.output_memory = pyneat.OutputMemory.Owned

    dummy = tensor_from_hwc_f32(np.zeros((spec.height, spec.width, spec.depth), dtype=np.float32))
    return graph.build([dummy], run_options)


def build_faster_rcnn_runners(backbone_rpn_path: Path, head_predictor_path: Path) -> FasterRcnnRunners:
    return FasterRcnnRunners(
        backbone_rpn=build_tensor_runner(
            RunnerSpec(backbone_rpn_path, INFER_HEIGHT, INFER_WIDTH, 3)
        ),
        head_predictor=build_tensor_runner(
            RunnerSpec(head_predictor_path, RPN_POST_NMS_TOPK, 1, 12544)
        ),
    )


def run_neat_runner(runner: Any, hwc: np.ndarray, timeout_ms: int, *, stage: str) -> list[np.ndarray]:
    try:
        tensor = tensor_from_hwc_f32(hwc)
    except BaseException as exc:
        raise RuntimeError(
            f"{stage}: failed to create input tensor from shape={getattr(hwc, 'shape', None)}, "
            f"strides={getattr(hwc, 'strides', None)}, c_contiguous={getattr(getattr(hwc, 'flags', None), 'c_contiguous', None)}: {exc}"
        ) from exc
    try:
        pushed = runner.push([tensor])
    except BaseException as exc:
        raise RuntimeError(f"{stage}: Run.push() raised: {exc}") from exc
    if not pushed:
        raise RuntimeError(f"{stage}: Run.push() returned False")
    try:
        sample = runner.pull(timeout_ms=timeout_ms)
    except BaseException as exc:
        raise RuntimeError(f"{stage}: Run.pull() raised: {exc}") from exc
    if sample is None:
        raise RuntimeError(f"{stage}: Run.pull() returned no sample")
    try:
        tensors = collect_tensors(sample)
    except BaseException as exc:
        raise RuntimeError(f"{stage}: failed to collect output tensors: {exc}") from exc
    return tensor_numpy_outputs(tensors, stage=stage)


def ensure_nhwc4(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"{name}: expected rank-4 NHWC tensor, got {arr.shape}")
    return arr


def parse_backbone_rpn_outputs(outputs: list[np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if len(outputs) != 6:
        raise ValueError(f"Expected 6 backbone+RPN outputs, got {len(outputs)}")
    p2, p3, p4, p5, logits_concat, deltas_concat = [
        ensure_nhwc4(arr, name)
        for arr, name in zip(outputs, ["p2", "p3", "p4", "p5", "logits_concat", "deltas_concat"])
    ]
    features = {
        "p2": p2.transpose(0, 3, 1, 2),
        "p3": p3.transpose(0, 3, 1, 2),
        "p4": p4.transpose(0, 3, 1, 2),
        "p5": p5.transpose(0, 3, 1, 2),
    }
    return features, logits_concat.transpose(0, 3, 1, 2), deltas_concat.transpose(0, 3, 1, 2)


def parse_head_predictor_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(outputs) != 2:
        raise ValueError(f"Expected 2 head+predictor outputs, got {len(outputs)}")
    scores, deltas = [ensure_nhwc4(arr, name) for arr, name in zip(outputs, ["scores", "deltas"])]
    return scores[0, :, 0, :], deltas[0, :, 0, :]


def _cell_anchors(size: int, aspect_ratios: list[float]) -> np.ndarray:
    area = float(size * size)
    anchors = []
    for ratio in aspect_ratios:
        w = math.sqrt(area / ratio)
        h = ratio * w
        anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    return np.array(anchors, dtype=np.float32)


def _generate_level_anchors(img_h: int, img_w: int, stride: int, size: int, aspect_ratios: list[float]) -> np.ndarray:
    cell = _cell_anchors(size, aspect_ratios)
    feat_h = img_h // stride
    feat_w = img_w // stride
    grid_x, grid_y = np.meshgrid(
        np.arange(feat_w, dtype=np.float32) * stride,
        np.arange(feat_h, dtype=np.float32) * stride,
    )
    shifts = np.stack([grid_x, grid_y, grid_x, grid_y], axis=-1).reshape(-1, 4)
    return (shifts[:, None, :] + cell[None, :, :]).reshape(-1, 4)


def _all_anchors() -> list[np.ndarray]:
    return [
        _generate_level_anchors(INFER_HEIGHT, INFER_WIDTH, stride, sizes[0], ANCHOR_ASPECT_RATIOS)
        for stride, sizes in zip(FPN_STRIDES, ANCHOR_SIZES)
    ]


def _apply_deltas(boxes: np.ndarray, deltas: np.ndarray, weights: tuple[float, float, float, float]) -> np.ndarray:
    wx, wy, ww, wh = weights
    bw = boxes[..., 2] - boxes[..., 0]
    bh = boxes[..., 3] - boxes[..., 1]
    bcx = boxes[..., 0] + 0.5 * bw
    bcy = boxes[..., 1] + 0.5 * bh
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = np.clip(deltas[..., 2] / ww, None, math.log(1000.0 / 16))
    dh = np.clip(deltas[..., 3] / wh, None, math.log(1000.0 / 16))
    pred_cx = dx * bw + bcx
    pred_cy = dy * bh + bcy
    pred_w = np.exp(dw) * bw
    pred_h = np.exp(dh) * bh
    return np.stack(
        [
            pred_cx - 0.5 * pred_w,
            pred_cy - 0.5 * pred_h,
            pred_cx + 0.5 * pred_w,
            pred_cy + 0.5 * pred_h,
        ],
        axis=-1,
    ).astype(np.float32)


def _clip_boxes(boxes: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    out = boxes.copy()
    out[..., 0] = np.clip(out[..., 0], 0, img_w)
    out[..., 1] = np.clip(out[..., 1], 0, img_h)
    out[..., 2] = np.clip(out[..., 2], 0, img_w)
    out[..., 3] = np.clip(out[..., 3], 0, img_h)
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        inter = (
            np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]))
            * np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        )
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[1:][iou <= thresh]
    return np.array(keep, dtype=np.int64)


def decode_rpn(logits_concat: np.ndarray, deltas_concat: np.ndarray) -> np.ndarray:
    anchors_per_level = _all_anchors()
    offsets = []
    off = 0
    for h, w in FPN_LEVEL_HW:
        offsets.append(off)
        off += h * w
    num_anchors = len(ANCHOR_ASPECT_RATIOS)

    all_proposals, all_scores = [], []
    for lvl, (h, w) in enumerate(FPN_LEVEL_HW):
        off = offsets[lvl]
        logits = logits_concat[:, :, off : off + h * w, :].reshape(1, 3, h, w)
        deltas = deltas_concat[:, :, off : off + h * w, :].reshape(1, 12, h, w)
        anchors = anchors_per_level[lvl]

        logits_flat = logits[0].transpose(1, 2, 0).reshape(-1)
        _, _, hi, wi = deltas.shape
        deltas_flat = deltas[0].reshape(num_anchors, 4, hi, wi).transpose(2, 3, 0, 1).reshape(-1, 4)

        topk = min(RPN_PRE_NMS_TOPK, len(logits_flat))
        idx = np.argpartition(logits_flat, -topk)[-topk:]
        idx = idx[np.argsort(logits_flat[idx])[::-1]]

        all_proposals.append(_apply_deltas(anchors[idx], deltas_flat[idx], RPN_BBOX_REG_WEIGHTS))
        all_scores.append(logits_flat[idx])

    proposals = _clip_boxes(np.concatenate(all_proposals), INFER_HEIGHT, INFER_WIDTH)
    scores = np.concatenate(all_scores)
    keep = _nms(proposals, scores, RPN_NMS_THRESH)[:RPN_POST_NMS_TOPK]
    proposals = proposals[keep]

    if len(proposals) < RPN_POST_NMS_TOPK:
        proposals = np.concatenate(
            [proposals, np.zeros((RPN_POST_NMS_TOPK - len(proposals), 4), dtype=np.float32)]
        )
    return proposals.astype(np.float32, copy=False)


def _roi_align_numpy(features: dict[str, np.ndarray], proposals: np.ndarray, output_size: int = 7) -> np.ndarray:
    level_keys = ["p2", "p3", "p4", "p5"]
    strides = [4, 8, 16, 32]
    canonical_level = 4
    canonical_size = 224
    min_level, max_level = 2, 5

    n = proposals.shape[0]
    channels = features["p2"].shape[1]
    output = np.zeros((n, channels, output_size, output_size), dtype=np.float32)

    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    areas = np.maximum(widths * heights, 1e-6)
    levels = np.floor(canonical_level + np.log2(np.sqrt(areas) / canonical_size + 1e-8))
    levels = np.clip(levels, min_level, max_level).astype(np.int32)

    for lvl_idx, (key, stride) in enumerate(zip(level_keys, strides)):
        lvl = lvl_idx + 2
        mask = levels == lvl
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        feat = features[key][0]
        _, feat_h, feat_w = feat.shape
        spatial_scale = 1.0 / stride

        for idx in indices:
            roi_x1, roi_y1, roi_x2, roi_y2 = proposals[idx] * spatial_scale
            roi_x1 -= 0.5
            roi_y1 -= 0.5
            roi_x2 -= 0.5
            roi_y2 -= 0.5
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            bin_w = roi_w / output_size
            bin_h = roi_h / output_size
            grid_h = max(1, int(np.ceil(bin_h)))
            grid_w = max(1, int(np.ceil(bin_w)))

            ph, pw = np.arange(output_size), np.arange(output_size)
            iy, ix = np.arange(grid_h), np.arange(grid_w)
            ph_g, pw_g, iy_g, ix_g = np.meshgrid(ph, pw, iy, ix, indexing="ij")
            sy = roi_y1 + bin_h * (ph_g + (iy_g + 0.5) / grid_h)
            sx = roi_x1 + bin_w * (pw_g + (ix_g + 0.5) / grid_w)
            sy = np.clip(sy, 0, feat_h - 1)
            sx = np.clip(sx, 0, feat_w - 1)
            y0 = np.floor(sy).astype(np.int32)
            x0 = np.floor(sx).astype(np.int32)
            y1 = np.minimum(y0 + 1, feat_h - 1)
            x1 = np.minimum(x0 + 1, feat_w - 1)
            ly = sy - y0
            lx = sx - x0
            hy = 1.0 - ly
            hx = 1.0 - lx
            val = (
                hy[None] * hx[None] * feat[:, y0, x0]
                + hy[None] * lx[None] * feat[:, y0, x1]
                + ly[None] * hx[None] * feat[:, y1, x0]
                + ly[None] * lx[None] * feat[:, y1, x1]
            )
            output[idx] = val.mean(axis=(-2, -1))
    return output


def decode_final_detections(
    score_logits: np.ndarray,
    box_deltas: np.ndarray,
    proposals: np.ndarray,
    *,
    confidence_threshold: float,
) -> dict[str, np.ndarray]:
    exp_s = np.exp(score_logits - score_logits.max(axis=1, keepdims=True))
    scores_all = (exp_s / exp_s.sum(axis=1, keepdims=True))[:, :ROI_NUM_CLASSES]
    boxes_all = _apply_deltas(
        proposals[:, None, :],
        box_deltas.reshape(RPN_POST_NMS_TOPK, ROI_NUM_CLASSES, 4),
        ROI_BBOX_REG_WEIGHTS,
    )
    boxes_all = _clip_boxes(boxes_all, INFER_HEIGHT, INFER_WIDTH)

    scores_flat = scores_all.reshape(-1)
    boxes_flat = boxes_all.reshape(-1, 4)
    classes_flat = np.tile(np.arange(ROI_NUM_CLASSES), RPN_POST_NMS_TOPK)
    mask = scores_flat > confidence_threshold
    if not mask.any():
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros(0, dtype=np.float32),
            "classes": np.zeros(0, dtype=np.int32),
        }

    boxes_f = boxes_flat[mask]
    scores_f = scores_flat[mask]
    classes_f = classes_flat[mask]
    offsets = classes_f.astype(np.float32) * (boxes_f.max() + 1)
    keep = _nms(boxes_f + offsets[:, None], scores_f, ROI_NMS_THRESH)[:MAX_DETECTIONS]
    return {
        "boxes": boxes_f[keep],
        "scores": scores_f[keep],
        "classes": classes_f[keep].astype(np.int32),
    }


def preprocess_image(original_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(original_bgr, (INFER_WIDTH, INFER_HEIGHT)).astype(np.float32).transpose(2, 0, 1)
    image = (image - PIXEL_MEAN[:, None, None]) / PIXEL_STD[:, None, None]
    return np.expand_dims(image, axis=0)


def post_process(predictions: dict[str, np.ndarray], original_bgr: np.ndarray) -> dict[str, np.ndarray]:
    orig_h, orig_w = original_bgr.shape[:2]
    scale_w = INFER_WIDTH / orig_w
    scale_h = INFER_HEIGHT / orig_h
    boxes = predictions["boxes"].copy()
    boxes[:, 0] /= scale_w
    boxes[:, 1] /= scale_h
    boxes[:, 2] /= scale_w
    boxes[:, 3] /= scale_h
    return {**predictions, "boxes": boxes}


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_id) % len(BOX_COLORS)]


def visualize_detections(image_bgr: np.ndarray, predictions: dict[str, np.ndarray], *, max_draw: int) -> np.ndarray:
    out = image_bgr.copy()
    limit = len(predictions["boxes"]) if max_draw <= 0 else min(len(predictions["boxes"]), max_draw)
    for box, score, cls in zip(
        predictions["boxes"][:limit], predictions["scores"][:limit], predictions["classes"][:limit]
    ):
        x1, y1, x2, y2 = box.astype(np.int32)
        color = class_color(int(cls))
        label = f"{class_name(int(cls))} {float(score):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def read_image(image_path: Path) -> np.ndarray:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def run_faster_rcnn(
    runners: FasterRcnnRunners,
    image_path: Path,
    *,
    timeout_ms: int,
    confidence_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    original_bgr = read_image(image_path)
    processed_nchw = preprocess_image(original_bgr)
    backbone_input = processed_nchw.transpose(0, 2, 3, 1)[0]

    backbone_outputs = run_neat_runner(runners.backbone_rpn, backbone_input, timeout_ms, stage="backbone_rpn")
    features, logits_concat, deltas_concat = parse_backbone_rpn_outputs(backbone_outputs)
    proposals = decode_rpn(logits_concat, deltas_concat)

    box_features = _roi_align_numpy(features, proposals)
    head_input = np.ascontiguousarray(box_features.reshape(RPN_POST_NMS_TOPK, 12544)[:, None, :], dtype=np.float32)
    head_outputs = run_neat_runner(runners.head_predictor, head_input, timeout_ms, stage="head_predictor")
    score_logits, box_deltas = parse_head_predictor_outputs(head_outputs)
    predictions = decode_final_detections(
        score_logits,
        box_deltas,
        proposals,
        confidence_threshold=confidence_threshold,
    )
    return post_process(predictions, original_bgr), original_bgr


def run_faster_rcnn_profiled(
    runners: FasterRcnnRunners,
    image_path: Path,
    *,
    timeout_ms: int,
    confidence_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    original_bgr = read_image(image_path)
    t1 = time.perf_counter()

    processed_nchw = preprocess_image(original_bgr)
    backbone_input = processed_nchw.transpose(0, 2, 3, 1)[0]
    t2 = time.perf_counter()

    backbone_outputs = run_neat_runner(runners.backbone_rpn, backbone_input, timeout_ms, stage="backbone_rpn")
    t3 = time.perf_counter()

    features, logits_concat, deltas_concat = parse_backbone_rpn_outputs(backbone_outputs)
    proposals = decode_rpn(logits_concat, deltas_concat)
    t4 = time.perf_counter()

    box_features = _roi_align_numpy(features, proposals)
    head_input = np.ascontiguousarray(box_features.reshape(RPN_POST_NMS_TOPK, 12544)[:, None, :], dtype=np.float32)
    t5 = time.perf_counter()

    head_outputs = run_neat_runner(runners.head_predictor, head_input, timeout_ms, stage="head_predictor")
    t6 = time.perf_counter()

    score_logits, box_deltas = parse_head_predictor_outputs(head_outputs)
    predictions = decode_final_detections(
        score_logits,
        box_deltas,
        proposals,
        confidence_threshold=confidence_threshold,
    )
    predictions = post_process(predictions, original_bgr)
    t7 = time.perf_counter()

    timings["read_image"] = t1 - t0
    timings["preprocess"] = t2 - t1
    timings["backbone_rpn"] = t3 - t2
    timings["rpn_decode"] = t4 - t3
    timings["roi_align"] = t5 - t4
    timings["head_predictor"] = t6 - t5
    timings["final_decode"] = t7 - t6
    timings["neat_total"] = timings["backbone_rpn"] + timings["head_predictor"]
    timings["cpu_glue_total"] = (
        timings["read_image"]
        + timings["preprocess"]
        + timings["rpn_decode"]
        + timings["roi_align"]
        + timings["final_decode"]
    )
    timings["pipeline_run"] = t7 - t0
    return predictions, original_bgr, timings


def format_profile_stats(name: str, values: list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    runs = float(len(arr))
    fps = runs / arr.sum()
    return (
        f"  {name}: mean={arr.mean():.8f}s, min={arr.min():.8f}s, "
        f"max={arr.max():.8f}s, FPS={fps:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Faster R-CNN folder object detection example")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    args = parser.parse_args()

    global cv2, np, pyneat, PIXEL_MEAN, PIXEL_STD, VERBOSE
    import cv2
    import numpy as np
    import pyneat

    PIXEL_MEAN = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    PIXEL_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    models_cfg = raw.get("models", {})
    io_cfg = raw.get("io", {})
    decode_cfg = raw.get("decode", {})
    runtime_cfg = raw.get("runtime", {})

    backbone_rpn_path = Path(models_cfg.get("backbone_rpn", {}).get("path", DEFAULT_BACKBONE_RPN_MODEL))
    head_predictor_path = Path(models_cfg.get("head_predictor", {}).get("path", DEFAULT_HEAD_PREDICTOR_MODEL))
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/faster-rcnn-object-detector"))
    confidence_threshold = float(decode_cfg.get("confidence_threshold", 0.5))
    max_draw = int(decode_cfg.get("max_draw", 50))
    profile = bool(runtime_cfg.get("profile", False))
    num_runs = int(runtime_cfg.get("num_runs", 100))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 20000))
    VERBOSE = bool(runtime_cfg.get("verbose", False))
    if VERBOSE:
        print(f"Patch version: {PATCH_VERSION}", flush=True)

    for model_path in (backbone_rpn_path, head_predictor_path):
        if not model_path.is_file():
            print(f"Model file does not exist: {model_path}", file=sys.stderr)
            return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and is_image(path))
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3

    try:
        runners = build_faster_rcnn_runners(backbone_rpn_path, head_predictor_path)
    except Exception as exc:
        logger.debug("Graph build failure", exc_info=exc)
        print(f"Error building NEAT runners: {exc}", file=sys.stderr)
        return 3

    try:
        if profile:
            image_path = image_paths[0]
            profile_order = [
                "read_image",
                "preprocess",
                "backbone_rpn",
                "rpn_decode",
                "roi_align",
                "head_predictor",
                "final_decode",
                "neat_total",
                "cpu_glue_total",
                "pipeline_run",
            ]
            profile_times: dict[str, list[float]] = {key: [] for key in profile_order}
            visualization_times: list[float] = []
            last_predictions: dict[str, np.ndarray] | None = None
            for _ in range(max(1, num_runs)):
                predictions, original_bgr, timings = run_faster_rcnn_profiled(
                    runners,
                    image_path,
                    timeout_ms=timeout_ms,
                    confidence_threshold=confidence_threshold,
                )
                t0 = time.perf_counter()
                _ = visualize_detections(original_bgr, predictions, max_draw=max_draw)
                t1 = time.perf_counter()
                for key in profile_order:
                    profile_times[key].append(timings[key])
                visualization_times.append(t1 - t0)
                last_predictions = predictions

            total_times = [
                pipeline + vis
                for pipeline, vis in zip(profile_times["pipeline_run"], visualization_times)
            ]
            print(f"Profiling over {len(profile_times['pipeline_run'])} runs (image='{image_path}'):")
            for key, label in [
                ("read_image", "Read image"),
                ("preprocess", "Preprocess"),
                ("backbone_rpn", "Backbone+RPN NEAT"),
                ("rpn_decode", "RPN decode+NMS"),
                ("roi_align", "ROI Align"),
                ("head_predictor", "Box head NEAT"),
                ("final_decode", "Final decode+NMS"),
                ("neat_total", "NEAT total"),
                ("cpu_glue_total", "CPU glue total"),
                ("pipeline_run", "Pipeline run"),
            ]:
                print(format_profile_stats(label, profile_times[key]))
            print(format_profile_stats("Visualization", visualization_times))
            print(format_profile_stats("Overall", total_times))
            print(f"Last run detections: {0 if last_predictions is None else len(last_predictions['boxes'])}")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        for image_path in image_paths:
            try:
                predictions, original_bgr = run_faster_rcnn(
                    runners,
                    image_path,
                    timeout_ms=timeout_ms,
                    confidence_threshold=confidence_threshold,
                )
            except Exception as exc:
                logger.debug("Inference failure", exc_info=exc)
                print(f"Error during inference for {image_path}: {exc}", file=sys.stderr)
                return 3

            output_path = output_dir / f"{image_path.stem}_faster_rcnn.png"
            out_img = visualize_detections(original_bgr, predictions, max_draw=max_draw)
            cv2.imwrite(str(output_path), out_img)
            processed += 1
            print(
                f"[{processed}/{len(image_paths)}] {image_path.name}: "
                f"{len(predictions['boxes'])} detections -> {output_path.name}"
            )

        print(f"Done: {processed} images processed")
        return 0
    finally:
        runners.close()


if __name__ == "__main__":
    raise SystemExit(main())
