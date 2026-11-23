"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass, field

from fractions import Fraction

import glob

import json

import os

from pathlib import Path

import subprocess

import sys

import time

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_single_stream_instance_segmenter')) / "common" / "config.yaml"

DEFAULT_LABELS = DEFAULT_CONFIG.parent / "coco_label.txt"

MASK_ALPHA = 0.55

MASK_THRESHOLD = 0.50

DEFAULT_INPUT_SIZE = 640

YOLO26 = "yolo26"

YOLOV8 = "yolov8"

METADATA_BYTE_BUDGET = 32768

MASK_STRIDE = 4

YOLO26_MASK_GRID = 160

YOLOV8_HEAD_TENSORS = 10

YOLOV8_STRIDES = (8, 16, 32)

DFL_BINS = 16

MASK_COEFFICIENTS = 32

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class OutputConfig:
    save_dir: str
    save_every: int
    mask_alpha: float
    mask_threshold: float
    draw_boxes: bool

@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    source_url: str
    model_family: str = YOLO26
    input_size: int = DEFAULT_INPUT_SIZE
    source_type: str = "rtsp"
    source_codec: str = "h264"
    latency_ms: int = 200
    tcp: bool = True
    source_fps: int = 0
    ssl_strict: bool = True
    frames: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    output: OutputConfig = OutputConfig("", 0, MASK_ALPHA, MASK_THRESHOLD, True)
    # The key that names the source: source.<type>.url, or source.usb.camera.
    source_key: str = "source.rtsp.url"
    # source.type=usb: the camera's stable ID (usb_camera_ids()) and one of its MJPEG modes.
    camera: str = ""
    source_width: int = 1920
    source_height: int = 1080

def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module

def section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value

def string_or(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value

def int_or(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)

def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)

def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return bool(value)

def parse_model_family(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"yolo26", "yolo-26", "yolov26"}:
        return YOLO26
    if lowered in {"yolov8", "yolo-v8", "yolo_v8"}:
        return YOLOV8
    raise ValueError("model.family must be yolo26 or yolov8")

def parse_source_type(value: str) -> str:
    lowered = value.lower()
    if lowered in {"rtsp", "http", "https", "usb"}:
        return "http" if lowered == "https" else lowered
    raise ValueError("source.type must be rtsp, http, or usb")

def parse_source_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    if lowered in {"mjpeg", "jpeg"}:
        return "mjpeg"
    raise ValueError("source.rtsp.codec must be h264, h265, or mjpeg")

def validate_config(cfg: AppConfig) -> None:
    if cfg.source_type == "usb":
        if not cfg.camera:
            raise ValueError("source.usb.camera must be set")
        if cfg.source_width <= 0 or cfg.source_height <= 0:
            raise ValueError("source.usb.width and source.usb.height must be > 0")
        if cfg.source_fps <= 0:
            raise ValueError("source.usb.fps must be > 0")
        # The camera's JPEG goes to Insight as RTP/JPEG (RFC 2435): 8-pixel blocks, 2040 at most.
        sides = (cfg.source_width, cfg.source_height)
        if not all(0 < side <= 2040 and side % 8 == 0 for side in sides):
            raise ValueError(
                "source.usb.width and source.usb.height must be multiples of 8 and at most 2040 "
                "(RTP/JPEG)"
            )
    elif not cfg.source_url:
        raise ValueError(f"{cfg.source_key} must be set")
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if cfg.input_size <= 0 or cfg.input_size % (MASK_STRIDE * max(YOLOV8_STRIDES)) != 0:
        raise ValueError(
            "model.input_size must be a positive multiple of "
            f"{MASK_STRIDE * max(YOLOV8_STRIDES)}"
        )
    # Path("") is ".", so an empty value in the file arrives here as ".".
    if str(cfg.labels_path) in ("", "."):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError(f"source.{cfg.source_type}.latency_ms must be >= 0")
    if cfg.source_fps < 0:
        raise ValueError(f"source.{cfg.source_type}.fps must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.output.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not 0.0 <= cfg.output.mask_alpha <= 1.0:
        raise ValueError("output.mask_alpha must be between 0 and 1")
    if not 0.0 <= cfg.output.mask_threshold <= 1.0:
        raise ValueError("output.mask_threshold must be between 0 and 1")

def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    source = section(raw, "source")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")

    labels_path = string_or(model, "labels", str(DEFAULT_LABELS))
    # Only the section source.type names is read; the others may hold placeholders.
    # The key that names the source is kept so --validate-config-only can report
    # it without echoing the URL, which can carry credentials.
    source_type = parse_source_type(string_or(source, "type", "rtsp"))
    source_key = f"source.{source_type}.{'camera' if source_type == 'usb' else 'url'}"
    source = section(source, source_type)
    # HTTP streams and USB cameras are MJPEG; only RTSP names a codec.
    source_codec = (
        parse_source_codec(string_or(source, "codec", "h264")) if source_type == "rtsp" else "mjpeg"
    )

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(labels_path),
        source_url=string_or(source, "url"),
        source_key=source_key,
        model_family=parse_model_family(string_or(model, "family", YOLO26)),
        input_size=int_or(model, "input_size", DEFAULT_INPUT_SIZE),
        source_type=source_type,
        camera=string_or(source, "camera"),
        source_width=int_or(source, "width", 1920),
        source_height=int_or(source, "height", 1080),
        source_codec=source_codec,
        latency_ms=int_or(source, "latency_ms", 200),
        tcp=bool_or(source, "tcp", True),
        source_fps=int_or(source, "fps", 0),
        ssl_strict=bool_or(source, "ssl_strict", True),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        output=OutputConfig(
            save_dir=string_or(output, "save_dir"),
            save_every=int_or(output, "save_every", 0),
            mask_alpha=float_or(output, "mask_alpha", MASK_ALPHA),
            mask_threshold=float_or(output, "mask_threshold", MASK_THRESHOLD),
            draw_boxes=bool_or(output, "draw_boxes", True),
        ),
    )
    validate_config(cfg)
    return cfg

def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise RuntimeError(f"labels file does not exist: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {labels_path}")
    return labels

def tensor_to_numpy(tensor) -> object:
    return np.asarray(tensor.to_numpy(copy=True))

def numpy_dtype(tensor) -> object:
    dtypes = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    dtype = dtypes.get(tensor.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported tensor dtype: {tensor.dtype}")
    return dtype

def tensor_to_hwc_f32(tensor) -> object:
    """One head tensor as a dense HWC float32 array, dropping a leading batch axis of 1."""
    shape = tuple(int(dim) for dim in tensor.shape)
    array = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=numpy_dtype(tensor))
    array = array.reshape(shape).astype(np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise RuntimeError(f"only batch size 1 is supported, got {array.shape[0]}")
        array = array[0]
    if array.ndim != 3:
        raise RuntimeError(f"unexpected head tensor rank {array.ndim}")
    return array

FRAME_RING_CAPACITY = 64

def detection(x1: float, y1: float, x2: float, y2: float, score: float, class_id: int, mask):
    """One decoded instance, in the single representation both families produce.

    ``x1..y2`` are frame pixels, ``score`` is a probability, and ``mask`` is a uint8 mask grid
    covering the letterboxed model input at ``MASK_STRIDE`` cells per pixel. Everything
    downstream of the decoders - overlays, polygons, Insight metadata - reads only these keys.
    """
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "score": float(score),
        "class_id": int(class_id),
        "mask": mask,
    }

def decode_yolo26_segments(tensors: list, frame_w: int, frame_h: int, max_detections: int):
    """YOLO26 boundary: the MLA already ran BoxDecode, so this only unpacks its payload."""
    decoded = pyneat.decode_segmentation(
        tensors,
        clamp_to=(frame_w, frame_h),
        top_k=max_detections,
        strict=False,
    )
    detections = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).astype(np.float32)
        masks = tensor_to_numpy(item.masks).astype(np.uint8)
        mask_shape = (-1, YOLO26_MASK_GRID, YOLO26_MASK_GRID)
        for row, mask in zip(boxes.reshape((-1, 6)), masks.reshape(mask_shape)):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(detection(x1, y1, x2, y2, score, int(class_id), mask))
            if len(detections) >= max_detections:
                return detections
    return detections

def letterbox_params(frame_w: int, frame_h: int, input_size: int) -> tuple[float, float, float]:
    """Scale and padding the preprocessor applies when it letterboxes a frame."""
    scale = min(input_size / frame_w, input_size / frame_h)
    return scale, (input_size - frame_w * scale) * 0.5, (input_size - frame_h * scale) * 0.5

def split_yolov8_heads(heads: list, input_size: int):
    """Group the packaged YOLOv8 head tensors and check them against ``model.input_size``."""
    if len(heads) < YOLOV8_HEAD_TENSORS:
        raise RuntimeError(
            f"YOLOv8 decode expects {YOLOV8_HEAD_TENSORS} head tensors, got {len(heads)}"
        )
    boxes, scores, coefficients, proto = heads[0:3], heads[3:6], heads[6:9], heads[9]
    if proto.ndim != 3 or proto.shape[2] != MASK_COEFFICIENTS:
        raise RuntimeError(f"unexpected prototype tensor shape {proto.shape}")
    if proto.shape[0] != proto.shape[1] or proto.shape[0] * MASK_STRIDE != input_size:
        raise RuntimeError(
            f"prototype grid {proto.shape[0]}x{proto.shape[1]} does not match "
            f"model.input_size {input_size}"
        )
    for stride, box, score, coefficient in zip(YOLOV8_STRIDES, boxes, scores, coefficients):
        grid = input_size // stride
        if box.shape[:2] != (grid, grid) or box.shape[2] != 4 * DFL_BINS:
            raise RuntimeError(
                f"unexpected box head shape {box.shape} for stride {stride} at "
                f"model.input_size {input_size}"
            )
        if score.shape[:2] != (grid, grid) or score.shape[2] <= 0:
            raise RuntimeError(f"unexpected class head shape {score.shape}")
        if coefficient.shape[:2] != (grid, grid) or coefficient.shape[2] != MASK_COEFFICIENTS:
            raise RuntimeError(f"unexpected mask-coefficient head shape {coefficient.shape}")
    return boxes, scores, coefficients, proto

def yolov8_candidates(boxes, scores, coefficients, input_size: int, min_score: float):
    """Boxes in letterboxed model pixels, class probabilities, and mask coefficients."""
    bins = np.arange(DFL_BINS, dtype=np.float32)
    threshold = np.float32(min_score)
    level_boxes, level_scores, level_classes, level_coefficients = [], [], [], []
    for box, score, coefficient in zip(boxes, scores, coefficients):
        stride = np.float32(input_size / box.shape[0])
        # The packaged class head already carries probabilities, so it is thresholded as is.
        class_ids = score.argmax(axis=2)
        best = np.take_along_axis(score, class_ids[..., None], axis=2)[..., 0]
        rows, columns = np.nonzero(best >= threshold)
        if rows.size == 0:
            continue
        logits = box[rows, columns].reshape((-1, 4, DFL_BINS)).astype(np.float32)
        weights = np.exp(logits - logits.max(axis=2, keepdims=True))
        distance = (weights @ bins) / weights.sum(axis=2) * stride
        center_x = (columns.astype(np.float32) + np.float32(0.5)) * stride
        center_y = (rows.astype(np.float32) + np.float32(0.5)) * stride
        level_box = np.clip(
            np.stack(
                [
                    center_x - distance[:, 0],
                    center_y - distance[:, 1],
                    center_x + distance[:, 2],
                    center_y + distance[:, 3],
                ],
                axis=1,
            ),
            np.float32(0.0),
            np.float32(input_size),
        )
        # Degenerate cells cannot become instances, and dropping them here keeps the
        # candidate list identical to the C++ decoder's.
        valid = (level_box[:, 2] > level_box[:, 0]) & (level_box[:, 3] > level_box[:, 1])
        level_boxes.append(level_box[valid])
        level_scores.append(best[rows, columns].astype(np.float32)[valid])
        level_classes.append(class_ids[rows, columns].astype(np.int32)[valid])
        level_coefficients.append(coefficient[rows, columns].astype(np.float32)[valid])

    if not level_boxes:
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        return (
            empty_boxes,
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, MASK_COEFFICIENTS), dtype=np.float32),
        )
    return (
        np.concatenate(level_boxes),
        np.concatenate(level_scores),
        np.concatenate(level_classes),
        np.concatenate(level_coefficients),
    )

def nms_per_class(boxes, scores, classes, nms_iou: float, max_detections: int):
    """Greedy per-class NMS, highest score first, capped at ``max_detections``."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.intp)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    # Stable descending, so candidates tying on score are kept in the order both
    # implementations visit their cells in.
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0 and len(keep) < max_detections:
        best = order[0]
        keep.append(best)
        rest = order[1:]
        if rest.size == 0:
            break
        overlap_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        intersection = overlap_w * overlap_h
        union = areas[best] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[(classes[rest] != classes[best]) | (iou <= nms_iou)]
    return np.asarray(keep, dtype=np.intp)

def yolov8_instance_mask(proto, coefficients, frame_rect, frame_shape):
    """Prototype mask for one instance, on the same grid BoxDecode returns for YOLO26."""
    mask = np.zeros(proto.shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = mask_rect_for_frame_rect(frame_rect, frame_shape, proto.shape[:2])
    activated = 255.0 / (1.0 + np.exp(-(proto[y0:y1, x0:x1] @ coefficients)))
    # Rounded, not truncated, so the C++ saturate_cast path produces identical mask bytes.
    mask[y0:y1, x0:x1] = np.clip(np.rint(activated), 0.0, 255.0).astype(np.uint8)
    return mask

def decode_yolov8_segments(
    heads: list,
    frame_w: int,
    frame_h: int,
    input_size: int,
    min_score: float,
    nms_iou: float,
    max_detections: int,
):
    """YOLOv8 boundary: decode raw heads here into the shared detection representation."""
    boxes, scores, coefficients, proto = split_yolov8_heads(heads, input_size)
    boxes, scores, classes, coefficients = yolov8_candidates(
        boxes, scores, coefficients, input_size, min_score
    )

    scale, pad_x, pad_y = letterbox_params(frame_w, frame_h, input_size)
    frame_shape = (frame_h, frame_w, 3)
    padding = np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float64)
    limit = np.array([frame_w, frame_h, frame_w, frame_h], dtype=np.float64)
    # Single precision throughout, so the C++ implementation decodes the same coordinates.
    frame_boxes = np.clip((boxes - padding) / scale, 0.0, limit).astype(np.float32)
    # A prediction lying entirely in the letterbox padding has no frame to occupy, so it is
    # dropped before the cap. Discarding it afterwards would let it consume one of the
    # max_detections slots and push out a valid lower-ranked detection.
    inside = (frame_boxes[:, 2] > frame_boxes[:, 0]) & (frame_boxes[:, 3] > frame_boxes[:, 1])
    frame_boxes, scores = frame_boxes[inside], scores[inside]
    classes, coefficients = classes[inside], coefficients[inside]

    keep = nms_per_class(frame_boxes, scores, classes, nms_iou, max_detections)
    detections = []
    for index in keep:
        x1, y1, x2, y2 = (float(value) for value in frame_boxes[index])
        candidate = detection(x1, y1, x2, y2, scores[index], int(classes[index]), None)
        frame_rect = frame_rect_for_detection(candidate, frame_shape)
        candidate["mask"] = yolov8_instance_mask(
            proto, coefficients[index], frame_rect, frame_shape
        )
        detections.append(candidate)
    return detections

def decode_segments(cfg: AppConfig, tensors: list, frame_w: int, frame_h: int):
    """The one place model family changes behavior."""
    if cfg.model_family == YOLO26:
        return decode_yolo26_segments(tensors, frame_w, frame_h, cfg.max_detections)
    return decode_yolov8_segments(
        [tensor_to_hwc_f32(tensor) for tensor in tensors[:YOLOV8_HEAD_TENSORS]],
        frame_w,
        frame_h,
        cfg.input_size,
        cfg.min_score,
        cfg.nms_iou,
        cfg.max_detections,
    )

def make_model(cfg: AppConfig, frame_w: int, frame_h: int, nv16_frames: bool = False):
    """Load the segmentation package. Only the postprocess contract differs per family."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    # The decoder emits NV12 (NV16 for a 4:2:2 USB camera); the model preprocess letterboxes it
    # onto the model input.
    opt.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16 if nv16_frames else pyneat.PreprocessColorFormat.NV12
    )
    if frame_w > 0 and frame_h > 0:
        opt.preprocess.input_max_width = frame_w
        opt.preprocess.input_max_height = frame_h
    if cfg.model_family == YOLO26:
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        # BoxDecode runs on device and emits the segmentation payload this app unpacks.
        opt.decode_type = pyneat.BoxDecodeType.YoloV26Seg
        opt.score_threshold = cfg.min_score
        opt.nms_iou_threshold = cfg.nms_iou
        opt.top_k = cfg.max_detections
    else:
        # YOLOv8 leaves decode_type unset so the route surfaces the raw float heads that
        # decode_yolov8_segments() consumes. The decode inverts a letterbox, so the resize
        # policy is requested rather than assumed, and the YOLO normalization is requested
        # explicitly: without it the model sees unnormalized pixels and its class scores
        # collapse to a fraction of their proper range.
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.preprocess.resize.enable = pyneat.AutoFlag.On
        opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    return pyneat.Model(cfg.model_path, opt)

def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"

def frame_rect_for_detection(det: dict, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x0 = max(0, min(frame_w - 1, int(np.floor(float(det["x1"])))))
    y0 = max(0, min(frame_h - 1, int(np.floor(float(det["y1"])))))
    x1 = max(x0 + 1, min(frame_w, int(np.ceil(float(det["x2"])))))
    y1 = max(y0 + 1, min(frame_h, int(np.ceil(float(det["y2"])))))
    return x0, y0, x1, y1

def mask_rect_for_frame_rect(
    frame_rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    mask_h, mask_w = mask_shape[:2]
    model_w = mask_w * MASK_STRIDE
    model_h = mask_h * MASK_STRIDE
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) * 0.5
    pad_y = (model_h - frame_h * scale) * 0.5

    def to_mask_x(frame_x: float) -> float:
        return (frame_x * scale + pad_x) * mask_w / model_w

    def to_mask_y(frame_y: float) -> float:
        return (frame_y * scale + pad_y) * mask_h / model_h

    fx0, fy0, fx1, fy1 = frame_rect
    x0 = max(0, min(mask_w - 1, int(np.floor(to_mask_x(fx0)))))
    y0 = max(0, min(mask_h - 1, int(np.floor(to_mask_y(fy0)))))
    x1 = max(x0 + 1, min(mask_w, int(np.ceil(to_mask_x(fx1)))))
    y1 = max(y0 + 1, min(mask_h, int(np.ceil(to_mask_y(fy1)))))
    return x0, y0, x1, y1

def project_letterbox_mask_roi(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
):
    x0, y0, x1, y1 = frame_rect
    mx0, my0, mx1, my1 = mask_rect_for_frame_rect(frame_rect, frame_shape, mask.shape)
    return cv2.resize(mask[my0:my1, mx0:mx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

def mask_polygon(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
    threshold: float,
) -> list[list[int]]:
    """Frame-absolute silhouette of ``mask`` inside ``frame_rect``.

    Empty when the thresholded mask holds nothing Insight can draw. ``threshold`` is a fraction of
    full scale, as ``output.mask_threshold`` is. Upscaling before thresholding is what makes the
    outline match the rendered overlay.
    """
    roi = project_letterbox_mask_roi(mask, frame_shape, frame_rect)
    _, binary = cv2.threshold(roi, threshold * 255.0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    # Contour points lie inside frame_rect, which is already clamped to the frame, so shifting them
    # into frame space cannot leave the image.
    x0, y0 = frame_rect[0], frame_rect[1]
    return [[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in polygon]

def metadata_segments(
    detections: list[dict],
    labels: list[str],
    frame_shape: tuple[int, ...],
    mask_threshold: float,
) -> list[dict]:
    segments = []
    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        x0, y0, x1, y1 = frame_rect_for_detection(det, frame_shape)
        polygon = mask_polygon(mask, frame_shape, (x0, y0, x1, y1), mask_threshold)
        if not polygon:
            continue
        segments.append(
            {
                "id": f"seg_{len(segments) + 1}",
                "label": class_name(labels, int(det["class_id"])),
                "confidence": float(det["score"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "mask_format": "polygon",
                "mask": polygon,
            }
        )
    return segments

def encode_segments(segments: list[dict]) -> tuple[str, int]:
    """``data`` object of a ``segmentation`` metadata message, and how many segments were dropped.

    Segments that do not fit the byte budget are dropped lowest-confidence first.
    """
    ordered = sorted(segments, key=lambda segment: segment["confidence"], reverse=True)
    kept = []
    total = len('{"segments":[]}')
    for segment in ordered:
        entry_bytes = len(json.dumps(segment, separators=(",", ":"))) + 1
        if total + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total += entry_bytes
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":")), len(ordered) - len(kept)
