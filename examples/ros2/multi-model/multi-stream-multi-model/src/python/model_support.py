"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import collections

from dataclasses import dataclass, field

from pathlib import Path

import glob

import json

import math

import os

import struct

import sys

import threading

import time

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_multi_stream_multi_model')) / "common" / "config.yaml"

TASK_DETECTION = "detection"

TASK_SEGMENTATION = "segmentation"

TASK_POSE = "pose"

TASKS = (TASK_DETECTION, TASK_SEGMENTATION, TASK_POSE)

DECODE_YOLOV8 = "yolov8"

DECODE_YOLO26 = "yolo26"

DECODE_FAMILIES = (DECODE_YOLOV8, DECODE_YOLO26)

METADATA_TYPES = {
    TASK_DETECTION: "object-detection",
    TASK_SEGMENTATION: "segmentation",
    TASK_POSE: "pose-estimation",
}

METADATA_BYTE_BUDGET = 32768

SOURCE_FRAME_QUEUE_DEPTH = 4

DECODER_HEADROOM_BUFFERS = 4

MAX_INFLIGHT_PER_STREAM = 16

MASK_STRIDE = 4

MASK_GRID = 160

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class StreamConfig:
    index: int
    url: str
    task: str
    decode: str
    model_path: str

@dataclass(frozen=True)
class AppConfig:
    streams: list[StreamConfig]
    labels_path: Path
    # Encoded RTSP path used for every stream: "h264" or "h265". Kept as a
    # string because load_app_config runs before the lazy pyneat import.
    codec: str = "h264"
    latency_ms: int = 100
    tcp: bool = True
    frames: int = 0
    fps: int = 0
    max_inflight_per_stream: int = 4
    min_score: float = 0.30
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
    mask_threshold: float = 0.50
    mask_alpha: float = 0.55
    min_keypoint_visibility: float = 0.30
    save_dir: str = ""
    save_every: int = 0

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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)

def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)

def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value

def parse_input_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    raise ValueError("input.codec must be h264/avc or h265/hevc")

def parse_task(value: str, index: int) -> str:
    task = value.strip().lower()
    if task not in TASKS:
        raise ValueError(f"streams[{index}].task must be one of {', '.join(TASKS)}")
    return task

def parse_decode(value: str, index: int) -> str:
    decode = value.strip().lower()
    if decode not in DECODE_FAMILIES:
        raise ValueError(f"streams[{index}].decode must be one of {', '.join(DECODE_FAMILIES)}")
    return decode

def parse_streams(raw: object) -> list[StreamConfig]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("streams must be a non-empty list")
    streams = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"streams[{index}] must be a mapping with url, task, and model")
        url = string_or(entry, "url")
        if not url.strip():
            raise ValueError(f"streams[{index}].url must be a non-empty string")
        model_path = string_or(entry, "model")
        if not model_path.strip():
            raise ValueError(f"streams[{index}].model must be a non-empty string")
        streams.append(
            StreamConfig(
                index=index,
                url=url,
                task=parse_task(string_or(entry, "task", TASK_DETECTION), index),
                decode=parse_decode(string_or(entry, "decode", DECODE_YOLO26), index),
                model_path=model_path,
            )
        )
    return streams

def validate_config(cfg: AppConfig) -> None:
    if not cfg.streams:
        raise ValueError("streams must be set")
    if len(cfg.streams) > 4:
        raise ValueError("this example supports up to four streams")
    # Path("") normalizes to ".", so an empty value would otherwise pass as the current folder.
    if str(cfg.labels_path) in ("", "."):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps != 0:
        raise ValueError("inference.fps must be 0; configure frame rate at the RTSP source")
    if not 0 < cfg.max_inflight_per_stream <= MAX_INFLIGHT_PER_STREAM:
        raise ValueError(
            f"inference.max_inflight_per_stream must be between 1 and {MAX_INFLIGHT_PER_STREAM}"
        )
    if cfg.frames != 0 and cfg.frames <= cfg.warmup_frames:
        raise ValueError("inference.frames must be 0 or greater than runtime.warmup_frames")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    port_count = len(cfg.streams)
    for name, base in (("video_port_base", cfg.video_port_base),
                       ("metadata_port_base", cfg.metadata_port_base)):
        if base > 65536 - port_count:
            raise ValueError(f"output.insight.{name} range must fit within 1..65535")
    if cfg.video_enabled and not (
        cfg.video_port_base + port_count <= cfg.metadata_port_base
        or cfg.metadata_port_base + port_count <= cfg.video_port_base
    ):
        raise ValueError("output.insight video and metadata port ranges must not overlap")
    if not 0.0 <= cfg.mask_threshold <= 1.0:
        raise ValueError("output.mask_threshold must be between 0 and 1")
    if not 0.0 <= cfg.mask_alpha <= 1.0:
        raise ValueError("output.mask_alpha must be between 0 and 1")
    if not 0.0 <= cfg.min_keypoint_visibility <= 1.0:
        raise ValueError("output.min_keypoint_visibility must be between 0 and 1")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")

def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(get_package_share_directory('neat_multi_stream_multi_model')) / "common" / "coco_label.txt"

    cfg = AppConfig(
        streams=parse_streams(raw.get("streams")),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        codec=parse_input_codec(string_or(input_cfg, "codec", "h264")),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        min_score=float_or(inference, "min_score", 0.30),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        mask_threshold=float_or(output, "mask_threshold", 0.50),
        mask_alpha=float_or(output, "mask_alpha", 0.55),
        min_keypoint_visibility=float_or(output, "min_keypoint_visibility", 0.30),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
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

def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"

def round_pixel(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)

def round_confidence(value: float) -> float:
    return math.floor(value * 1000.0 + 0.5) / 1000.0

def tensor_to_numpy(tensor) -> object:
    # A frame with nothing detected decodes to a zero-row tensor. Copying one throws, because a
    # zero-byte payload has nothing to map, so treat "no rows" as an empty result.
    shape = tuple(getattr(tensor, "shape", ()) or ())
    if not shape or 0 in shape:
        return np.empty(shape or (0,), dtype=np.float32)
    return np.asarray(tensor.to_numpy(copy=True))

def extract_tensors(sample) -> list:
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)

    tensors = []
    for field_sample in sample.fields:
        tensors.extend(extract_tensors(field_sample))
    return tensors

def extract_tensor_bbox_payload(sample, tensor=None) -> bytes:
    tensor = tensor if tensor is not None else getattr(sample, "tensor", None)
    if tensor is None:
        raise RuntimeError("capture_missing_tensor")
    fmt = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    semantic = getattr(tensor, "semantic", None)
    tess = getattr(semantic, "tess", None)
    if not fmt and tess is not None:
        fmt = getattr(tess, "format", "")
    fmt = str(fmt).upper()
    if fmt and fmt != "BBOX":
        raise RuntimeError(f"capture_expected_bbox format={fmt}")
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("capture_empty_payload")
    return payload

def extract_bbox_payload(sample) -> bytes:
    if sample.kind == pyneat.SampleKind.Bundle:
        for field_sample in sample.fields:
            try:
                return extract_bbox_payload(field_sample)
            except RuntimeError:
                continue
        raise RuntimeError("bundle missing BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return extract_tensor_bbox_payload(sample, sample.tensors[0])
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError("capture_expected_tensor")
    return extract_tensor_bbox_payload(sample)

def parse_boxes_strict(payload: bytes, img_w: int, img_h: int, expected_topk: int) -> list[dict]:
    """Decode the BoxDecode detection payload: a count header then 24-byte box records."""
    if len(payload) < 4:
        raise RuntimeError("bbox buffer too small")

    header = struct.unpack_from("<I", payload, 0)[0]
    max_boxes = (len(payload) - 4) // 24
    if header > max_boxes:
        raise RuntimeError("bbox header exceeds payload count")
    if expected_topk > 0 and header > expected_topk:
        raise RuntimeError("bbox header exceeds expected topk")

    boxes = []
    offset = 4
    for _ in range(header):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        boxes.append(
            {
                "x1": max(0.0, min(float(x), float(img_w))),
                "y1": max(0.0, min(float(y), float(img_h))),
                "x2": max(0.0, min(float(x + w), float(img_w))),
                "y2": max(0.0, min(float(y + h), float(img_h))),
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes

def decode_segments(sample, frame_w: int, frame_h: int, max_detections: int) -> list[dict]:
    """Decode one segmentation sample into instances carrying a 160x160 mask each."""
    tensors = extract_tensors(sample)
    if not tensors:
        raise RuntimeError("segmentation sample carried no tensors")

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
        for row, mask in zip(
            boxes.reshape((-1, 6)), masks.reshape((-1, MASK_GRID, MASK_GRID))
        ):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score),
                    "class_id": int(class_id),
                    "mask": mask,
                }
            )
            if len(detections) >= max_detections:
                return detections
    return detections

def decode_poses(sample, frame_w: int, frame_h: int, max_poses: int) -> list[dict]:
    """Decode one pose sample into pose records in source-frame pixel space.

    ``decode_pose`` returns boxes as ``[N, 6]`` (x1, y1, x2, y2, score, class_id) and
    keypoints as ``[N, 17, 3]`` (x, y, visibility), positionally aligned. Clamping applies to
    box coordinates only; keypoints are emitted unclamped, so the visibility floor and the
    drawing code are what keep stray joints off the overlay.
    """
    tensors = extract_tensors(sample)
    if not tensors:
        raise RuntimeError("pose sample carried no tensors")

    decoded = pyneat.decode_pose(tensors, clamp_to=(frame_w, frame_h), top_k=max_poses)
    poses: list[dict] = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).reshape((-1, 6))
        keypoints = tensor_to_numpy(item.keypoints).reshape((-1, len(COCO_KEYPOINT_NAMES), 3))
        if boxes.shape[0] != keypoints.shape[0]:
            raise RuntimeError(
                f"pose decode returned {boxes.shape[0]} boxes but {keypoints.shape[0]} keypoint sets"
            )
        for box, points in zip(boxes, keypoints):
            if len(poses) >= max_poses:
                return poses
            poses.append(
                {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "score": float(box[4]),
                    "keypoints": [
                        {"x": float(x), "y": float(y), "visibility": float(v)}
                        for x, y, v in points
                    ],
                }
            )
    return poses

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
    """Map a frame rectangle onto the mask grid through the model's letterbox transform."""
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

    Empty when the thresholded mask holds nothing Insight can draw. ``threshold`` is a fraction
    of full scale, as ``output.mask_threshold`` is. Upscaling before thresholding is what makes
    the outline match the rendered overlay.
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
    # Contour points lie inside frame_rect, which is already clamped to the frame, so shifting
    # them into frame space cannot leave the image.
    x0, y0 = frame_rect[0], frame_rect[1]
    return [[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in polygon]

def detection_metadata_data(
    boxes: list[dict], labels: list[str], frame_w: int, frame_h: int
) -> tuple[dict, int]:
    """``data`` object of an ``object-detection`` message, and how many boxes the budget dropped.

    A raised ``inference.max_detections`` can overrun one message, so boxes beyond the byte
    budget are dropped lowest-confidence first, as segmentation and pose do.
    """
    objects = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, int(box["x2"] - box["x1"]))
        h = max(0, int(box["y2"] - box["y1"]))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        objects.append(
            {
                "id": f"obj_{index}",
                "label": class_name(labels, int(box["class_id"])),
                "confidence": round_confidence(float(box["score"])),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )

    ordered = sorted(objects, key=lambda obj: obj["confidence"], reverse=True)
    kept = []
    total = len('{"objects":[]}')
    for obj in ordered:
        entry_bytes = len(json.dumps(obj, separators=(",", ":"))) + 1
        if total + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total += entry_bytes
        kept.append(obj)
    return {"objects": kept}, len(ordered) - len(kept)

def segmentation_metadata_data(
    detections: list[dict],
    labels: list[str],
    frame_shape: tuple[int, ...],
    mask_threshold: float,
) -> tuple[dict, int]:
    """``data`` object of a ``segmentation`` message, and how many segments the budget dropped.

    The silhouette ships as ``mask_format: "polygon"`` in frame pixels: a polygon keeps the
    contour the mask has at frame resolution, while run-length encoding would ship a mask crop
    only a few pixels across and leave Insight to stretch it.
    """
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
                "confidence": round_confidence(float(det["score"])),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "mask_format": "polygon",
                "mask": polygon,
            }
        )

    ordered = sorted(segments, key=lambda segment: segment["confidence"], reverse=True)
    kept = []
    total = len('{"segments":[]}')
    for segment in ordered:
        entry_bytes = len(json.dumps(segment, separators=(",", ":"))) + 1
        if total + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total += entry_bytes
        kept.append(segment)
    return {"segments": kept}, len(ordered) - len(kept)

def pose_metadata_data(poses: list[dict]) -> tuple[dict, int]:
    """``data`` object of a ``pose-estimation`` message, and how many poses the budget dropped.

    Pixel rounding and three-decimal confidence preserve overlay precision. A raised
    ``inference.max_detections`` can still overrun one message, so poses beyond the byte budget
    are dropped lowest-confidence first, as segmentation does.
    """
    published = []
    for index, pose in enumerate(poses, start=1):
        keypoints = [
            {
                "name": COCO_KEYPOINT_NAMES[k],
                "x": round_pixel(point["x"]),
                "y": round_pixel(point["y"]),
                "confidence": round_confidence(point["visibility"]),
            }
            for k, point in enumerate(pose["keypoints"])
        ]
        published.append(
            {
                "id": f"pose_{index}",
                "label": "person",
                "confidence": round_confidence(pose["score"]),
                "bbox": [
                    round_pixel(pose["x1"]),
                    round_pixel(pose["y1"]),
                    round_pixel(max(0.0, pose["x2"] - pose["x1"])),
                    round_pixel(max(0.0, pose["y2"] - pose["y1"])),
                ],
                "keypoints": keypoints,
            }
        )

    ordered = sorted(published, key=lambda pose: pose["confidence"], reverse=True)
    kept = []
    total = len('{"poses":[]}')
    for pose in ordered:
        entry_bytes = len(json.dumps(pose, separators=(",", ":"))) + 1
        if total + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total += entry_bytes
        kept.append(pose)
    return {"poses": kept}, len(ordered) - len(kept)

ENCODED_TO_DECODER = "encoded_to_decoder"

ENCODED_TO_VIDEO = "encoded_to_video"

def decode_type_for(task: str, decode: str):
    """Map a stream's task and declared head layout onto an on-device BoxDecode family."""
    yolo26 = decode == DECODE_YOLO26
    if task == TASK_POSE:
        return pyneat.BoxDecodeType.YoloV26Pose if yolo26 else pyneat.BoxDecodeType.YoloV8Pose
    if task == TASK_SEGMENTATION:
        return pyneat.BoxDecodeType.YoloV26Seg if yolo26 else pyneat.BoxDecodeType.YoloV8Seg
    return pyneat.BoxDecodeType.YoloV26 if yolo26 else pyneat.BoxDecodeType.YoloV8

def build_model(cfg: AppConfig, stream_cfg: StreamConfig, frame_w: int, frame_h: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    if frame_w > 0 and frame_h > 0:
        opt.preprocess.input_max_width = frame_w
        opt.preprocess.input_max_height = frame_h
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = decode_type_for(stream_cfg.task, stream_cfg.decode)
    if stream_cfg.task == TASK_POSE:
        # YOLO pose ships single-class ("person") score heads. The packaged MPK still declares
        # the 80-class detector metadata, so state the real class count rather than inheriting it.
        opt.num_classes = 1
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(stream_cfg.model_path, opt)

def build_model_run_options(model_queue_depth: int) -> pyneat.RunOptions:
    """Block, not KeepLatest: the feeder already bounds frames in flight, and Block keeps push
    and pull strictly FIFO-paired so a pulled result belongs to the frame that was pushed."""
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = max(1, model_queue_depth)
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options

MODEL_INPUT = "image"

MODEL_OUTPUT = "results"

MAX_CONSECUTIVE_FRAME_FAILURES = 30
