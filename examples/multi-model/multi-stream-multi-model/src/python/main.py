"""Multi-camera RTSP example running a different model per stream, published to Insight."""

from __future__ import annotations

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

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

TASK_DETECTION = "detection"
TASK_SEGMENTATION = "segmentation"
TASK_POSE = "pose"
TASKS = (TASK_DETECTION, TASK_SEGMENTATION, TASK_POSE)

DECODE_YOLOV8 = "yolov8"
DECODE_YOLO26 = "yolo26"
DECODE_FAMILIES = (DECODE_YOLOV8, DECODE_YOLO26)
""" ``streams[].decode`` values. The family follows the shape of the archive's detection head,
not the model's version number: YOLO26 packages carry raw l/t/r/b distance heads, while the
Model Zoo YOLO11 and YOLOv8 packages carry the YOLOv8 head layout, so YOLO11 uses ``yolov8``.
"""

METADATA_TYPES = {
    TASK_DETECTION: "object-detection",
    TASK_SEGMENTATION: "segmentation",
    TASK_POSE: "pose-estimation",
}
"""Insight overlay contract per task. Insight picks its renderer from this string."""

# Insight drops a metadata datagram it cannot reassemble, so segmentation polygons are
# budgeted rather than truncated mid-frame.
METADATA_BYTE_BUDGET = 32768

# BoxDecode emits segmentation masks on a 160x160 grid covering the model's 640x640
# letterboxed input, so one mask cell spans four input pixels.
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
"""COCO keypoint names in the order the BoxDecode pose payload emits them.

The order is fixed by the wire format, not by configuration. Insight joins skeleton edges by
name, so these strings are part of the published metadata contract.
"""

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
"""Skeleton edges as index pairs into ``COCO_KEYPOINT_NAMES``, matching Insight's topology."""

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


@dataclass
class StreamRuntime:
    index: int
    task: str
    url: str
    model_path: str
    source_options: object
    model: object
    metadata_sender: object
    labels: list[str]
    profile: "ProfileWindow"
    latest_debug_frame: object | None
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int
    processed: int = 0
    dropped_segments: int = 0
    closed: bool = False
    # This stream's model, in a Run of its own. Four model graphs in one Run couple to each
    # other: one stream stalls for good while its neighbours keep running at the source rate.
    model_graph: object = None
    model_run: object = None
    # Frames handed to the model but not yet pulled back. This is the backpressure bound;
    # OverflowPolicy does not provide one, because push() returns as soon as the frame lands in
    # the graph's internal edge queue.
    in_flight: int = 0
    feed_done: bool = False
    debug_frame_lock: object = field(default_factory=threading.Lock)
    # Timing of frames handed to the model, oldest first.
    #
    # The model lives in its own Run, and a tensor pushed into it arrives with no PTS, so its
    # results come back stamped -1. Insight correlates metadata to video by timestamp, so the
    # camera's own PTS has to be carried across the Run boundary by hand. Push and pull are
    # FIFO-paired under OverflowPolicy.Block, which makes a queue enough to re-pair them.
    pending: object = field(default_factory=collections.deque)
    pending_lock: object = field(default_factory=threading.Lock)


@dataclass
class AppRuntime:
    # Every stream's ``rtsp -> decode`` chain plus its video passthrough, in ONE graph and Run.
    # Neat requests a decoder-admission lease only when a single graph holds more than one
    # decoder, and without that lease four concurrent decoders do not reach the source rate.
    source_graph: object
    source_run: object | None
    streams: list[StreamRuntime] = field(default_factory=list)


class ProfileWindow:
    def __init__(self, enabled: bool, stream_index: int, task: str, interval: int = 100) -> None:
        self.enabled = enabled
        self.stream_index = stream_index
        self.task = task
        self.interval = interval
        self.frames = 0
        self.instances = 0
        self.start_ms = 0.0
        self.result_pull_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(self, result_pull_ms: float, metadata_send_ms: float, instance_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.instances += instance_count
        self.result_pull_ms += result_pull_ms
        self.metadata_send_ms += metadata_send_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        output_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile stream={self.stream_index} task={self.task}] frames={self.frames} "
            f"output_fps={output_fps} "
            f"avg_result_pull_ms={self.result_pull_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_instances={self.instances / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.instances = 0
        self.start_ms = 0.0
        self.result_pull_ms = 0.0
        self.metadata_send_ms = 0.0


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


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-camera RTSP multi-model Insight example"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


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
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
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
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if cfg.max_inflight_per_stream != -1 and cfg.max_inflight_per_stream <= 0:
        raise ValueError("inference.max_inflight_per_stream must be -1 or > 0")
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
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

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


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (255, 194, 0),
        (168, 153, 44),
    ]
    return palette[max(0, class_id) % len(palette)]


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


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


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


def first_tensor_from_sample(sample):
    tensors = extract_tensors(sample)
    return tensors[0] if tensors else None


def tensor_bgr_from_decoded(tensor):
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


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
) -> dict:
    """Build the ``data`` object Insight's ``object-detection`` overlay consumes."""
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
    return {"objects": objects}


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


def pose_metadata_data(poses: list[dict]) -> dict:
    """Build the ``data`` object Insight's ``pose-estimation`` overlay consumes.

    Pixel rounding and three-decimal confidence preserve overlay precision while keeping the
    configured pose maximum within Core's logical metadata-message limit.
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
    return {"poses": published}


def rtsp_codec(codec: str):
    """Map the parsed `input.codec` config token onto the Core RTSP codec selector."""
    return pyneat.RtspCodec.H265 if codec == "h265" else pyneat.RtspCodec.H264


def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    if fps <= 0:
        raise RuntimeError("failed to probe RTSP frame rate")
    return width, height, fps


def build_source_options(cfg: AppConfig, url: str, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.codec = rtsp_codec(cfg.codec)
    opt.dec_width = width
    opt.dec_height = height
    if cfg.codec == "h264":
        opt.fallback_h264_width = width
        opt.fallback_h264_height = height
    opt.source_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def output_caps_enabled(caps) -> bool:
    return caps.enable or caps.width > 0 or caps.height > 0 or caps.fps > 0


def build_encoded_source_graph(opt) -> pyneat.Graph:
    source = pyneat.Graph("rtsp_encoded_source")

    encoded_opt = pyneat.RtspEncodedInputOptions()
    encoded_opt.url = opt.url
    encoded_opt.codec = opt.codec
    encoded_opt.latency_ms = opt.latency_ms
    encoded_opt.tcp = opt.tcp
    encoded_opt.source_fps = opt.source_fps
    if opt.codec == pyneat.RtspCodec.H264:
        encoded_opt.fallback_h264_width = opt.fallback_h264_width
        encoded_opt.fallback_h264_height = opt.fallback_h264_height
    source.add(pyneat.groups.rtsp_encoded_input(encoded_opt))
    return source


def encoded_format_tag(codec):
    return pyneat.Format.H265 if codec == pyneat.RtspCodec.H265 else pyneat.Format.H264


def encoded_decode_input_options(codec):
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = encoded_format_tag(codec)
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
    return opt


def encoded_video_input_options(codec):
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = encoded_format_tag(codec)
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
    elif hasattr(opt, "use_simaai_pool"):
        opt.use_simaai_pool = False
    return opt


def build_decode_graph(input_name: str, opt) -> pyneat.Graph:
    decode = pyneat.Graph("decode")
    use_h265 = opt.codec == pyneat.RtspCodec.H265

    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H265 if use_h265 else pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = opt.sima_allocator_type
    dec.out_format = pyneat.Format.NV12
    dec.decoder_name = opt.decoder_name
    dec.raw_output = opt.decoder_raw_output
    dec.next_element = opt.decoder_next_element
    dec.dec_width = opt.dec_width
    dec.dec_height = opt.dec_height
    dec.dec_fps = opt.source_fps
    dec.num_buffers = opt.num_buffers
    decode.connect(
        pyneat.nodes.input(input_name, encoded_decode_input_options(opt.codec)),
        pyneat.nodes.sima_decode(dec),
    )
    if opt.use_videoconvert:
        decode.add(pyneat.nodes.video_convert())
    if opt.use_videoscale:
        decode.add(pyneat.nodes.video_scale())
    if output_caps_enabled(opt.output_caps):
        decode.add(
            pyneat.nodes.caps_raw(
                "NV12",
                opt.output_caps.width,
                opt.output_caps.height,
                opt.output_caps.fps,
                opt.output_caps.memory,
            )
        )
    if opt.extra_fragment:
        decode.add(pyneat.nodes.custom(opt.extra_fragment))
    return decode


def build_video_sender_graph(input_name: str, codec, video_options) -> pyneat.Graph:
    video = pyneat.Graph("video_sender")
    video.connect(
        pyneat.nodes.input(input_name, encoded_video_input_options(codec)),
        pyneat.groups.video_sender(video_options),
    )
    return video


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


def build_source_run_options() -> pyneat.RunOptions:
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def build_model_run_options(model_queue_depth: int) -> pyneat.RunOptions:
    """Block, not KeepLatest: the feeder already bounds frames in flight, and Block keeps push
    and pull strictly FIFO-paired so a pulled result belongs to the frame that was pushed."""
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = max(1, model_queue_depth)
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def save_frames_enabled(cfg: AppConfig) -> bool:
    return bool(cfg.save_dir) and cfg.save_every > 0


def stream_id_for(stream_index: int) -> str:
    return f"stream{stream_index}"


def source_frame_name(stream_index: int) -> str:
    return f"frame_{stream_index}"


# Every model graph uses the same endpoint names; they live in separate Runs.
MODEL_INPUT = "image"
MODEL_OUTPUT = "results"


def realtime_link(stream_index: int, queue_depth: int):
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.queue_depth = queue_depth
    link.stream_id = stream_id_for(stream_index)
    return link


def make_nv12_input_options(width: int, height: int, fps: int):
    """The host pushes decoded frames in here, so the input is a plain NV12 image endpoint."""
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Image
    opt.format = pyneat.Format.NV12
    opt.width = width
    opt.height = height
    opt.depth = 1
    opt.max_width = width
    opt.max_height = height
    opt.max_depth = 1
    opt.fps_n = max(1, fps)
    opt.fps_d = 1
    return opt


def build_model_graph(cfg: AppConfig, stream: StreamRuntime) -> pyneat.Graph:
    graph = pyneat.Graph(f"model_{stream.index}")
    graph.add(
        pyneat.nodes.input(
            MODEL_INPUT, make_nv12_input_options(stream.frame_w, stream.frame_h, stream.output_fps)
        )
    )
    graph.add(stream.model)
    # At least as many buffers as frames in flight, or finished results back up inside the graph
    # and throttle the pipelining the feeder is paying for.
    graph.add(
        pyneat.nodes.output(
            MODEL_OUTPUT, pyneat.OutputOptions.every_frame(max(1, cfg.max_inflight_per_stream))
        )
    )
    return graph


def copy_nv12_for_model(tensor):
    """Move a decoded frame out of the zero-copy pool into a tensor the model input accepts.

    The frame has to leave the pool: it outlives the pulled sample, and holding pool buffers
    starves the decoder. It also has to be EV74-placed, because the model's input route is
    device-visible and rejects host-backed tensors rather than silently copying them. ``cvu()``
    does both and keeps the NV12 plane layout, which rebuilding the tensor from a NumPy array
    would lose.
    """
    return tensor.contiguous().cvu()


def make_video_options(cfg: AppConfig, stream_index: int):
    video_options = pyneat.VideoSenderOptions.passthrough(rtsp_codec(cfg.codec))
    video_options.host = cfg.insight_host
    video_options.channel = stream_index
    video_options.video_port_base = cfg.video_port_base
    video_options.async_ = True
    return video_options


def build_stream_runtime(cfg: AppConfig, stream_cfg: StreamConfig, labels: list[str]) -> StreamRuntime:
    frame_w, frame_h, fps = probe_rtsp(stream_cfg.url)
    output_fps = cfg.fps if cfg.fps > 0 else fps

    source_options = build_source_options(cfg, stream_cfg.url, fps, frame_w, frame_h)
    model = build_model(cfg, stream_cfg, frame_w, frame_h)

    video_port = 0
    if cfg.video_enabled:
        video_port = make_video_options(cfg, stream_cfg.index).video_port

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = stream_cfg.index
    metadata_options.metadata_port_base = cfg.metadata_port_base
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(
        f"[stream {stream_cfg.index}] task={stream_cfg.task} decode={stream_cfg.decode} "
        f"model={Path(stream_cfg.model_path).name} rtsp={stream_cfg.url} "
        f"stream={frame_w}x{frame_h}@{output_fps} insight={cfg.insight_host} "
        f"video={video_port if cfg.video_enabled else 'disabled'} "
        f"metadata={metadata_sender.metadata_port()}"
    )
    return StreamRuntime(
        index=stream_cfg.index,
        task=stream_cfg.task,
        url=stream_cfg.url,
        model_path=stream_cfg.model_path,
        source_options=source_options,
        model=model,
        metadata_sender=metadata_sender,
        labels=labels,
        profile=ProfileWindow(cfg.profile, stream_cfg.index, stream_cfg.task),
        latest_debug_frame=None,
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=output_fps,
        video_port=video_port,
    )


def connect_source_stream(app: AppRuntime, cfg: AppConfig, stream: StreamRuntime) -> None:
    """Add one stream's ``rtsp -> decode -> output`` chain, plus its video passthrough, to the
    shared source graph.

    Video stays in-graph as an encoded passthrough: Insight correlates the RTP timestamp with the
    metadata timestamp, and re-encoding on the host would cost a copy this example does not need.
    """
    source = build_encoded_source_graph(stream.source_options)
    decoder = build_decode_graph("decode_h264", stream.source_options)

    frames = pyneat.Graph(f"frames_{stream.index}")
    # A few buffers, and dropping: Insight can only draw an overlay on a frame it has metadata
    # for, so a frame skipped here shows up as a blink in the viewer. Depth absorbs a brief
    # feeder hiccup; dropping still sheds load if the model genuinely cannot keep up.
    frame_output = pyneat.OutputOptions.every_frame(4)
    frame_output.drop = True
    frames.add(pyneat.nodes.output(source_frame_name(stream.index), frame_output))

    if cfg.video_enabled:
        encoded_branch = pyneat.graphs.branch("encoded", ["decode_h264", "video_h264"])
        app.source_graph.connect(source, encoded_branch)
        app.source_graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3))

        video_options = make_video_options(cfg, stream.index)
        app.source_graph.connect(
            encoded_branch,
            build_video_sender_graph("video_h264", rtsp_codec(cfg.codec), video_options),
            realtime_link(stream.index, 3),
        )
    else:
        app.source_graph.connect(source, decoder, realtime_link(stream.index, 3))

    app.source_graph.connect(decoder, frames)


def send_metadata(cfg: AppConfig, stream: StreamRuntime, stamp, results: list[dict]) -> int:
    """Publish one message on this stream's Insight channel, typed by the stream's task."""
    dropped = 0
    if stream.task == TASK_DETECTION:
        data = detection_metadata_data(results, stream.labels, stream.frame_w, stream.frame_h)
    elif stream.task == TASK_SEGMENTATION:
        data, dropped = segmentation_metadata_data(
            results, stream.labels, (stream.frame_h, stream.frame_w), cfg.mask_threshold
        )
    else:
        data = pose_metadata_data(results)

    pts_ns, source_frame_id = stamp
    timestamp_ms = int(pts_ns // 1_000_000) if pts_ns >= 0 else -1
    frame_id = str(source_frame_id) if source_frame_id >= 0 else ""
    if not stream.metadata_sender.send_metadata(
        METADATA_TYPES[stream.task],
        json.dumps(data, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    ):
        print(f"[warn] stream {stream.index}: insight metadata send failed", file=sys.stderr)
    return dropped


def decode_results(cfg: AppConfig, stream: StreamRuntime, sample) -> list[dict]:
    if stream.task == TASK_DETECTION:
        payload = extract_bbox_payload(sample)
        return parse_boxes_strict(payload, stream.frame_w, stream.frame_h, cfg.max_detections)
    if stream.task == TASK_SEGMENTATION:
        return decode_segments(sample, stream.frame_w, stream.frame_h, cfg.max_detections)
    return decode_poses(sample, stream.frame_w, stream.frame_h, cfg.max_detections)


def draw_boxes(frame, boxes: list[dict], labels: list[str], min_score: float) -> None:
    for box in boxes:
        score = float(box["score"])
        if score < min_score:
            continue
        class_id = int(box["class_id"])
        color = class_color(class_id)
        x0, y0, x1, y1 = frame_rect_for_detection(box, frame.shape)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            frame,
            f"{class_name(labels, class_id)} {score:.2f}",
            (x0, max(0, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_segments(
    frame, detections: list[dict], labels: list[str], cfg: AppConfig
) -> None:
    for det in detections:
        if float(det["score"]) < cfg.min_score:
            continue
        mask = det.get("mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        x0, y0, x1, y1 = frame_rect_for_detection(det, frame.shape)
        roi_mask = project_letterbox_mask_roi(mask, frame.shape, (x0, y0, x1, y1))
        _, binary_mask = cv2.threshold(roi_mask, cfg.mask_threshold * 255.0, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binary_mask) > 0:
            color = class_color(int(det["class_id"]))
            roi = frame[y0:y1, x0:x1]
            mask_color = np.full(roi.shape, color, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0)
            cv2.copyTo(blended, binary_mask, roi)
    draw_boxes(frame, detections, labels, cfg.min_score)


def draw_poses(frame, poses: list[dict], min_visibility: float) -> None:
    color = (10, 249, 72)

    def visible(point: dict) -> bool:
        return float(point["visibility"]) >= min_visibility

    for pose in poses:
        x0, y0, x1, y1 = frame_rect_for_detection(pose, frame.shape)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        keypoints = pose["keypoints"]
        for start, end in COCO_SKELETON:
            if not (visible(keypoints[start]) and visible(keypoints[end])):
                continue
            cv2.line(
                frame,
                (round_pixel(keypoints[start]["x"]), round_pixel(keypoints[start]["y"])),
                (round_pixel(keypoints[end]["x"]), round_pixel(keypoints[end]["y"])),
                color,
                2,
                cv2.LINE_AA,
            )
        for point in keypoints:
            if not visible(point):
                continue
            cv2.circle(
                frame, (round_pixel(point["x"]), round_pixel(point["y"])), 3, (255, 194, 0), -1
            )


def maybe_save_debug_frame(cfg: AppConfig, stream: StreamRuntime, results: list[dict]) -> None:
    if not save_frames_enabled(cfg) or stream.processed % cfg.save_every != 0:
        return
    with stream.debug_frame_lock:
        if stream.latest_debug_frame is None:
            return
        frame = stream.latest_debug_frame.copy()
    if stream.task == TASK_DETECTION:
        draw_boxes(frame, results, stream.labels, cfg.min_score)
    elif stream.task == TASK_SEGMENTATION:
        draw_segments(frame, results, stream.labels, cfg)
    else:
        draw_poses(frame, results, cfg.min_keypoint_visibility)

    out_path = Path(cfg.save_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def process_output_sample(
    cfg: AppConfig, stream: StreamRuntime, sample, stamp, result_pull_ms: float
) -> None:
    if cfg.frames > 0 and stream.processed >= cfg.frames:
        return

    results = decode_results(cfg, stream, sample)

    stream.processed += 1
    if stream.processed <= cfg.warmup_frames:
        return

    metadata_start = time_ms()
    stream.dropped_segments += send_metadata(cfg, stream, stamp, results)
    metadata_end = time_ms()
    maybe_save_debug_frame(cfg, stream, results)
    stream.profile.add(result_pull_ms, metadata_end - metadata_start, len(results))


def run_stream_feeder(app: AppRuntime, cfg: AppConfig, stream: StreamRuntime) -> None:
    """Pull this stream's decoded frames from the shared source Run and hand them to its model.

    The in-flight gate is the real backpressure. ``OverflowPolicy.Block`` does not bound in-graph
    buffering: ``push`` returns as soon as the frame lands in the graph's internal edge queue, so
    a model slower than its camera accumulates seconds of latency while still reporting the source
    rate. Frames arriving while the gate is closed are dropped, which is what a live camera wants.
    """
    frame_name = source_frame_name(stream.index)
    depth = max(1, cfg.max_inflight_per_stream)
    try:
        while not stream.closed:
            sample = app.source_run.pull(frame_name, 50)
            if sample is None:
                continue
            # Wait for a slot rather than discarding the frame: every frame dropped here is a
            # frame Insight renders with no overlay. Bounded, so a wedged model cannot hang the
            # feeder.
            waited = 0
            while stream.in_flight >= depth and waited < 50 and not stream.closed:
                time.sleep(0.001)
                waited += 1
            if stream.in_flight >= depth:
                continue
            tensor = first_tensor_from_sample(sample)
            if tensor is None:
                continue
            try:
                frame = copy_nv12_for_model(tensor)
            except Exception as exc:
                print(f"[warn] stream {stream.index} frame copy: {exc}", file=sys.stderr)
                continue
            if save_frames_enabled(cfg):
                with stream.debug_frame_lock:
                    stream.latest_debug_frame = tensor_bgr_from_decoded(tensor)
            # Recorded before the push: under Block the result can come back the instant push
            # returns, and a consumer that found the queue empty would have nothing to pair it
            # with.
            with stream.pending_lock:
                stream.pending.append((sample.pts_ns, sample.frame_id))
            stream.in_flight += 1
            if not stream.model_run.push(MODEL_INPUT, [frame]):
                stream.in_flight -= 1
                with stream.pending_lock:
                    if stream.pending:
                        stream.pending.pop()
                break
    finally:
        stream.feed_done = True


def run_stream_consumer(cfg: AppConfig, stream: StreamRuntime) -> None:
    """Pull this stream's results back and publish them.

    Split from the feeder so the model holds several frames at once; a single thread doing
    push-then-pull would serialise every stage and cap the stream at the slowest one.
    """
    try:
        while cfg.frames <= 0 or stream.processed < cfg.frames:
            pull_start = time_ms()
            sample = stream.model_run.pull(MODEL_OUTPUT, 50)
            pull_end = time_ms()
            if sample is None:
                # Nothing in flight and the feeder has stopped: nothing left to wait for.
                if stream.feed_done and stream.in_flight <= 0:
                    break
                continue
            stream.in_flight -= 1
            with stream.pending_lock:
                stamp = stream.pending.popleft() if stream.pending else (-1, -1)
            process_output_sample(cfg, stream, sample, stamp, pull_end - pull_start)
    finally:
        stream.closed = True


def run_app(cfg: AppConfig) -> None:
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
    if save_frames_enabled(cfg):
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_path)
    app = AppRuntime(source_graph=pyneat.Graph(), source_run=None)
    for stream_cfg in cfg.streams:
        stream = build_stream_runtime(cfg, stream_cfg, labels)
        app.streams.append(stream)
        connect_source_stream(app, cfg, stream)
        stream.model_graph = build_model_graph(cfg, stream)
        stream.model_run = stream.model_graph.build(
            build_model_run_options(cfg.max_inflight_per_stream)
        )

    workers: list[threading.Thread] = []
    try:
        if cfg.profile:
            print(f"Backend:\n{app.source_graph.describe_backend()}")
        # Built last, and only once every model is loaded: build() starts the pipeline, so
        # building the sources earlier lets the first camera stream while the remaining models
        # load, filling its edge queue and killing it before anything pulls.
        app.source_run = app.source_graph.build(build_source_run_options())

        for stream in app.streams:
            workers.append(
                threading.Thread(target=run_stream_feeder, args=(app, cfg, stream), daemon=True)
            )
            workers.append(
                threading.Thread(target=run_stream_consumer, args=(cfg, stream), daemon=True)
            )
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        raise
    finally:
        for stream in app.streams:
            stream.closed = True
        for worker in workers:
            worker.join(timeout=2.0)
        for stream in app.streams:
            if stream.model_run is not None:
                stream.model_run.close()
        if app.source_run is not None:
            app.source_run.close()
        for stream in app.streams:
            stream.profile.flush()
            dropped = (
                f" dropped_segments={stream.dropped_segments}"
                if stream.task == TASK_SEGMENTATION
                else ""
            )
            print(
                f"[stream {stream.index}] task={stream.task} "
                f"processed={stream.processed}{dropped}"
            )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            tasks = ",".join(stream.task for stream in cfg.streams)
            print(
                f"Config validated: {args.config} (streams={len(cfg.streams)}, "
                f"tasks={tasks}, "
                f"max_inflight_per_stream={cfg.max_inflight_per_stream})"
            )
            return 0

        load_runtime_dependencies()
        run_app(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
