"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass

import glob

import json

import math

from pathlib import Path

import sys

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_multi_stream_obb_detector')) / "common" / "config.yaml"

MAX_STREAMS = 4

PULL_TIMEOUT_MS = 50

YOLOV8_NMS_IOU = 0.7

KIND_NAMES = {bool: "true or false", int: "an integer", float: "numeric", str: "a string"}

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: str
    family: str
    rtsp_urls: list[str]
    codec: str
    tcp: bool
    latency_ms: int
    frames: int
    max_inflight_per_stream: int
    max_inflight_total: int
    min_score: float
    max_detections: int
    insight_host: str
    video_port_base: int
    metadata_port_base: int
    debug_dir: str
    save_every: int

@dataclass(frozen=True)
class Detections:
    """One frame's oriented boxes in source-frame pixels, one array row per box.

    Each `boxes` row is cx, cy, width, height, angle_radians; a positive angle turns the
    box clockwise. Arrays keep a frame with hundreds of boxes cheap to publish.
    """

    boxes: object
    scores: object
    class_ids: object

def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module

def setting(raw: dict, key: str, default, kind: type):
    """Reads the dotted `key` from the parsed YAML, or `default` when it is absent."""
    value = raw
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None:
        return default
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ValueError(f"{key} must be {KIND_NAMES[kind]}")
    return value

def parse_family(value: str) -> str:
    lowered = value.lower()
    if lowered in {"yolo26", "yolov8", "yolo11"}:
        return lowered
    raise ValueError("model.family must be yolo26, yolov8, or yolo11")

def parse_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    if lowered in {"mjpeg", "jpeg"}:
        return "mjpeg"
    raise ValueError("input.codec must be h264/avc, h265/hevc, or mjpeg")

def parse_streams(raw: dict) -> list[str]:
    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    for index, url in enumerate(streams):
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
    return streams

def validate_config(cfg: AppConfig) -> None:
    checks = [
        (cfg.model_path, "model.path must be set"),
        (cfg.labels_path, "model.labels must be set"),
        (len(cfg.rtsp_urls) <= MAX_STREAMS, "streams supports up to four entries"),
        (cfg.insight_host, "output.insight.host must be set"),
        (cfg.latency_ms >= 0, "input.latency_ms must be >= 0"),
        (cfg.frames >= 0, "inference.frames must be >= 0"),
        (
            cfg.max_inflight_per_stream == -1 or cfg.max_inflight_per_stream > 0,
            "inference.max_inflight_per_stream must be -1 or > 0",
        ),
        (
            cfg.max_inflight_total == -1 or cfg.max_inflight_total > 0,
            "inference.max_inflight_total must be -1 or > 0",
        ),
        (0.0 <= cfg.min_score <= 1.0, "inference.min_score must be between 0 and 1"),
        (cfg.max_detections > 0, "inference.max_detections must be > 0"),
        (cfg.video_port_base > 0, "output.insight.video_port_base must be > 0"),
        (cfg.metadata_port_base > 0, "output.insight.metadata_port_base must be > 0"),
        (cfg.save_every >= 0, "output.save_every must be >= 0"),
    ]
    for passed, message in checks:
        if not passed:
            raise ValueError(message)

def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    default_labels = str(Path(get_package_share_directory('neat_multi_stream_obb_detector')) / "common" / "dota_label.txt")
    cfg = AppConfig(
        model_path=setting(raw, "model.path", "", str),
        labels_path=setting(raw, "model.labels", default_labels, str),
        family=parse_family(setting(raw, "model.family", "yolo26", str)),
        rtsp_urls=parse_streams(raw),
        codec=parse_codec(setting(raw, "input.codec", "h264", str)),
        tcp=setting(raw, "input.tcp", True, bool),
        latency_ms=setting(raw, "input.latency_ms", 100, int),
        frames=setting(raw, "inference.frames", 0, int),
        max_inflight_per_stream=setting(raw, "inference.max_inflight_per_stream", 4, int),
        max_inflight_total=setting(raw, "inference.max_inflight_total", 16, int),
        min_score=setting(raw, "inference.min_score", 0.30, float),
        max_detections=setting(raw, "inference.max_detections", 100, int),
        insight_host=setting(raw, "output.insight.host", "", str),
        video_port_base=setting(raw, "output.insight.video_port_base", 9000, int),
        metadata_port_base=setting(raw, "output.insight.metadata_port_base", 9100, int),
        debug_dir=setting(raw, "output.debug_dir", "", str),
        save_every=setting(raw, "output.save_every", 0, int),
    )
    validate_config(cfg)
    return cfg

def load_labels(labels_path: str) -> list[str]:
    path = Path(labels_path)
    if not path.is_file():
        raise RuntimeError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {path}")
    return labels

def build_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    # COCO_YOLO letterboxes, which keeps one scale on both axes so rotated boxes map back.
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    if cfg.family == "yolo26":
        opt.decode_type = pyneat.BoxDecodeType.YoloV26Obb
        # YOLO26 needs no NMS, and its oriented-box decoder requires it disabled.
        opt.nms_iou_threshold = 0.0
    else:
        # YOLOv8 and YOLO11 share one decoder, which removes overlapping boxes with rotated NMS.
        opt.decode_type = pyneat.BoxDecodeType.YoloV8Obb
        opt.nms_iou_threshold = YOLOV8_NMS_IOU
    opt.score_threshold = cfg.min_score
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)

def sample_tensors(sample) -> list:
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for field in sample.fields:
        tensors.extend(sample_tensors(field))
    return tensors

def decode_detections(sample) -> Detections:
    """Reads the decoder's oriented boxes, already mapped to source-frame pixels."""
    tensors = sample_tensors(sample)
    if not tensors:
        raise RuntimeError("detection sample carried no tensors")
    result = pyneat.decode_obb(tensors)[0]
    # float64, so that rounded values stay short in JSON (float32 0.9 prints 0.899999976).
    return Detections(
        boxes=result.geometry.to_numpy().astype(np.float64).reshape(-1, 5),
        scores=result.scores.to_numpy().astype(np.float64),
        class_ids=result.class_ids.to_numpy().astype(np.int64),
    )

def label_for(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"

def insight_objects(detections: Detections, labels: list[str]) -> list[dict]:
    """Builds one frame's Insight `object-detection` entries.

    `oriented_bbox` is what Insight draws. `bbox` is the box's upright extent, which
    places the label and is what viewers without oriented-box support draw instead.
    Values are rounded so that a frame with hundreds of boxes stays under the
    64 KB limit of one metadata message.
    """
    cx, cy, width, height, angle = detections.boxes.T
    cos, sin = np.abs(np.cos(angle)), np.abs(np.sin(angle))
    extent_w = width * cos + height * sin
    extent_h = width * sin + height * cos
    bboxes = np.column_stack([cx - extent_w / 2, cy - extent_h / 2, extent_w, extent_h])
    oriented_bboxes = np.column_stack([detections.boxes[:, :4].round(1), angle.round(4)])
    rows = zip(
        bboxes.round(1).tolist(),
        oriented_bboxes.tolist(),
        detections.scores.round(3).tolist(),
        detections.class_ids.tolist(),
    )
    return [
        {
            "id": f"obj_{index}",
            "label": label_for(labels, class_id),
            "confidence": confidence,
            "bbox": bbox,
            "oriented_bbox": oriented_bbox,
        }
        for index, (bbox, oriented_bbox, confidence, class_id) in enumerate(rows, start=1)
    ]
