"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass

from pathlib import Path

import glob

import json

import os

import struct

import sys

import time

import yaml

from utils.tracker import (
    ClassTrackerConfig,
    MultiClassTracker,
    TrackedDetection,
    load_labels,
    parse_class_configs,
)

DEFAULT_CONFIG = Path(get_package_share_directory('neat_multi_stream_tracker')) / "common" / "config.yaml"

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class AppConfig:
    model_path: str
    rtsp_urls: list[str]
    # Encoded RTSP path used for every stream: "h264" or "h265". Kept as a
    # string because load_app_config runs before the lazy pyneat import.
    codec: str = "h264"
    latency_ms: int = 100
    tcp: bool = True
    frames: int = 0
    fps: int = 0
    max_inflight_per_stream: int = 4
    max_inflight_total: int = 16
    min_score: float = 0.10
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    tracker_classes: tuple[ClassTrackerConfig, ...] = ()
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
    save_dir: str = ""
    save_every: int = 0
    detections_log: str = ""

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

def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not cfg.rtsp_urls:
        raise ValueError("streams must be set")
    if len(cfg.rtsp_urls) > 4:
        raise ValueError("this phase supports up to four streams")
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
    if cfg.max_inflight_total != -1 and cfg.max_inflight_total <= 0:
        raise ValueError("inference.max_inflight_total must be -1 or > 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if not cfg.tracker_classes:
        raise ValueError("tracking.classes must be set")
    for class_cfg in cfg.tracker_classes:
        if cfg.min_score > class_cfg.low_score_threshold:
            raise ValueError(
                f"inference.min_score must be <= low_score_threshold of class '{class_cfg.label}'"
            )
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
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
    tracking = section(raw, "tracking")
    output = section(raw, "output")
    insight = section(output, "insight")

    default_labels = Path(get_package_share_directory('neat_multi_stream_tracker')) / "common" / "coco_label.txt"
    labels = load_labels(Path(string_or(model, "labels", str(default_labels))).expanduser())

    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    rtsp_urls = []
    for index, value in enumerate(streams):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(value)

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        rtsp_urls=rtsp_urls,
        codec=parse_input_codec(string_or(input_cfg, "codec", "h264")),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(inference, "max_inflight_total", 16),
        min_score=float_or(inference, "min_score", 0.10),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        tracker_classes=parse_class_configs(tracking.get("classes"), labels),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
        detections_log=string_or(output, "detections_log"),
    )
    validate_config(cfg)
    return cfg

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
        for field in sample.fields:
            try:
                return extract_bbox_payload(field)
            except RuntimeError:
                continue
        raise RuntimeError("bundle missing BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return extract_tensor_bbox_payload(sample, sample.tensors[0])
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError("capture_expected_tensor")
    return extract_tensor_bbox_payload(sample)

def parse_boxes_strict(payload: bytes, img_w: int, img_h: int, expected_topk: int) -> list[dict]:
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

def build_metadata_tracks(
    tracks: list[TrackedDetection], frame_w: int, frame_h: int
) -> list[dict]:
    metadata_tracks = []
    for track in tracks:
        # Clip both endpoints before measuring: a predicted box can extend past
        # either edge, and clamping only the origin leaves the width too large.
        x1 = min(max(track.x1, 0.0), float(frame_w))
        y1 = min(max(track.y1, 0.0), float(frame_h))
        x2 = min(max(track.x2, 0.0), float(frame_w))
        y2 = min(max(track.y2, 0.0), float(frame_h))
        x = int(x1)
        y = int(y1)
        w = max(0, int(x2) - x)
        h = max(0, int(y2) - y)
        metadata_tracks.append(
            {
                "id": str(track.track_id),
                "label": track.label,
                "confidence": float(track.score),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_tracks

def build_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)

def build_run_options() -> pyneat.RunOptions:
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options

TRACK_COLORS = (
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (200, 130, 0), (48, 130, 245),
    (180, 30, 145), (240, 240, 70), (230, 50, 240), (60, 245, 210), (212, 190, 250),
)

IDLE_PULLS_BEFORE_CLOSE = 100
