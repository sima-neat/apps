"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import glob

import json

import math

import os

import struct

import sys

import time

from collections import deque

from dataclasses import dataclass

from pathlib import Path

import yaml

from utils.tracker import ObjectTracker, TrackedDetection, TrackerConfig

DEFAULT_CONFIG = Path(get_package_share_directory('neat_yolo26_tiny_drone_tracker')) / "common" / "config.yaml"

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
    max_inflight_total: int = 4
    num_classes: int = 1
    target_class_id: int = 0
    target_label: str = "drone"
    min_score: float = 0.05
    nms_iou: float = 0.60
    max_detections: int = 100
    profile: bool = False
    warmup_frames: int = 30
    tracker_high_score: float = 0.20
    tracker_new_track_score: float = 0.30
    tracker_iou_threshold: float = 0.05
    tracker_max_center_distance: float = 3.0
    tracker_velocity_momentum: float = 0.75
    tracker_max_missing: int = 30
    tracker_min_confirmed_hits: int = 2
    tracker_max_active: int = 256
    tracker_center_distance_enabled: bool = True
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
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
    if cfg.num_classes <= 0:
        raise ValueError("inference.num_classes must be > 0")
    if cfg.target_class_id < 0:
        raise ValueError("inference.target_class_id must be >= 0")
    if cfg.target_class_id >= cfg.num_classes:
        raise ValueError(
            f"inference.target_class_id ({cfg.target_class_id}) must be less than "
            f"inference.num_classes ({cfg.num_classes})"
        )
    if not cfg.target_label.strip():
        raise ValueError("inference.target_label must be set")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if not math.isfinite(cfg.tracker_iou_threshold) or not (
        0.0 <= cfg.tracker_iou_threshold <= 1.0
    ):
        raise ValueError("tracking.match_iou_threshold must be between 0 and 1")
    if not math.isfinite(cfg.tracker_high_score) or not (
        cfg.min_score <= cfg.tracker_high_score <= 1.0
    ):
        raise ValueError(
            "tracking.high_score_threshold must be in [inference.min_score, 1]"
        )
    if not math.isfinite(cfg.tracker_new_track_score) or not (
        cfg.tracker_high_score <= cfg.tracker_new_track_score <= 1.0
    ):
        raise ValueError(
            "tracking.new_track_threshold must be in [high_score_threshold, 1]"
        )
    if (
        not math.isfinite(cfg.tracker_max_center_distance)
        or cfg.tracker_max_center_distance < 0.0
    ):
        raise ValueError("tracking.max_center_distance must be >= 0")
    if not math.isfinite(cfg.tracker_velocity_momentum) or not (
        0.0 <= cfg.tracker_velocity_momentum < 1.0
    ):
        raise ValueError("tracking.velocity_momentum must be in [0, 1)")
    if cfg.tracker_max_missing < 0:
        raise ValueError("tracking.max_missing_frames must be >= 0")
    if cfg.tracker_min_confirmed_hits < 1:
        raise ValueError("tracking.min_confirmed_hits must be >= 1")
    if cfg.tracker_max_active < 1:
        raise ValueError("tracking.max_active_tracks must be >= 1")
    max_port_base = 65535 - (len(cfg.rtsp_urls) - 1)
    port_range = (
        f"must be between 1 and {max_port_base} for {len(cfg.rtsp_urls)} streams"
    )
    if not 1 <= cfg.video_port_base <= max_port_base:
        raise ValueError(f"output.insight.video_port_base {port_range}")
    if not 1 <= cfg.metadata_port_base <= max_port_base:
        raise ValueError(f"output.insight.metadata_port_base {port_range}")
    if cfg.video_enabled:
        last_port_offset = len(cfg.rtsp_urls) - 1
        video_port_end = cfg.video_port_base + last_port_offset
        metadata_port_end = cfg.metadata_port_base + last_port_offset
        if not (
            video_port_end < cfg.metadata_port_base
            or metadata_port_end < cfg.video_port_base
        ):
            raise ValueError(
                "output.insight video and metadata port ranges must not overlap "
                "when output.video_enabled is true"
            )
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

    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    rtsp_urls = []
    for index, value in enumerate(streams):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(value)

    min_score = float_or(inference, "min_score", 0.05)
    tracker_high_score = float_or(tracking, "high_score_threshold", min_score)
    tracker_new_track_score = float_or(
        tracking, "new_track_threshold", tracker_high_score
    )

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        rtsp_urls=rtsp_urls,
        codec=parse_input_codec(string_or(input_cfg, "codec", "h264")),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(inference, "max_inflight_total", 4),
        num_classes=int_or(inference, "num_classes", 1),
        target_class_id=int_or(inference, "target_class_id", 0),
        target_label=string_or(inference, "target_label", "drone"),
        min_score=min_score,
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 100),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        tracker_high_score=tracker_high_score,
        tracker_new_track_score=tracker_new_track_score,
        tracker_iou_threshold=float_or(tracking, "match_iou_threshold", 0.05),
        tracker_max_center_distance=float_or(tracking, "max_center_distance", 3.0),
        tracker_velocity_momentum=float_or(tracking, "velocity_momentum", 0.75),
        tracker_max_missing=int_or(tracking, "max_missing_frames", 30),
        tracker_min_confirmed_hits=int_or(tracking, "min_confirmed_hits", 2),
        tracker_max_active=int_or(tracking, "max_active_tracks", 256),
        tracker_center_distance_enabled=bool_or(
            tracking, "center_distance_enabled", True
        ),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
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

def parse_boxes_strict(
    payload: bytes, img_w: int, img_h: int, expected_topk: int
) -> list[dict]:
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

def filter_target_class(boxes: list[dict], target_class_id: int) -> list[dict]:
    return [box for box in boxes if int(box["class_id"]) == target_class_id]

def build_metadata_tracks(
    tracks: list[TrackedDetection], frame_w: int, frame_h: int, target_label: str
) -> list[dict]:
    metadata_tracks = []
    for track in tracks:
        x = max(0, int(track.x1))
        y = max(0, int(track.y1))
        w = max(0, int(track.x2 - track.x1))
        h = max(0, int(track.y2 - track.y1))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        metadata_tracks.append(
            {
                "id": str(track.track_id),
                "label": target_label,
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
    opt.num_classes = cfg.num_classes
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
