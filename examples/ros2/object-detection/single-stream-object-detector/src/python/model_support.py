"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass

from fractions import Fraction

import glob

import json

import os

from pathlib import Path

import struct

import subprocess

import sys

import time

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_single_stream_object_detector')) / "common" / "config.yaml"

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    source_url: str
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
    save_dir: str = ""
    save_every: int = 0
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
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value

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
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")

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
    default_labels = Path(get_package_share_directory('neat_single_stream_object_detector')) / "common" / "coco_label.txt"

    labels_path = string_or(model, "labels", str(default_labels))
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
        save_dir=string_or(output, "save_dir"),
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
    try:
        payload = tensor.copy_payload_bytes()
    except Exception as exc:
        raise RuntimeError(f"capture_payload_failed err={exc}") from exc
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

def build_metadata_boxes(boxes: list[dict], labels: list[str], frame_w: int, frame_h: int) -> list[dict]:
    metadata_boxes = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, int(box["x2"] - box["x1"]))
        h = max(0, int(box["y2"] - box["y1"]))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        class_id = int(box["class_id"])
        metadata_boxes.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(box["score"]),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_boxes

def make_model(cfg: AppConfig, frame_w: int, frame_h: int, nv16_frames: bool = False):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16 if nv16_frames else pyneat.PreprocessColorFormat.NV12
    )
    if frame_w > 0 and frame_h > 0:
        opt.preprocess.input_max_width = frame_w
        opt.preprocess.input_max_height = frame_h
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)
