"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import glob

import json

import math

import os

import sys

import time

from dataclasses import dataclass

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_multi_stream_pose_estimator')) / "common" / "config.yaml"

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
    input_max_width: int = 1920
    input_max_height: int = 1080
    frames: int = 0
    max_inflight_per_stream: int = 4
    max_inflight_total: int = 16
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_poses: int = 50
    min_keypoint_visibility: float = 0.30
    profile: bool = False
    warmup_frames: int = 30
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
        raise ValueError(f"{key} must be a mapping")  # noqa: TRY004
    return value

def string_or(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")  # noqa: TRY004
    return value

def int_or(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")  # noqa: TRY004
    return int(value)

def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")  # noqa: TRY004
    return float(value)

def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")  # noqa: TRY004
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
    if cfg.input_max_width <= 0:
        raise ValueError("input.max_width must be > 0")
    if cfg.input_max_height <= 0:
        raise ValueError("input.max_height must be > 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.max_inflight_per_stream != -1 and cfg.max_inflight_per_stream <= 0:
        raise ValueError("inference.max_inflight_per_stream must be -1 or > 0")
    if cfg.max_inflight_total != -1 and cfg.max_inflight_total <= 0:
        raise ValueError("inference.max_inflight_total must be -1 or > 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_poses <= 0:
        raise ValueError("inference.max_poses must be > 0")
    if not 0.0 <= cfg.min_keypoint_visibility <= 1.0:
        raise ValueError("output.min_keypoint_visibility must be between 0 and 1")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")

def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")  # noqa: TRY004

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
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

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        rtsp_urls=rtsp_urls,
        codec=parse_input_codec(string_or(input_cfg, "codec", "h264")),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        input_max_width=int_or(input_cfg, "max_width", 1920),
        input_max_height=int_or(input_cfg, "max_height", 1080),
        frames=int_or(inference, "frames", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(inference, "max_inflight_total", 16),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_poses=int_or(inference, "max_poses", 50),
        min_keypoint_visibility=float_or(output, "min_keypoint_visibility", 0.30),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg

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

def tensor_to_numpy(tensor) -> object:
    # A frame with no person decodes to a zero-row tensor. Copying one throws, because a
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
    for field in sample.fields:
        tensors.extend(extract_tensors(field))
    return tensors

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
        keypoints = tensor_to_numpy(item.keypoints).reshape((-1, 17, 3))
        if boxes.shape[0] != keypoints.shape[0]:
            raise RuntimeError(
                f"pose decode returned {boxes.shape[0]} boxes but {keypoints.shape[0]} keypoint sets"
            )
        for box, points in zip(boxes, keypoints):
            if len(poses) >= max_poses:
                return poses
            if round(float(box[5])) != 0:
                raise RuntimeError("pose decode returned a non-person class id")
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

def pose_metadata_data(poses: list[dict]) -> dict:
    """Build the ``data`` object Insight's ``pose-estimation`` overlay consumes.

    Pixel rounding and three-decimal confidence preserve overlay precision while keeping the
    configured 50-pose maximum within Core's logical metadata-message limit.

    Poses carry no ``id``: Insight colors poses by ``id``, and this app does not track people
    across frames, so a per-frame index would change a person's color.
    """
    def round_pixel(value: float) -> int:
        return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)

    def round_confidence(value: float) -> float:
        return math.floor(value * 1000.0 + 0.5) / 1000.0

    published = []
    for pose in poses:
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

def build_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.input_max_width = cfg.input_max_width
    opt.preprocess.input_max_height = cfg.input_max_height
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26Pose
    # YOLO26 pose ships single-class ("person") score heads. The packaged MPK still declares
    # the 80-class detector metadata, so state the real class count rather than inheriting it.
    opt.num_classes = 1
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_poses
    return pyneat.Model(cfg.model_path, opt)
