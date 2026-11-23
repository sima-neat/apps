"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import json

import logging

import math

import sys

from collections.abc import Mapping

from decimal import Decimal, InvalidOperation

from pathlib import Path

from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(get_package_share_directory('neat_ssd_mobilenet_object_detector')) / "common" / "config.yaml"

COMMON_DIR = Path(get_package_share_directory('neat_ssd_mobilenet_object_detector')) / "common"

DEFAULT_LABELS_PATH = Path(
    "examples/object-detection/ssd-mobilenet-object-detector/src/common/coco_labels.txt"
)

DEFAULT_PREPROCESSING_PROFILE = "tensorflow_ssd"

NUM_CLASSES = 91  # index 0 = background, 1..90 = COCO ids.

BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]

def _resolve_asset(configured: str, default_name: str) -> Path:
    """Resolve a labels asset: the configured path if present. Substitute the packaged
    src/common copy only for the empty/default reference, not a missing custom path."""
    candidate = Path(configured)
    if candidate.is_file():
        return candidate
    if not configured or candidate == DEFAULT_LABELS_PATH:
        fallback = COMMON_DIR / default_name
        if fallback.is_file():
            return fallback
    return candidate

def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    labels = [
        line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels

def class_name(labels: list[str], class_idx: int) -> str:
    if 0 <= class_idx < len(labels):
        name = labels[class_idx]
        if name and name != "N/A":
            return name
    return f"class_{class_idx}"

def class_color(class_idx: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_idx) % len(BOX_COLORS)]

def normalization_for_profile(profile: str) -> tuple[list[float], list[float]]:
    if profile == "tensorflow_ssd":
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if profile == "torchvision_ssdlite":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(
        "model.preprocessing_profile must be tensorflow_ssd or torchvision_ssdlite"
    )

def image_to_tensor(image_bgr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(image_bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )

def decode_detections(
    tensors, image_width: int, image_height: int, max_detections: int
) -> list[dict[str, Any]]:
    """Decode the single BBOX tensor through Core's public detection API."""
    if len(tensors) != 1:
        raise RuntimeError(f"expected one BBOX tensor, got {len(tensors)}")
    result = pyneat.detections.decode_bbox_tensor(
        tensors[0], image_width, image_height, max_detections, False
    )
    return [
        {
            "box": [box.x1, box.y1, box.x2, box.y2],
            "score": float(box.score),
            "class_id": int(box.class_id),
        }
        for box in result.boxes
    ]

def visualize_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    labels: list[str],
) -> np.ndarray:
    image_copy = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (round(v) for v in det["box"])
        color = class_color(det["class_id"])
        text = f"{class_name(labels, det['class_id'])} {det['score']:.2f}"
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_copy,
            text,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return image_copy
