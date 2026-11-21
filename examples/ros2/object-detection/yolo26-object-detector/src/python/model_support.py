"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import struct

import sys

import time

from pathlib import Path

import yaml

MIN_SCORE = 0.25

NMS_IOU = 0.45

MAX_DET = 100

DEFAULT_CONFIG = Path(get_package_share_directory('neat_yolo26_object_detector')) / "common" / "config.yaml"

BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]

def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels

def extract_bbox_payload_from_tensors(tensors) -> bytes | None:
    if len(tensors) != 1:
        return None
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None

def parse_bbox_payload(payload: bytes, img_w: int, img_h: int, min_score: float) -> list[dict]:
    if len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    out = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1 or float(score) < min_score:
            continue
        out.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score), class_id=int(cls_id)))
    return out

def draw_boxes(frame: np.ndarray, boxes: list[dict], labels: list[str]) -> np.ndarray:
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        cls_id = b["class_id"]
        score = b["score"]
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text = f"{label} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame
