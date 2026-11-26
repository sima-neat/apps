"""The Insight metadata entries of one frame: objects, segments, or poses in frame pixels.

Mirrors src/cpp/utils/insight_metadata.h, so both implementations send the same JSON.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

UNKNOWN_LABEL = "unknown"
# MetadataSender rejects a message above 65507 bytes. Entries beyond this budget are left out,
# lowest confidence first, so a crowded frame still sends its best entries.
METADATA_BYTE_BUDGET = 32768
# A mask cell belongs to the object when its value is at least half of full scale (255).
MASK_THRESHOLD = 0.5
# One mask cell spans four pixels of the model's letterboxed input: the BoxDecode masks are a
# 160x160 grid, for a 640x640 model input.
MASK_STRIDE = 4
# The order in which the BoxDecode pose payload emits the 17 COCO keypoints. Insight joins the
# skeleton by these names.
COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def round_confidence(value: float) -> float:
    return math.floor(float(value) * 1000.0 + 0.5) / 1000.0


def round_pixel(value: float) -> int:
    """Rounds half away from zero, as C++ std::lround does."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def label(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else UNKNOWN_LABEL


def objects(boxes, labels: list[str], frame_w: int, frame_h: int) -> list[dict]:
    """The `objects` entries of `boxes`, `pyneat.detections.Box` records in frame pixels."""
    entries = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box.x1))
        y = max(0, int(box.y1))
        width = max(0, min(int(box.x2 - box.x1), frame_w - x))
        height = max(0, min(int(box.y2 - box.y1), frame_h - y))
        entries.append(
            {
                "id": f"obj_{index}",
                "label": label(labels, int(box.class_id)),
                "confidence": round_confidence(box.score),
                "bbox": [float(x), float(y), float(width), float(height)],
            }
        )
    return entries


def frame_rect(box: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """The whole pixels `box` (x1, y1, x2, y2) covers, at least one, inside the frame."""
    x0 = max(0, min(frame_w - 1, math.floor(box[0])))
    y0 = max(0, min(frame_h - 1, math.floor(box[1])))
    x1 = max(x0 + 1, min(frame_w, math.ceil(box[2])))
    y1 = max(y0 + 1, min(frame_h, math.ceil(box[3])))
    return x0, y0, x1, y1


def mask_rect(
    rect: tuple[int, int, int, int], frame_w: int, frame_h: int, mask_w: int, mask_h: int
) -> tuple[int, int, int, int]:
    """The mask cells under the frame rectangle `rect`, through the model's letterbox."""
    model_w = mask_w * MASK_STRIDE
    model_h = mask_h * MASK_STRIDE
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) * 0.5
    pad_y = (model_h - frame_h * scale) * 0.5

    def cell_x(x: float) -> float:
        return (x * scale + pad_x) / MASK_STRIDE

    def cell_y(y: float) -> float:
        return (y * scale + pad_y) / MASK_STRIDE

    x0 = max(0, min(mask_w - 1, math.floor(cell_x(rect[0]))))
    y0 = max(0, min(mask_h - 1, math.floor(cell_y(rect[1]))))
    x1 = max(x0 + 1, min(mask_w, math.ceil(cell_x(rect[2]))))
    y1 = max(y0 + 1, min(mask_h, math.ceil(cell_y(rect[3]))))
    return x0, y0, x1, y1


def mask_polygon(
    mask: np.ndarray, rect: tuple[int, int, int, int], frame_w: int, frame_h: int
) -> list[list[int]]:
    """The outline, in frame pixels, of the largest region of `mask` inside `rect`, or empty
    when it has fewer than three corners."""
    x0, y0, x1, y1 = rect
    mx0, my0, mx1, my1 = mask_rect(rect, frame_w, frame_h, mask.shape[1], mask.shape[0])
    cells = np.ascontiguousarray(mask[my0:my1, mx0:mx1])
    roi = cv2.resize(cells, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(roi, MASK_THRESHOLD * 255.0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    return [[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in polygon]


def segments(
    boxes: np.ndarray, masks: np.ndarray, labels: list[str], frame_w: int, frame_h: int
) -> list[dict]:
    """The `segments` entries of `boxes` [N, 6] (x1, y1, x2, y2, score, class_id) in frame
    pixels and their `masks` [N, 160, 160], each with its outline as a polygon."""
    if len(boxes) != len(masks):
        raise RuntimeError(
            f"segmentation decode returned {len(boxes)} boxes but {len(masks)} masks"
        )
    entries = []
    for box, mask in zip(boxes, masks):
        rect = frame_rect(box, frame_w, frame_h)
        polygon = mask_polygon(mask, rect, frame_w, frame_h)
        if not polygon:
            continue
        entries.append(
            {
                "id": f"seg_{len(entries) + 1}",
                "label": label(labels, int(box[5])),
                "confidence": round_confidence(box[4]),
                "bbox": [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]],
                "mask_format": "polygon",
                "mask": polygon,
            }
        )
    return entries


def poses(boxes: np.ndarray, keypoints: np.ndarray) -> list[dict]:
    """The `poses` entries of `boxes` [N, 6] in frame pixels and their `keypoints` [N, 17, 3]
    (x, y, visibility)."""
    if len(boxes) != len(keypoints):
        raise RuntimeError(
            f"pose decode returned {len(boxes)} boxes but {len(keypoints)} keypoint sets"
        )
    entries = []
    for index, (box, points) in enumerate(zip(boxes, keypoints), start=1):
        entries.append(
            {
                "id": f"pose_{index}",
                "label": "person",
                "confidence": round_confidence(box[4]),
                "bbox": [
                    round_pixel(box[0]),
                    round_pixel(box[1]),
                    round_pixel(max(0.0, box[2] - box[0])),
                    round_pixel(max(0.0, box[3] - box[1])),
                ],
                "keypoints": [
                    {
                        "name": name,
                        "x": round_pixel(x),
                        "y": round_pixel(y),
                        "confidence": round_confidence(visibility),
                    }
                    for name, (x, y, visibility) in zip(COCO_KEYPOINT_NAMES, points)
                ],
            }
        )
    return entries


def data_json(key: str, entries: list[dict]) -> tuple[str, int]:
    """The `data` of one message, `{key: [...]}`, with the entries that fit
    `METADATA_BYTE_BUDGET`, highest confidence first, and how many were left out."""
    kept = []
    size = len(json.dumps({key: []}, separators=(",", ":")))
    # Stable, so entries with the same confidence keep their order, as in C++.
    for entry in sorted(entries, key=lambda entry: entry["confidence"], reverse=True):
        entry_size = len(json.dumps(entry, separators=(",", ":"))) + 1
        if size + entry_size > METADATA_BYTE_BUDGET:
            break
        size += entry_size
        kept.append(entry)
    return json.dumps({key: kept}, separators=(",", ":")), len(entries) - len(kept)
