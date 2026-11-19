"""RF-DETR postprocessing: the proposals the transformer reads and the metadata Insight draws.

Mirrors utils/postprocess.h in the C++ implementation, so both send the same metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

NUM_CLASSES = 91  # COCO class IDs 0 to 90. ID 0 and the "unused" labels name no object.
NUM_SELECT = (
    300  # RF-DETR's reference postprocess ranks the 300 best (query, class) scores.
)
# The metadata sender rejects messages over 65,507 bytes; 32 KiB of segments keeps the whole
# message, envelope included, under that limit.
SEGMENTS_BYTE_BUDGET = 32_768


@dataclass(frozen=True)
class PostprocessOptions:
    width: int  # Frame size the boxes and polygons are scaled to, in pixels.
    height: int
    min_score: float
    max_detections: int
    mask_threshold: float  # Segmentation only.
    mask_grid_size: int  # Segmentation only.


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Indices of the `count` largest values, largest first; equal values keep index order."""
    return np.argsort(-values, kind="stable")[:count]


def top_proposals(scores: np.ndarray, proposals: np.ndarray, count: int) -> np.ndarray:
    """The `count` proposal boxes with the highest scores, best first, as a (count, 4) array."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    proposals = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
    return proposals[top_indices(scores, count)]


def sigmoid(values: np.ndarray) -> np.ndarray:
    # Clamping keeps exp() finite in float32.
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def ranked_objects(
    logits: np.ndarray, labels: list[str], min_score: float
) -> Iterator[tuple[int, str, float]]:
    """(query, label, score) of each object scoring at least `min_score`, best first."""
    scores = sigmoid(np.asarray(logits, dtype=np.float32)).reshape(-1)
    for index in top_indices(scores, NUM_SELECT):
        score = float(scores[index])
        if score < min_score:
            return
        query, class_id = divmod(int(index), NUM_CLASSES)
        if class_id != 0 and labels[class_id] != "unused":
            yield query, labels[class_id], score


def detection_json(
    boxes: np.ndarray,
    logits: np.ndarray,
    labels: list[str],
    options: PostprocessOptions,
) -> str:
    """The "objects" metadata: each box in frame pixels as [x, y, width, height]."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    width, height = float(options.width), float(options.height)
    objects = []
    for query, label, score in ranked_objects(logits, labels, options.min_score):
        if len(objects) == options.max_detections:
            break
        cx, cy, box_width, box_height = (float(value) for value in boxes[query])
        x = max(0.0, min((cx - box_width / 2.0) * width, width))
        y = max(0.0, min((cy - box_height / 2.0) * height, height))
        x2 = max(x, min((cx + box_width / 2.0) * width, width))
        y2 = max(y, min((cy + box_height / 2.0) * height, height))
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": label,
                "confidence": score,
                "bbox": [x, y, x2 - x, y2 - y],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


def frame_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """A normalized (cx, cy, w, h) box as whole pixels (x, y, width, height) inside the frame."""
    cx, cy, box_width, box_height = (float(value) for value in box)
    x0 = max(0, min(width - 1, round((cx - box_width / 2.0) * width)))
    y0 = max(0, min(height - 1, round((cy - box_height / 2.0) * height)))
    x1 = max(x0 + 1, min(width, round((cx + box_width / 2.0) * width)))
    y1 = max(y0 + 1, min(height, round((cy + box_height / 2.0) * height)))
    return x0, y0, x1 - x0, y1 - y0


def mask_polygon(
    masks: np.ndarray,
    query: int,
    box: tuple[int, int, int, int],
    options: PostprocessOptions,
) -> list[list[int]]:
    """The outline of one query's mask inside its box, in frame pixels; empty when none."""
    mask_size = masks.shape[0]
    x, y, box_width, box_height = box
    mx0 = max(0, min(mask_size - 1, int(np.floor(x * mask_size / options.width))))
    my0 = max(0, min(mask_size - 1, int(np.floor(y * mask_size / options.height))))
    mx1 = max(
        mx0 + 1,
        min(mask_size, int(np.ceil((x + box_width) * mask_size / options.width))),
    )
    my1 = max(
        my0 + 1,
        min(mask_size, int(np.ceil((y + box_height) * mask_size / options.height))),
    )
    mask = sigmoid(masks[my0:my1, mx0:mx1, query])
    if options.mask_grid_size != mask_size:
        size = tuple(
            (n * options.mask_grid_size + mask_size - 1) // mask_size
            for n in mask.shape[::-1]
        )
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
    binary = (mask >= options.mask_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)
    if len(polygon) < 3:
        return []
    scale_x = (box_width - 1) / max(binary.shape[1] - 1, 1)
    scale_y = (box_height - 1) / max(binary.shape[0] - 1, 1)
    return [
        [
            min(x + box_width - 1, x + round(int(point[0][0]) * scale_x)),
            min(y + box_height - 1, y + round(int(point[0][1]) * scale_y)),
        ]
        for point in polygon
    ]


def segmentation_json(
    boxes: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray,
    mask_size: int,
    labels: list[str],
    options: PostprocessOptions,
) -> str:
    """The "segments" metadata: each object's box and its outline as a polygon, in frame pixels."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    masks = np.asarray(masks, dtype=np.float32).reshape(mask_size, mask_size, -1)
    segments = []
    size = len('{"segments":[]}')
    for query, label, score in ranked_objects(logits, labels, options.min_score):
        if len(segments) == options.max_detections:
            break
        box = frame_box(boxes[query], options.width, options.height)
        polygon = mask_polygon(masks, query, box, options)
        if not polygon:
            continue
        segment = {
            "id": f"seg_{len(segments) + 1}",
            "label": label,
            "confidence": score,
            "bbox": list(box),
            "mask_format": "polygon",
            "mask": polygon,
        }
        segment_size = len(json.dumps(segment, separators=(",", ":"))) + 1
        if size + segment_size > SEGMENTS_BYTE_BUDGET:
            break
        size += segment_size
        segments.append(segment)
    return json.dumps({"segments": segments}, separators=(",", ":"))
