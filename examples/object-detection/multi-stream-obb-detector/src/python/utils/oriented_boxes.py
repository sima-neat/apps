"""The Insight object-detection data for one frame's oriented boxes.

Mirrors src/cpp/utils/oriented_boxes.h, so both implementations send the same JSON.
"""

from __future__ import annotations

import json

import numpy as np

UNKNOWN_LABEL = "unknown"


def detections_json(
    geometry: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, labels: list[str]
) -> str:
    """The `data` of one frame's `object-detection` message, as `{"objects": [...]}`.

    `geometry` has one row per box: cx, cy, width, height, and the angle in radians, in
    source-frame pixels. A positive angle turns the box clockwise. `oriented_bbox` is what
    Insight draws. `bbox` is the box's upright extent, which places the label. Pixels are
    rounded to 0.1, angles to 4 decimals, and scores to 3, so that a frame with hundreds of
    boxes stays under the 65507-byte limit of one metadata message.
    """
    # float64, so that rounded values stay short in JSON (float32 0.9 prints 0.899999976).
    geometry = geometry.astype(np.float64).reshape(-1, 5)
    cx, cy, width, height, angle = geometry.T
    cos, sin = np.abs(np.cos(angle)), np.abs(np.sin(angle))
    extent_w = width * cos + height * sin
    extent_h = width * sin + height * cos
    bboxes = np.column_stack([cx - extent_w / 2, cy - extent_h / 2, extent_w, extent_h])
    oriented = np.column_stack([geometry[:, :4].round(1), angle.round(4)])
    rows = zip(
        bboxes.round(1).tolist(),
        oriented.tolist(),
        scores.astype(np.float64).round(3).tolist(),
        class_ids.tolist(),
    )
    objects = [
        {
            "id": f"obj_{index}",
            "label": labels[class_id] if 0 <= class_id < len(labels) else UNKNOWN_LABEL,
            "confidence": confidence,
            "bbox": bbox,
            "oriented_bbox": oriented_bbox,
        }
        for index, (bbox, oriented_bbox, confidence, class_id) in enumerate(rows, start=1)
    ]
    return json.dumps({"objects": objects}, separators=(",", ":"))
