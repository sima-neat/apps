"""Draw defect boxes and their labels on a BGR image. Mirrors src/cpp/utils/overlay.h."""

from __future__ import annotations

import math

import cv2
import numpy as np

# BGR, one per class in pcb_label.txt order.
CLASS_COLORS = [
    (56, 56, 255),
    (29, 178, 255),
    (10, 249, 72),
    (255, 194, 0),
    (255, 0, 200),
    (49, 210, 207),
]


def pixel(value: float, limit: int) -> int:
    """A box coordinate as a pixel index from 0 to limit - 1, rounded half up like C++ std::round
    (Python's round() rounds half to even)."""
    return min(max(math.floor(value + 0.5), 0), limit - 1)


def draw_boxes(image: np.ndarray, boxes: list, labels: list[str]) -> None:
    """Draw each box in its class's color, with its label and score, on `image` in place. A box
    with no width or height inside the image is skipped."""
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1 = pixel(box.x1, width), pixel(box.y1, height)
        x2, y2 = pixel(box.x2, width), pixel(box.y2, height)
        if x2 <= x1 or y2 <= y1:
            continue
        color = CLASS_COLORS[box.class_id % len(CLASS_COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{labels[box.class_id]} {box.score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(0, y1 - text_height - 4)
        cv2.rectangle(image, (x1, label_top), (x1 + text_width + 2, y1), color, -1)
        cv2.putText(image, text, (x1 + 1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
