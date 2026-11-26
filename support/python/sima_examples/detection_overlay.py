"""Draw detection boxes, with their labels and scores, on an image.

Mirrors support/cpp/sima_examples/detection_overlay.h.
"""

from __future__ import annotations

import cv2
import numpy as np

# Box colors in BGR, one per class, repeating every eight classes.
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


def draw_detections(image: np.ndarray, boxes: list, labels: list[str]) -> np.ndarray:
    """A copy of `image` with each box, its label, and its score drawn on it.

    A box is any object with x1, y1, x2, y2, score, and class_id, such as Neat's.
    """
    annotated = image.copy()
    for box in boxes:
        color = BOX_COLORS[box.class_id % len(BOX_COLORS)]
        # The decoder clamps boxes to the image, so the coordinates are not negative.
        x1, y1 = int(box.x1 + 0.5), int(box.y1 + 0.5)
        x2, y2 = int(box.x2 + 0.5), int(box.y2 + 0.5)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{labels[box.class_id]} {box.score:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated
