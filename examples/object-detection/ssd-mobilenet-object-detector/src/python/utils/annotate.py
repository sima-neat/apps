"""The annotated images and the JSON detections report. Mirrors src/cpp/utils/annotate.h."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def output_name(image_path: Path) -> str:
    """The annotated PNG's file name. It keeps the source extension, so frame.jpg and frame.png
    do not write the same file."""
    return f"{image_path.stem}_{image_path.suffix[1:]}.png"


def draw_detections(image: np.ndarray, boxes: list, labels: list[str]) -> np.ndarray:
    """A copy of `image` with each box, its label, and its score drawn on it."""
    annotated = image.copy()
    for box in boxes:
        color = BOX_COLORS[box.class_id % len(BOX_COLORS)]
        # The decoder clamps boxes to the image, so the coordinates are not negative.
        x1, y1 = int(box.x1 + 0.5), int(box.y1 + 0.5)
        x2, y2 = int(box.x2 + 0.5), int(box.y2 + 0.5)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{labels[box.class_id]} {box.score:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def detections_record(
    image_path: Path, image: np.ndarray, boxes: list, labels: list[str]
) -> dict[str, Any]:
    """One image's entry in the report: its name, size, and every box with its label."""
    height, width = image.shape[:2]
    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "detections": [
            {
                "class_id": box.class_id,
                "label": labels[box.class_id],
                "score": box.score,
                "box": [box.x1, box.y1, box.x2, box.y2],
            }
            for box in boxes
        ],
    }


def write_report(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"images": records}, indent=2) + "\n", encoding="utf-8")
