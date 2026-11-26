"""The annotated images' file names and the JSON detections report. Mirrors
src/cpp/utils/annotate.h.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sima_examples import image_folder


def output_name(image_path: Path) -> str:
    """The annotated PNG's file name. It keeps the source extension, so frame.jpg and frame.png
    do not write the same file."""
    return image_folder.output_name(image_path, f"_{image_path.suffix[1:]}")


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
