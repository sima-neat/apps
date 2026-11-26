"""Helpers for the examples that read a folder of image files and save annotated copies: list the
images, read one, name the annotated PNG, and draw the detections on it.

Mirrors support/cpp/sima_examples/image_folder.h.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

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


def image_paths(folder: Path) -> list[Path]:
    """The .jpg, .jpeg, .png, and .bmp files directly in `folder`, in file-name order.

    A suffix matches in any case, such as .PNG. Subfolders and their files are skipped.
    """
    if not folder.is_dir():
        raise ValueError(f"image folder does not exist: {folder}")
    paths = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError(f"no .jpg, .jpeg, .png, or .bmp files in {folder}")
    return paths


def read_image(path: Path) -> np.ndarray:
    """The image at `path` as a BGR array. Raises ValueError when OpenCV cannot read it."""
    import cv2  # Imported here so listing images needs no OpenCV.

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def output_name(image_path: Path, suffix: str) -> str:
    """The annotated PNG's file name: the image's stem, then `suffix`, then .png.

    An example picks the suffix that marks its results, such as "_detr".
    """
    return f"{image_path.stem}{suffix}.png"


def draw_detections(image: np.ndarray, boxes: list, labels: list[str]) -> np.ndarray:
    """A copy of `image` with each box, its label, and its score drawn on it.

    A box is any object with x1, y1, x2, y2, score, and class_id, such as Neat's.
    """
    import cv2  # Imported here so listing images needs no OpenCV.

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
