"""Check that the model can take each image, and save the results.
Mirrors src/cpp/utils/images.h."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def image_problem(image: np.ndarray | None, max_size: tuple[int, int]) -> str:
    """Why the model cannot take `image`, or "" when it can."""
    if image is None:
        return "not a readable image"
    height, width = image.shape[:2]
    if width > max_size[0] or height > max_size[1]:
        return (
            f"{width}x{height} is larger than app.input_max_width x app.input_max_height "
            f"({max_size[0]}x{max_size[1]})"
        )
    return ""


def save_image(path: Path, image: np.ndarray) -> bool:
    """Write `image` to `path`; False when it could not be written, so one image does not stop
    the batch. OpenCV reports some failures by return value and raises for others, such as an
    encoder error."""
    try:
        return cv2.imwrite(str(path), image)
    except cv2.error:
        return False
