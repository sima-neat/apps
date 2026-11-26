"""Read an image file, for the examples that read images from disk.

Mirrors support/cpp/sima_examples/image_io.h.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    """The image at `path` as a BGR array. Raises ValueError when OpenCV cannot read it."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image
