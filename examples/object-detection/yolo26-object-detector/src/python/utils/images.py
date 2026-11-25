"""The input images: the files in the folder, and reading one. Mirrors src/cpp/utils/images.h."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def image_paths(folder: Path) -> list[Path]:
    """The images directly in `folder`, in file-name order."""
    if not folder.is_dir():
        raise ValueError(f"input.source is not a folder: {folder}")
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"input.source has no .jpg, .jpeg, .png, or .bmp files: {folder}")
    return paths


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image
