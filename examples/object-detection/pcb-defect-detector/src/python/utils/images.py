"""Find the input images and check that the model can take each one. Mirrors
src/cpp/utils/images.h."""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_images(folder: Path) -> list[Path]:
    images = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"input.source: no .jpg, .jpeg, .png, or .bmp images in {folder}")
    return images


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
