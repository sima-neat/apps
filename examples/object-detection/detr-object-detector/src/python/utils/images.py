"""The largest size among the input images. Mirrors src/cpp/utils/images.h."""

from __future__ import annotations

from pathlib import Path

from sima_examples.image_io import read_image


def largest_image_size(paths: list[Path]) -> tuple[int, int]:
    """The largest width and the largest height among the images, which may be different
    images."""
    width = height = 0
    for path in paths:
        image_height, image_width = read_image(path).shape[:2]
        width, height = max(width, image_width), max(height, image_height)
    return width, height
