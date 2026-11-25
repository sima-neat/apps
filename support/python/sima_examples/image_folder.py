"""List the images in a folder, for the examples that read a folder of image files.

Mirrors support/cpp/sima_examples/image_folder.h.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


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
