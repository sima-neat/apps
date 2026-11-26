"""The annotated images' file names. Mirrors src/cpp/utils/annotate.h."""

from __future__ import annotations

from pathlib import Path


def output_name(image_path: Path) -> str:
    """The annotated PNG's file name."""
    return f"{image_path.stem}_detr.png"
