"""Read a class-label file with one label per line. Mirrors support/cpp/sima_examples/labels.h."""

from __future__ import annotations

from pathlib import Path


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels
