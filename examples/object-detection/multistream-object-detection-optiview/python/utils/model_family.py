"""Model-family helpers for the Python multistream object detection example."""

from __future__ import annotations

from enum import Enum


class ModelFamily(Enum):
    AUTO = "auto"
    YOLOV8 = "yolov8"


def to_string(family: ModelFamily) -> str:
    return family.value


def resolve_model_family(
    model_path: str, hint: ModelFamily = ModelFamily.AUTO
) -> ModelFamily:
    if hint is not ModelFamily.AUTO:
        return hint

    lowered = str(model_path).lower()
    if "yolo_v8" in lowered or "yolov8" in lowered:
        return ModelFamily.YOLOV8
    raise ValueError(f"unable to infer model family from model path: {model_path}")
