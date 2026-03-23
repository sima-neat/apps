"""Model-family helpers for the multistream YOLOX/YOLOv8 example."""

from __future__ import annotations


VALID_MODEL_FAMILIES = {"auto", "yolox", "yolov8"}


def parse_model_family(value: str) -> str:
    family = value.strip().lower()
    if family not in VALID_MODEL_FAMILIES:
        raise ValueError(f"model.family must be one of {sorted(VALID_MODEL_FAMILIES)}")
    return family


def resolve_model_family(model_path: str, family_hint: str = "auto") -> str:
    family = parse_model_family(family_hint)
    if family != "auto":
        return family

    lowered = model_path.lower()
    if "yolox" in lowered:
        return "yolox"
    if "yolo_v8" in lowered or "yolov8" in lowered:
        return "yolov8"
    raise ValueError(f"unable to infer model family from model path: {model_path}")
