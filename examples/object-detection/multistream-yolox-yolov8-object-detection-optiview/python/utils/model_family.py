"""Model-family helpers for the multistream YOLOv8 example."""

from __future__ import annotations


VALID_MODEL_FAMILIES = {"auto", "yolov8"}
YOLOX_NOT_SUPPORTED_MESSAGE = (
    "YOLOX model packs are not supported yet by this example; future support is planned."
)


def parse_model_family(value: str) -> str:
    family = value.strip().lower()
    if family == "yolox":
        raise ValueError(YOLOX_NOT_SUPPORTED_MESSAGE)
    if family not in VALID_MODEL_FAMILIES:
        raise ValueError(f"model.family must be one of {sorted(VALID_MODEL_FAMILIES)}")
    return family


def resolve_model_family(model_path: str, family_hint: str = "auto") -> str:
    family = parse_model_family(family_hint)
    if family != "auto":
        return family

    lowered = model_path.lower()
    if "yolox" in lowered:
        raise ValueError(YOLOX_NOT_SUPPORTED_MESSAGE)
    if "yolo_v8" in lowered or "yolov8" in lowered:
        return "yolov8"
    raise ValueError(f"unable to infer model family from model path: {model_path}")
