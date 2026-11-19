"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

import argparse

import sys

import time

from dataclasses import dataclass

from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_detr_object_detector')) / "common" / "config.yaml"

DEFAULT_MODEL_PATH = "models/detr_resnet_50/w16a16/detr_resnet_50_w16a16_mpk.tar.gz"

PERSON_CLASS_ID = 1

DETR_COCO_LABELS = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]

@dataclass(frozen=True)
class Config:
    model_path: Path
    input_dir: Path
    output_dir: Path
    confidence_threshold: float
    max_draw: int
    person_only: bool
    profile: bool
    num_runs: int

def load_config(path: Path) -> Config:
    """Read the YAML config and reject values Neat would read as "use the package default"."""
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_cfg = raw.get("model", {})
    io_cfg = raw.get("io", {})
    decode_cfg = raw.get("decode", {})
    runtime_cfg = raw.get("runtime", {})
    config = Config(
        model_path=Path(model_cfg.get("path", DEFAULT_MODEL_PATH)),
        input_dir=Path(io_cfg.get("input_dir", "assets/datasets/coco")),
        output_dir=Path(io_cfg.get("output_dir", "sandbox/detr-object-detector")),
        confidence_threshold=float(decode_cfg.get("confidence_threshold", 0.7)),
        max_draw=int(decode_cfg.get("max_draw", 50)),
        person_only=bool(decode_cfg.get("person_only", False)),
        profile=bool(runtime_cfg.get("profile", False)),
        num_runs=int(runtime_cfg.get("num_runs", 100)),
    )
    # Neat's decoder keeps the package's own threshold and top-k when either is 0.
    if not 0.0 < config.confidence_threshold <= 1.0:
        raise ValueError("decode.confidence_threshold must be in (0.0, 1.0]")
    if config.max_draw < 1:
        raise ValueError("decode.max_draw must be >= 1")
    if config.num_runs < 1:
        raise ValueError("runtime.num_runs must be >= 1")
    return config

def class_name(class_id: int) -> str:
    if 0 <= class_id < len(DETR_COCO_LABELS):
        label = DETR_COCO_LABELS[class_id]
        if label != "N/A":
            return label
    return f"class_{class_id}"

def class_color(class_id: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_id) % len(BOX_COLORS)]

def image_to_tensor(image_bgr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(image_bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image_bgr


def largest_image_size(image_paths: list[Path]) -> tuple[int, int]:
    """Return the largest width and height among the images, so Neat accepts every one."""
    max_width = max_height = 0
    for image_path in image_paths:
        height, width = read_image(image_path).shape[:2]
        max_width, max_height = max(max_width, width), max(max_height, height)
    return max_width, max_height
