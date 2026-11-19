"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import re

import sys

import time

from collections import Counter

import math

from dataclasses import dataclass, replace

from pathlib import Path

import cv2

import numpy as np

import pyneat

import yaml

class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that leaves yes/no/on/off as strings.

    PyYAML follows YAML 1.1 and coerces those to booleans, but the C++ parser
    accepts only true/false. Narrowing the resolver keeps both twins rejecting
    the same spellings.
    """

_ConfigLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)

DEFAULT_CONFIG = Path(get_package_share_directory('neat_pcb_defect_detector')) / "common" / "config.yaml"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

OUTPUT_TAG = "_pcb"

DEFAULT_INPUT_MAX_WIDTH = 3840

DEFAULT_INPUT_MAX_HEIGHT = 2160

DEFECT_COLORS = [
    (56, 56, 255),
    (29, 178, 255),
    (10, 249, 72),
    (255, 194, 0),
    (255, 0, 200),
    (49, 210, 207),
]

@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings resolved from config.yaml."""

    model_path: str
    labels_path: Path
    input_max_width: int
    input_max_height: int
    input_dir: Path
    output_dir: Path
    score_threshold: float
    nms_iou: float
    max_detections: int
    num_runs: int
    queue_depth: int
    profile: bool
    overlay: bool

def _section(raw: dict, name: str) -> dict:
    """One config section, rejecting a non-mapping the way the C++ parser does."""
    section = raw.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    return section

_INT_TEXT = re.compile(r"^[+-]?[0-9]+$")

def _value(section: dict, key: str):
    """Raw config value, or None when unset.

    The C++ parser treats an empty value and any spelling of null, quoted or
    not, as unset and falls back to the default.
    """
    value = section.get(key)
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value

def _str(section: dict, key: str, default: str) -> str:
    """Text config value; unset falls back to the default like the C++ parser."""
    value = _value(section, key)
    return default if value is None else str(value)

def _int(section: dict, key: str, default: int, name: str) -> int:
    """Integer config value, accepting the same spellings as the C++ parser.

    The C++ parser unquotes before parsing, so "8000" and 8000 are the same
    setting. A YAML float such as 1.0 reaches it as the text "1.0", which it
    rejects, so it is rejected here too. Inside quotes it skips leading
    whitespace but rejects trailing whitespace.
    """
    value = _value(section, key)
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value == value.rstrip() and _INT_TEXT.match(value.lstrip()):
        return int(value)
    raise ValueError(f"{name} must be an integer, got {value!r}")

def _float(section: dict, key: str, default: float, name: str) -> float:
    """Floating-point config value, accepting the same spellings as the C++ parser.

    Like the integer parser, it skips leading but rejects trailing whitespace
    inside quotes.
    """
    value = _value(section, key)
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and "_" not in value and value == value.rstrip():
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"{name} must be numeric, got {value!r}")

def _bool(section: dict, key: str, default: bool, name: str) -> bool:
    """Boolean config value, accepting the same spellings as the C++ parser.

    A quoted YAML scalar arrives as a string, and every nonempty string is
    truthy in Python, so "false" would silently mean True. The C++ parser
    compares the quoted text exactly, so surrounding whitespace is rejected.
    """
    value = _value(section, key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name} must be true or false, got {value!r}")

def build_app_config(raw: dict) -> AppConfig:
    """Map a parsed config mapping onto AppConfig without validating it."""
    model_cfg = _section(raw, "model")
    io_cfg = _section(raw, "io")
    decode_cfg = _section(raw, "decode")
    runtime_cfg = _section(raw, "runtime")
    output_cfg = _section(raw, "output")

    return AppConfig(
        model_path=_str(model_cfg, "path", ""),
        labels_path=Path(_str(
            model_cfg,
            "labels",
            "examples/object-detection/pcb-defect-detector/src/common/pcb_label.txt",
        )),
        input_max_width=_int(model_cfg, "input_max_width", DEFAULT_INPUT_MAX_WIDTH,
                             "model.input_max_width"),
        input_max_height=_int(model_cfg, "input_max_height", DEFAULT_INPUT_MAX_HEIGHT,
                              "model.input_max_height"),
        input_dir=Path(_str(io_cfg, "input_dir", "assets/datasets/pcb")),
        output_dir=Path(_str(io_cfg, "output_dir", "sandbox/pcb-defect-detector")),
        score_threshold=_float(decode_cfg, "score_threshold", 0.25, "decode.score_threshold"),
        nms_iou=_float(decode_cfg, "nms_iou", 0.45, "decode.nms_iou"),
        max_detections=_int(decode_cfg, "max_detections", 300, "decode.max_detections"),
        num_runs=_int(runtime_cfg, "num_runs", 1, "runtime.num_runs"),
        queue_depth=_int(runtime_cfg, "queue_depth", 8, "runtime.queue_depth"),
        profile=_bool(runtime_cfg, "profile", False, "runtime.profile"),
        overlay=_bool(output_cfg, "overlay", True, "output.overlay"),
    )

def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels

def class_name(class_id: int, labels: list[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"

def class_color(class_id: int) -> tuple[int, int, int]:
    return DEFECT_COLORS[abs(class_id) % len(DEFECT_COLORS)]

def round_half_up(value: float) -> int:
    """Round like C++ std::round. Python's round() is banker's rounding, which
    disagrees on exact .5 values and shifts drawn boxes by a pixel."""
    return int(math.floor(value + 0.5))

def decode_detections(outputs, img_w: int, img_h: int, cfg: AppConfig) -> list[dict]:
    """Decode the model's BBOX payload into source-image coordinates.

    Raises RuntimeError if the response is not a usable detection result;
    returns [] when a valid response found nothing.
    """
    if len(outputs) != 1:
        raise RuntimeError(
            f"expected one BBOX tensor from model-managed BoxDecode, got {len(outputs)}"
        )

    bbox_tensor = outputs[0]
    detection_format = pyneat.detections.read_detection_format(bbox_tensor)
    if not pyneat.detections.format_is_bbox(detection_format):
        raise RuntimeError(
            f"model returned no BBOX detection tensor (format: '{detection_format or '<none>'}'); "
            "the model package must expose YOLO26 BoxDecode output"
        )

    # strict=True rejects a truncated or over-long payload instead of decoding it
    # as zero boxes. The decoder sizes the payload for top_k boxes, so the result
    # never exceeds max_detections.
    try:
        result = pyneat.detections.decode_bbox_tensor(bbox_tensor, img_w, img_h, 0, True)
    except Exception as error:
        raise RuntimeError(f"malformed BBOX payload: {error}") from error

    detections: list[dict] = []
    for box in result.boxes:
        # Core stores the threshold as float32; compare there so a boundary
        # score is kept or dropped identically in both twins.
        if np.float32(box.score) < np.float32(cfg.score_threshold):
            continue
        x1, y1 = float(box.x1), float(box.y1)
        x2, y2 = float(box.x2), float(box.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(box.score), class_id=int(box.class_id))
        )
    return detections

def draw_boxes(bgr, detections: list[dict], labels: list[str]) -> None:
    """Draw class-colored defect boxes and labels on a BGR image in place."""
    for detection in detections:
        x1 = max(0, round_half_up(detection["x1"]))
        y1 = max(0, round_half_up(detection["y1"]))
        x2 = min(bgr.shape[1] - 1, round_half_up(detection["x2"]))
        y2 = min(bgr.shape[0] - 1, round_half_up(detection["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue

        color = class_color(detection["class_id"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)

        text = f"{class_name(detection['class_id'], labels)} {detection['score']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(0, y1 - text_h - 4)
        cv2.rectangle(bgr, (x1, label_y), (x1 + text_w + 2, y1), color, -1)
        cv2.putText(
            bgr,
            text,
            (x1 + 1, max(10, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

def to_bgr_tensor(bgr):
    """Wrap a raw BGR uint8 image in a pyneat tensor placed in EV74 memory."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )

def model_options(cfg: AppConfig, labels: list[str]) -> pyneat.ModelOptions:
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # Resize stays at the model package default: letterbox, grey padding.
    opt.preprocess.input_max_width = cfg.input_max_width
    opt.preprocess.input_max_height = cfg.input_max_height
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    opt.num_classes = len(labels)
    return opt

def run_options(cfg: AppConfig) -> pyneat.RunOptions:
    opt = pyneat.RunOptions()
    opt.queue_depth = cfg.queue_depth
    # Realtime is the only preset that applies a new input size on the first
    # frame; the others hold a frame until a second one of the same size
    # arrives, which a one-image-at-a-time batch never sends. Its KeepLatest
    # overflow policy never drops here, because run() pushes blocking.
    opt.preset = pyneat.RunPreset.Realtime
    return opt
