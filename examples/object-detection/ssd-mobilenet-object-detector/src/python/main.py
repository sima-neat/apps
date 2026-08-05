"""SSD folder object detection via the model-managed BoxDecodeType.Ssd pipeline.

Runs TensorFlow SSD-MobileNet v1/v2 (300x300) and TorchVision SSDlite-MobileNetV3
(320x320) with an explicit preprocessing profile.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import struct
import sys
import time
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple

import yaml


VERBOSE = False
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/ssd_mobilenet_v2_heads_mpk.tar.gz"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
DEFAULT_LABELS_PATH = Path(
    "examples/object-detection/ssd-mobilenet-object-detector/src/common/coco_labels.txt"
)

# Default model frame. SSD300 and SSD-MobileNet v1/v2 are 300x300; v3 is 320x320.
# Override via `model.frame` in the config to match the model pack.
DEFAULT_MODEL_SIZE = 300
DEFAULT_PREPROCESSING_PROFILE = "tensorflow_ssd"
NUM_CLASSES = 91  # index 0 = background, 1..90 = COCO ids.
BBOX_RECORD_SIZE = 24  # int32 x, y, w, h + float32 score + int32 class_id

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


class AggregateSuppressionOptions(NamedTuple):
    """Application policy for hiding same-class crowd regions after model-managed decode."""

    enabled: bool = False
    min_parent_area_fraction: float = 0.20
    min_child_containment: float = 0.90
    max_child_area_ratio: float = 0.25
    min_children: int = 2


def config_value_or(section: Mapping[str, Any], key: str, default: Any) -> Any:
    """Return a scalar config value, treating YAML null as an omitted setting."""
    value = section.get(key)
    return default if value is None else value


def config_mapping_or_empty(value: Any, qualified_key: str) -> dict[str, Any]:
    """Normalize a missing YAML mapping and reject every other node type."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{qualified_key} must be a mapping")
    return value


def config_string_or(
    section: Mapping[str, Any], key: str, default: str, qualified_key: str
) -> str:
    """Parse a YAML string without coercing collections, numbers, or Booleans."""
    value = config_value_or(section, key, default)
    if not isinstance(value, str):
        raise ValueError(f"{qualified_key} must be a string")
    return value


def config_bool_or(
    section: Mapping[str, Any], key: str, default: bool, qualified_key: str
) -> bool:
    """Parse a YAML Boolean with the same null/default and string rules as C++."""
    value = config_value_or(section, key, default)
    if value is True or value is False:
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"{qualified_key} must be true or false")


def config_float_or(
    section: Mapping[str, Any], key: str, default: float, qualified_key: str
) -> float:
    """Parse a numeric scalar without accepting Python's Boolean-as-integer coercion."""
    value = config_value_or(section, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{qualified_key} must be a number")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{qualified_key} must be a number") from exc


def config_int_or(
    section: Mapping[str, Any], key: str, default: int, qualified_key: str
) -> int:
    """Parse an integer-valued scalar without truncating fractions or Booleans."""
    value = config_value_or(section, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{qualified_key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError(f"{qualified_key} must be an integer")
    try:
        parsed = Decimal(value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{qualified_key} must be an integer") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise ValueError(f"{qualified_key} must be an integer")
    return int(parsed)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def output_stem(image_path: Path) -> str:
    """Output stem keeping the (case-preserved) source extension so frame.jpg/frame.png don't collide."""
    ext = image_path.suffix.lstrip(".")
    return f"{image_path.stem}_{ext}" if ext else image_path.stem


def paths_alias(lhs: Path, rhs: Path) -> bool:
    """Return true for canonical-path aliases and existing hard-link aliases."""
    if lhs.resolve() == rhs.resolve():
        return True
    try:
        return lhs.samefile(rhs)
    except OSError:
        return False


def clear_output_images(output_dir: Path, expected_names: set[str]) -> int:
    """Remove expected output entries by pathname without following symlinks."""
    removed = 0
    for name in expected_names:
        path = output_dir / name
        # exists() follows symlinks and is false for dangling links; is_symlink() closes that gap.
        if path.is_symlink() or path.exists():
            if path.is_dir() and not path.is_symlink():
                raise IsADirectoryError(
                    f"generated overlay path is a directory: {path}"
                )
            path.unlink()
            removed += 1
    return removed


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[ssd-debug] {msg}", flush=True)


def _resolve_asset(configured: str, default_name: str) -> Path:
    """Resolve a labels asset: the configured path if present. Substitute the packaged
    src/common copy only for the empty/default reference, not a missing custom path."""
    candidate = Path(configured)
    if candidate.is_file():
        return candidate
    if not configured or candidate == DEFAULT_LABELS_PATH:
        fallback = COMMON_DIR / default_name
        if fallback.is_file():
            return fallback
    return candidate


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    labels = [
        line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels


def class_name(labels: list[str], class_idx: int) -> str:
    if 0 <= class_idx < len(labels):
        name = labels[class_idx]
        if name and name != "N/A":
            return name
    return f"class_{class_idx}"


def class_color(class_idx: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_idx) % len(BOX_COLORS)]


def normalization_for_profile(profile: str) -> tuple[list[float], list[float]]:
    if profile == "tensorflow_ssd":
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if profile == "torchvision_ssdlite":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    raise ValueError(
        "model.preprocessing_profile must be tensorflow_ssd or torchvision_ssdlite"
    )


def make_model_options(
    score_threshold: float,
    nms_iou: float,
    max_detections: int,
    model_frame: int,
    preprocessing_profile: str = DEFAULT_PREPROCESSING_PROFILE,
) -> "pyneat.ModelOptions":
    """Model-managed SSD decode with an explicit source-model preprocessing profile.

    Frame is 300 for SSD300/v1/v2 and 320 for either 320x320 recipe.
    """
    mean, stddev = normalization_for_profile(preprocessing_profile)
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    # STRETCH, not the default Letterbox: every registered SSD recipe uses a direct square resize.
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    opt.preprocess.resize.width = model_frame
    opt.preprocess.resize.height = model_frame
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean = mean
    opt.preprocess.normalize.stddev = stddev
    opt.preprocess.normalize.has_explicit_stats = True
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.decode_type = pyneat.BoxDecodeType.Ssd
    opt.num_classes = NUM_CLASSES
    opt.score_threshold = score_threshold
    opt.nms_iou_threshold = nms_iou
    opt.top_k = max_detections
    return opt


def image_to_tensor(image_bgr: "np.ndarray") -> "pyneat.Tensor":
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(image_bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def extract_bbox_payload(tensors) -> bytes | None:
    if len(tensors) != 1:
        return None
    payload = tensors[0].copy_payload_bytes()
    return payload or None


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int) -> list[dict[str, Any]]:
    """Parse the packed BBOX records emitted by the model-managed BoxDecode stage."""
    if len(payload) < 4:
        return []
    count = min(
        struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // BBOX_RECORD_SIZE
    )
    detections: list[dict[str, Any]] = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += BBOX_RECORD_SIZE
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {"box": [x1, y1, x2, y2], "score": float(score), "class_id": int(cls_id)}
        )
    return detections


def suppress_aggregate_detections(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    options: AggregateSuppressionOptions,
) -> list[dict[str, Any]]:
    """Remove a large region only when it contains multiple smaller same-class objects.

    COCO-trained detectors can emit an ``iscrowd`` training region as an ordinary class box;
    the packed runtime output has no crowd flag. This optional app policy leaves Core's faithful
    SSD decode/NMS and raw output untouched and suppresses only the aggregate visualization.
    """
    if (
        not options.enabled
        or len(detections) < options.min_children + 1
        or image_width <= 0
        or image_height <= 0
    ):
        return detections

    image_area = float(image_width * image_height)
    coords = [tuple(float(value) for value in det["box"]) for det in detections]
    areas = [max(0.0, x2 - x1) * max(0.0, y2 - y1) for x1, y1, x2, y2 in coords]
    by_class: dict[int, list[int]] = {}
    for index, det in enumerate(detections):
        by_class.setdefault(int(det["class_id"]), []).append(index)

    suppressed = [False] * len(detections)
    for parent_index, (px1, py1, px2, py2) in enumerate(coords):
        parent_area = areas[parent_index]
        if parent_area < image_area * options.min_parent_area_fraction:
            continue

        children = 0
        max_child_area = parent_area * options.max_child_area_ratio
        class_id = int(detections[parent_index]["class_id"])
        for child_index in by_class[class_id]:
            if child_index == parent_index:
                continue
            child_area = areas[child_index]
            if child_area <= 0.0 or child_area > max_child_area:
                continue
            cx1, cy1, cx2, cy2 = coords[child_index]
            intersection_width = max(0.0, min(px2, cx2) - max(px1, cx1))
            intersection_height = max(0.0, min(py2, cy2) - max(py1, cy1))
            if (
                intersection_width * intersection_height / child_area
                >= options.min_child_containment
            ):
                children += 1
                if children >= options.min_children:
                    suppressed[parent_index] = True
                    break

    return [det for index, det in enumerate(detections) if not suppressed[index]]


def visualize_detections(
    image_bgr: "np.ndarray",
    detections: list[dict[str, Any]],
    labels: list[str],
) -> "np.ndarray":
    image_copy = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det["box"])
        color = class_color(det["class_id"])
        text = f"{class_name(labels, det['class_id'])} {det['score']:.2f}"
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_copy,
            text,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return image_copy


def detections_record(
    image_path: Path,
    image_bgr: "np.ndarray",
    detections: list[dict[str, Any]],
    displayed_detections: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Machine-readable detection record, written when io.detections_json is set."""
    height, width = image_bgr.shape[:2]
    displayed_ids = {id(det) for det in displayed_detections}
    return {
        "image": image_path.name,
        "width": int(width),
        "height": int(height),
        "detections": [
            {
                "class_id": det["class_id"],
                "label": class_name(labels, det["class_id"]),
                "score": det["score"],
                "box": det["box"],
                "displayed": id(det) in displayed_ids,
            }
            for det in detections
        ],
    }


def format_profile_stats(name: str, values: list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    fps = float(len(arr)) / arr.sum() if arr.sum() > 0 else 0.0
    return (
        f"  {name}: mean={arr.mean():.8f}s, min={arr.min():.8f}s, "
        f"max={arr.max():.8f}s, FPS={fps:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SSD folder object detection example (SSD300, MobileNet v1/v2/v3)"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration"
    )
    args = parser.parse_args()

    try:
        with args.config.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        print(
            f"failed to open config file: {args.config} ({exc.strerror})",
            file=sys.stderr,
        )
        return 2
    except yaml.YAMLError as exc:
        print(f"failed to parse config file: {args.config} ({exc})", file=sys.stderr)
        return 2

    try:
        raw = config_mapping_or_empty(loaded, "config root")
        model_cfg = config_mapping_or_empty(raw.get("model"), "model")
        io_cfg = config_mapping_or_empty(raw.get("io"), "io")
        decode_cfg = config_mapping_or_empty(raw.get("decode"), "decode")
        runtime_cfg = config_mapping_or_empty(raw.get("runtime"), "runtime")
        output_cfg = config_mapping_or_empty(raw.get("output"), "output")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        model_path = Path(
            config_string_or(model_cfg, "path", DEFAULT_MODEL_PATH, "model.path")
        )
        preprocessing_profile = config_string_or(
            model_cfg,
            "preprocessing_profile",
            DEFAULT_PREPROCESSING_PROFILE,
            "model.preprocessing_profile",
        )
        labels_path = _resolve_asset(
            config_string_or(model_cfg, "labels", "", "model.labels"),
            "coco_labels.txt",
        )
        input_dir = Path(
            config_string_or(
                io_cfg, "input_dir", "assets/datasets/coco", "io.input_dir"
            )
        )
        output_dir = Path(
            config_string_or(
                io_cfg,
                "output_dir",
                "sandbox/ssd-mobilenet-object-detector",
                "io.output_dir",
            )
        )
        detections_json = config_string_or(
            io_cfg, "detections_json", "", "io.detections_json"
        )
        model_frame = config_int_or(
            model_cfg, "frame", DEFAULT_MODEL_SIZE, "model.frame"
        )
        score_threshold = config_float_or(
            decode_cfg, "score_threshold", 0.55, "decode.score_threshold"
        )
        nms_iou = config_float_or(decode_cfg, "nms_iou", 0.60, "decode.nms_iou")
        max_detections = config_int_or(
            decode_cfg, "max_detections", 100, "decode.max_detections"
        )
        aggregate_suppression_enabled = config_bool_or(
            output_cfg,
            "aggregate_suppression",
            False,
            "output.aggregate_suppression",
        )
        profile = config_bool_or(runtime_cfg, "profile", False, "runtime.profile")
        overlay = config_bool_or(output_cfg, "overlay", True, "output.overlay")
        verbose = config_bool_or(runtime_cfg, "verbose", False, "runtime.verbose")
        aggregate_suppression = AggregateSuppressionOptions(
            enabled=aggregate_suppression_enabled,
            min_parent_area_fraction=config_float_or(
                output_cfg,
                "aggregate_min_parent_area_fraction",
                0.20,
                "output.aggregate_min_parent_area_fraction",
            ),
            min_child_containment=config_float_or(
                output_cfg,
                "aggregate_min_child_containment",
                0.90,
                "output.aggregate_min_child_containment",
            ),
            max_child_area_ratio=config_float_or(
                output_cfg,
                "aggregate_max_child_area_ratio",
                0.25,
                "output.aggregate_max_child_area_ratio",
            ),
            min_children=config_int_or(
                output_cfg,
                "aggregate_min_children",
                2,
                "output.aggregate_min_children",
            ),
        )
        num_runs = config_int_or(runtime_cfg, "num_runs", 1, "runtime.num_runs")
        timeout_ms = config_int_or(
            runtime_cfg, "timeout_ms", 20000, "runtime.timeout_ms"
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    global VERBOSE
    VERBOSE = verbose

    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        print("decode.score_threshold must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if not math.isfinite(nms_iou) or not 0.0 <= nms_iou <= 1.0:
        print("decode.nms_iou must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if max_detections < 1:
        print("decode.max_detections must be >= 1", file=sys.stderr)
        return 2
    if (
        not math.isfinite(aggregate_suppression.min_parent_area_fraction)
        or not 0.0 <= aggregate_suppression.min_parent_area_fraction <= 1.0
    ):
        print(
            "output.aggregate_min_parent_area_fraction must be in [0.0, 1.0]",
            file=sys.stderr,
        )
        return 2
    if (
        not math.isfinite(aggregate_suppression.min_child_containment)
        or not 0.0 < aggregate_suppression.min_child_containment <= 1.0
    ):
        print(
            "output.aggregate_min_child_containment must be in (0.0, 1.0]",
            file=sys.stderr,
        )
        return 2
    if (
        not math.isfinite(aggregate_suppression.max_child_area_ratio)
        or not 0.0 < aggregate_suppression.max_child_area_ratio <= 1.0
    ):
        print(
            "output.aggregate_max_child_area_ratio must be in (0.0, 1.0]",
            file=sys.stderr,
        )
        return 2
    if aggregate_suppression.min_children < 2:
        print("output.aggregate_min_children must be >= 2", file=sys.stderr)
        return 2
    if timeout_ms <= 0:
        print("runtime.timeout_ms must be > 0", file=sys.stderr)
        return 2
    if num_runs < 1:
        print("runtime.num_runs must be >= 1", file=sys.stderr)
        return 2
    if model_frame not in (300, 320):
        print(
            f"model.frame must be 300 (SSD300/MobileNet v1/v2) or 320 (v3), got {model_frame}",
            file=sys.stderr,
        )
        return 2
    try:
        normalization_for_profile(preprocessing_profile)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if preprocessing_profile == "torchvision_ssdlite" and model_frame != 320:
        print(
            "model.preprocessing_profile torchvision_ssdlite requires model.frame=320",
            file=sys.stderr,
        )
        return 2
    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    # Only overlay runs write into output_dir, so only they must not alias input_dir.
    if not profile and overlay and paths_alias(output_dir, input_dir):
        print("io.output_dir must differ from io.input_dir", file=sys.stderr)
        return 2

    try:
        labels = load_labels(labels_path)
    except Exception as exc:
        print(f"Error loading assets: {exc}", file=sys.stderr)
        return 2

    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3
    if not profile and overlay:
        consumed_paths = (args.config, model_path, labels_path, *image_paths)
        for image_path in image_paths:
            overlay_path = output_dir / f"{output_stem(image_path)}.png"
            if overlay_path.is_dir() and not overlay_path.is_symlink():
                print(
                    f"generated overlay path must not be a directory: {overlay_path}",
                    file=sys.stderr,
                )
                return 2
            if any(paths_alias(overlay_path, consumed) for consumed in consumed_paths):
                print(
                    f"generated overlay must not overwrite a consumed input: {overlay_path}",
                    file=sys.stderr,
                )
                return 2
    if not profile and detections_json:
        report_path = Path(detections_json)
        for consumed_path in (args.config, model_path, labels_path):
            if paths_alias(consumed_path, report_path):
                print(
                    f"io.detections_json must not overwrite a consumed input: {consumed_path}",
                    file=sys.stderr,
                )
                return 2
        for image_path in image_paths:
            if paths_alias(image_path, report_path):
                print(
                    f"io.detections_json must not overwrite an input image: {image_path}",
                    file=sys.stderr,
                )
                return 2
            overlay_path = output_dir / f"{output_stem(image_path)}.png"
            if overlay and paths_alias(overlay_path, report_path):
                print(
                    f"io.detections_json must not overwrite a generated overlay: {overlay_path}",
                    file=sys.stderr,
                )
                return 2
        if is_image(report_path) and paths_alias(report_path.parent, input_dir):
            print(
                "io.detections_json must not use an image filename inside io.input_dir: "
                f"{report_path}",
                file=sys.stderr,
            )
            return 2

    # Imported after config validation so config errors are reported without a runtime install.
    global cv2, np, pyneat
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    import pyneat  # noqa: F401

    try:
        model = pyneat.Model(
            str(model_path),
            make_model_options(
                score_threshold,
                nms_iou,
                max_detections,
                model_frame,
                preprocessing_profile,
            ),
        )
        seed_bgr = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if seed_bgr is None:
            print(f"Failed to read build seed image: {image_paths[0]}", file=sys.stderr)
            return 3
        seed_tensor = image_to_tensor(seed_bgr)
        runner = model.build([seed_tensor], route_options=pyneat.ModelRouteOptions())
        runner.run([seed_tensor], timeout_ms=timeout_ms)
        _log("warmup done")
    except Exception as exc:
        logger.debug("Model/pipeline build failure", exc_info=exc)
        # Recipe/frame mismatch surfaces here; point at the two config knobs.
        print(f"Error loading model: {exc}", file=sys.stderr)
        print(
            f"  hint: check model.path ({model_path}) and model.frame ({model_frame}); "
            "use 300 for SSD300/MobileNet v1/v2, 320 for v3.",
            file=sys.stderr,
        )
        return 3

    if profile:
        try:
            image_path = image_paths[0]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Failed to read image: {image_path}", file=sys.stderr)
                return 4
            tensor = image_to_tensor(bgr)
            infer_times: list[float] = []
            parse_times: list[float] = []
            last_detections: list[dict[str, Any]] = []
            last_displayed_detections: list[dict[str, Any]] = []

            for _ in range(num_runs):
                t0 = time.perf_counter()
                out = runner.run([tensor], timeout_ms=timeout_ms)
                t1 = time.perf_counter()
                payload = extract_bbox_payload(out)
                # Any missing run makes the reported stats incomplete, so fail the whole profile.
                if not payload:
                    print(
                        "Profiling failed: model returned no BBOX payload",
                        file=sys.stderr,
                    )
                    return 4
                last_detections = parse_bbox_payload(
                    payload, bgr.shape[1], bgr.shape[0]
                )
                last_displayed_detections = suppress_aggregate_detections(
                    last_detections,
                    bgr.shape[1],
                    bgr.shape[0],
                    aggregate_suppression,
                )
                t2 = time.perf_counter()
                infer_times.append(t1 - t0)
                parse_times.append(t2 - t1)

            print(f"Profiling over {len(infer_times)} runs (image='{image_path}'):")
            print(
                format_profile_stats(
                    "Pipeline run (preprocess+infer+decode)", infer_times
                )
            )
            print(format_profile_stats("Output parsing + display policy", parse_times))
            print(
                f"Last run detections: {len(last_detections)} raw, "
                f"{len(last_displayed_detections)} displayed"
            )
            for i, det in enumerate(last_detections[:20]):
                box = det["box"]
                print(
                    f"  [{i}] class={class_name(labels, det['class_id'])}({det['class_id']}) "
                    f"score={det['score']:.4f} box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
                )
            return 0
        except Exception as exc:
            logger.debug("Profiling failure", exc_info=exc)
            print(f"Error during profiling: {exc}", file=sys.stderr)
            return 4
        finally:
            runner.close()

    # Only overlay runs touch output_dir, so a JSON-only run leaves it alone.
    if overlay:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            expected_outputs = {f"{output_stem(p)}.png" for p in image_paths}
            removed_outputs = clear_output_images(output_dir, expected_outputs)
        except OSError as exc:
            runner.close()
            print(f"Failed to prepare output directory: {exc}", file=sys.stderr)
            return 4
        if removed_outputs:
            print(f"Cleared {removed_outputs} stale output images")

    records: list[dict[str, Any]] = []
    processed = 0
    try:
        for image_path in image_paths:
            _log(f"Reading image: {image_path}")
            orig_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if orig_bgr is None:
                print(f"Failed to read image: {image_path}", file=sys.stderr)
                return 3

            try:
                out = runner.run([image_to_tensor(orig_bgr)], timeout_ms=timeout_ms)
            except Exception as exc:
                logger.debug("Inference failure", exc_info=exc)
                print(
                    f"Error during inference for {image_path}: {exc}", file=sys.stderr
                )
                return 3

            payload = extract_bbox_payload(out)
            if not payload:
                print(
                    f"Model returned no BBOX payload for {image_path}", file=sys.stderr
                )
                return 4
            detections = parse_bbox_payload(
                payload, orig_bgr.shape[1], orig_bgr.shape[0]
            )
            displayed_detections = suppress_aggregate_detections(
                detections,
                orig_bgr.shape[1],
                orig_bgr.shape[0],
                aggregate_suppression,
            )

            output_name = ""
            if overlay:
                output_path = output_dir / f"{output_stem(image_path)}.png"
                out_img = visualize_detections(orig_bgr, displayed_detections, labels)
                # imwrite returns False (never raises) on failure; a failed overlay fails the run.
                if not cv2.imwrite(str(output_path), out_img):
                    print(f"Failed to write: {output_path}", file=sys.stderr)
                    return 4
                output_name = output_path.name

            if detections_json:
                records.append(
                    detections_record(
                        image_path, orig_bgr, detections, displayed_detections, labels
                    )
                )
            processed += 1
            det_str = (
                f"({len(detections)} detections, {len(displayed_detections)} displayed)"
            )
            if overlay:
                print(
                    f"[{processed}/{len(image_paths)}] {image_path.name} -> {output_name} {det_str}"
                )
            else:
                print(f"[{processed}/{len(image_paths)}] {image_path.name} {det_str}")
    finally:
        runner.close()

    if detections_json:
        json_path = Path(detections_json)
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps({"images": records}, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(
                f"Failed to write detections json: {json_path} ({exc.strerror})",
                file=sys.stderr,
            )
            return 4
        print(f"Wrote detections: {json_path}")

    print(f"Done: {processed} images processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
