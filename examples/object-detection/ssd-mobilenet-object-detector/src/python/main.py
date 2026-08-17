"""SSD folder object detection via the model-managed BoxDecodeType.Ssd pipeline.

Runs TensorFlow SSD-MobileNet v1/v2 (300x300) and TorchVision SSDlite-MobileNetV3
(320x320) with an explicit preprocessing profile.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
DEFAULT_LABELS_PATH = Path(
    "examples/object-detection/ssd-mobilenet-object-detector/src/common/coco_labels.txt"
)

DEFAULT_PREPROCESSING_PROFILE = "tensorflow_ssd"
NUM_CLASSES = 91  # index 0 = background, 1..90 = COCO ids.

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


def config_value_or(section: Mapping[str, Any], key: str, default: Any) -> Any:
    """Return a scalar config value, treating YAML null as an omitted setting."""
    value = section.get(key)
    return default if value is None else value


def config_mapping_or_empty(value: Any, qualified_key: str) -> dict[str, Any]:
    """Normalize a missing YAML mapping and reject every other node type."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{qualified_key} must be a mapping")  # noqa: TRY004
    return value


def config_string_or(
    section: Mapping[str, Any], key: str, default: str, qualified_key: str
) -> str:
    """Parse a YAML string without coercing collections, numbers, or Booleans."""
    value = config_value_or(section, key, default)
    if not isinstance(value, str):
        raise ValueError(f"{qualified_key} must be a string")  # noqa: TRY004
    return value


def config_float_or(
    section: Mapping[str, Any], key: str, default: float, qualified_key: str
) -> float:
    """Parse a numeric scalar without accepting Python's Boolean-as-integer coercion."""
    value = config_value_or(section, key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{qualified_key} must be a number")  # noqa: TRY004
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
        raise ValueError(f"{qualified_key} must be an integer")  # noqa: TRY004
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


def image_to_tensor(image_bgr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(image_bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def decode_detections(
    tensors, image_width: int, image_height: int, max_detections: int
) -> list[dict[str, Any]]:
    """Decode the single BBOX tensor through Core's public detection API."""
    if len(tensors) != 1:
        raise RuntimeError(f"expected one BBOX tensor, got {len(tensors)}")
    result = pyneat.detections.decode_bbox_tensor(
        tensors[0], image_width, image_height, max_detections, False
    )
    return [
        {
            "box": [box.x1, box.y1, box.x2, box.y2],
            "score": float(box.score),
            "class_id": int(box.class_id),
        }
        for box in result.boxes
    ]


def visualize_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    labels: list[str],
) -> np.ndarray:
    image_copy = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (round(v) for v in det["box"])
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
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    """Machine-readable detection record, written when io.detections_json is set."""
    height, width = image_bgr.shape[:2]
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
            }
            for det in detections
        ],
    }


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
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        model_path_value = config_string_or(model_cfg, "path", "", "model.path")
        if not model_path_value.strip():
            raise ValueError("model.path must be a nonempty path")
        model_path = Path(model_path_value)
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
        score_threshold = config_float_or(
            decode_cfg, "score_threshold", 0.55, "decode.score_threshold"
        )
        nms_iou = config_float_or(decode_cfg, "nms_iou", 0.60, "decode.nms_iou")
        max_detections = config_int_or(
            decode_cfg, "max_detections", 100, "decode.max_detections"
        )
        timeout_ms = config_int_or(
            runtime_cfg, "timeout_ms", 20000, "runtime.timeout_ms"
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        print("decode.score_threshold must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if not math.isfinite(nms_iou) or not 0.0 <= nms_iou <= 1.0:
        print("decode.nms_iou must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if max_detections < 1:
        print("decode.max_detections must be >= 1", file=sys.stderr)
        return 2
    if timeout_ms <= 0:
        print("runtime.timeout_ms must be > 0", file=sys.stderr)
        return 2
    try:
        normalization_for_profile(preprocessing_profile)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if output_dir.resolve() == input_dir.resolve():
        print("io.output_dir must differ from io.input_dir", file=sys.stderr)
        return 2

    try:
        labels = load_labels(labels_path)
    except (OSError, ValueError) as exc:
        print(f"Error loading assets: {exc}", file=sys.stderr)
        return 2

    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3
    # Imported after config validation so config errors are reported without a runtime install.
    global cv2, np, pyneat
    import cv2
    import numpy as np
    import pyneat

    try:
        mean, stddev = normalization_for_profile(preprocessing_profile)
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.enable = pyneat.AutoFlag.On
        options.preprocess.resize.enable = pyneat.AutoFlag.On
        options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
        options.preprocess.normalize.enable = pyneat.AutoFlag.On
        options.preprocess.normalize.mean = mean
        options.preprocess.normalize.stddev = stddev
        options.preprocess.normalize.has_explicit_stats = True
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        options.preprocess.color_convert.output_format = (
            pyneat.PreprocessColorFormat.RGB
        )
        options.decode_type = pyneat.BoxDecodeType.Ssd
        options.num_classes = NUM_CLASSES
        options.score_threshold = score_threshold
        options.nms_iou_threshold = nms_iou
        options.top_k = max_detections
        model = pyneat.Model(str(model_path), options)
        seed_bgr = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if seed_bgr is None:
            print(f"Failed to read build seed image: {image_paths[0]}", file=sys.stderr)
            return 3
        seed_tensor = image_to_tensor(seed_bgr)
        runner = model.build([seed_tensor], route_options=pyneat.ModelRouteOptions())
        runner.run([seed_tensor], timeout_ms=timeout_ms)
    except Exception as exc:
        logger.debug("Model/pipeline build failure", exc_info=exc)
        # Core resolves the preprocess frame from the MPK and validates the SSD recipe.
        print(f"Error loading model: {exc}", file=sys.stderr)
        print(
            f"  hint: verify model.path ({model_path}) points to a supported SSD model "
            "pack whose MLA input contract contains its model frame.",
            file=sys.stderr,
        )
        return 3

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        runner.close()
        print(f"Failed to prepare output directory: {exc}", file=sys.stderr)
        return 4

    records: list[dict[str, Any]] = []
    processed = 0
    try:
        for image_path in image_paths:
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

            detections = decode_detections(
                out, orig_bgr.shape[1], orig_bgr.shape[0], max_detections
            )

            output_path = output_dir / f"{output_stem(image_path)}.png"
            out_img = visualize_detections(orig_bgr, detections, labels)
            if not cv2.imwrite(str(output_path), out_img):
                print(f"Failed to write: {output_path}", file=sys.stderr)
                return 4

            if detections_json:
                records.append(
                    detections_record(image_path, orig_bgr, detections, labels)
                )
            processed += 1
            det_str = f"({len(detections)} detections)"
            print(
                f"[{processed}/{len(image_paths)}] {image_path.name} -> "
                f"{output_path.name} {det_str}"
            )
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
