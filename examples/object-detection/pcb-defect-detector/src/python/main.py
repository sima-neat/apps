"""PCB defect detection over a folder of images using pyneat.

Images of any resolution are letterboxed to the model input (640x640) before
inference. The compiled YOLO26n model pack owns the rest of the Neat path:

    color convert + normalize -> MLA inference -> on-device YOLO26 box decode

Detections come back in letterboxed coordinates, are mapped onto the original
frame, and are drawn on the original image, one annotated image per input.

Exit codes: 0 success | 1 invalid configuration | 2 missing config or input
directory | 3 no images | 4 runtime error.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_INPUT_SIZE = 640
# Grey pad value used by the YOLO letterbox convention.
PAD_VALUE = 114

# One BBOX record: x, y, w, h (int32), score (float32), class_id (int32).
BBOX_RECORD_FORMAT = "<iiiifi"
BBOX_RECORD_SIZE = struct.calcsize(BBOX_RECORD_FORMAT)

# BGR colors, index-aligned with pcb_label.txt.
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
    input_size: int
    input_dir: Path
    output_dir: Path
    score_threshold: float
    nms_iou: float
    max_detections: int
    timeout_ms: int
    num_runs: int
    queue_depth: int
    profile: bool
    overlay: bool


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_app_config(raw: dict) -> AppConfig:
    """Map a parsed config mapping onto AppConfig without validating it."""
    model_cfg = raw.get("model") or {}
    io_cfg = raw.get("io") or {}
    decode_cfg = raw.get("decode") or {}
    runtime_cfg = raw.get("runtime") or {}
    output_cfg = raw.get("output") or {}

    return AppConfig(
        model_path=str(model_cfg.get("path", "")),
        labels_path=Path(str(model_cfg.get("labels", ""))),
        input_size=int(model_cfg.get("input_size", DEFAULT_INPUT_SIZE)),
        input_dir=Path(str(io_cfg.get("input_dir", "assets/datasets/pcb"))),
        output_dir=Path(str(io_cfg.get("output_dir", "sandbox/pcb-defect-detector"))),
        score_threshold=float(decode_cfg.get("score_threshold", 0.25)),
        nms_iou=float(decode_cfg.get("nms_iou", 0.45)),
        max_detections=int(decode_cfg.get("max_detections", 300)),
        timeout_ms=int(runtime_cfg.get("timeout_ms", 8000)),
        num_runs=int(runtime_cfg.get("num_runs", 1)),
        queue_depth=int(runtime_cfg.get("queue_depth", 8)),
        profile=bool(runtime_cfg.get("profile", False)),
        overlay=bool(output_cfg.get("overlay", True)),
    )


def validate_config(cfg: AppConfig) -> None:
    """Raise ValueError when a resolved configuration cannot be run."""
    if not cfg.model_path:
        raise ValueError("model.path must be set to a compiled model package")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must point to a labels file")
    if cfg.input_size < 1:
        raise ValueError(f"model.input_size must be >= 1, got {cfg.input_size}")
    if not 0.0 <= cfg.score_threshold <= 1.0:
        raise ValueError(
            f"decode.score_threshold must be in [0.0, 1.0], got {cfg.score_threshold}"
        )
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError(f"decode.nms_iou must be in [0.0, 1.0], got {cfg.nms_iou}")
    if cfg.max_detections < 1:
        raise ValueError(f"decode.max_detections must be >= 1, got {cfg.max_detections}")
    if cfg.timeout_ms <= 0:
        raise ValueError(f"runtime.timeout_ms must be > 0, got {cfg.timeout_ms}")
    if cfg.num_runs < 1:
        raise ValueError(f"runtime.num_runs must be >= 1, got {cfg.num_runs}")
    if cfg.queue_depth < 1:
        raise ValueError(f"runtime.queue_depth must be >= 1, got {cfg.queue_depth}")


def load_app_config(config_path: Path) -> AppConfig:
    """Read, resolve, and validate the example configuration."""
    cfg = build_app_config(load_config(config_path))
    validate_config(cfg)
    return cfg


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and is_image(path))


def class_name(class_id: int, labels: list[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    return DEFECT_COLORS[max(0, class_id) % len(DEFECT_COLORS)]


@dataclass(frozen=True)
class Letterbox:
    """How a source frame was fitted into the square model input."""

    image: object
    scale: float = 1.0
    pad_x: int = 0
    pad_y: int = 0


def round_half_up(value: float) -> int:
    """Round like C++ std::round. Python's round() is banker's rounding, which
    disagrees on exact .5 values and shifts drawn boxes by a pixel."""
    return int(math.floor(value + 0.5))


def letterbox(bgr, size: int) -> Letterbox:
    """Aspect-preserving resize into a size x size canvas, centered on a grey pad."""
    import cv2
    import numpy as np

    height, width = bgr.shape[:2]
    if (width, height) == (size, size):
        return Letterbox(image=bgr)  # Already model-sized: no resample, so pixels stay exact.

    scale = min(size / width, size / height)
    scaled_w = max(1, round_half_up(width * scale))
    scaled_h = max(1, round_half_up(height * scale))
    pad_x = (size - scaled_w) // 2
    pad_y = (size - scaled_h) // 2

    canvas = np.full((size, size, bgr.shape[2]), PAD_VALUE, dtype=bgr.dtype)
    canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = cv2.resize(
        bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
    )
    return Letterbox(image=canvas, scale=scale, pad_x=pad_x, pad_y=pad_y)


def to_source_coordinates(detections: list[dict], lb: Letterbox, width: int, height: int) -> list[dict]:
    """Undo the letterbox so boxes land on the original frame."""

    def unpad(value: float, pad: int, limit: int) -> float:
        return max(0.0, min((value - pad) / lb.scale, float(limit)))

    mapped: list[dict] = []
    for detection in detections:
        box = dict(
            detection,
            x1=unpad(detection["x1"], lb.pad_x, width),
            x2=unpad(detection["x2"], lb.pad_x, width),
            y1=unpad(detection["y1"], lb.pad_y, height),
            y2=unpad(detection["y2"], lb.pad_y, height),
        )
        if box["x2"] > box["x1"] and box["y2"] > box["y1"]:
            mapped.append(box)
    return mapped


def output_path_for(image_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{image_path.stem}.png"


def clear_output_images(output_dir: Path, input_dir: Path) -> int:
    """Remove stale annotated images so a rerun cannot leave orphaned results."""
    if output_dir.resolve() == input_dir.resolve():
        print(
            f"Skipping output cleanup because output_dir matches input_dir: {output_dir}",
            file=sys.stderr,
        )
        return 0

    removed = 0
    for path in output_dir.iterdir():
        if path.is_file() and is_image(path):
            path.unlink()
            removed += 1
    return removed


def extract_bbox_payload(tensors) -> bytes | None:
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def parse_bbox_payload(
    payload: bytes,
    img_w: int,
    img_h: int,
    min_score: float,
    max_detections: int = 0,
) -> list[dict]:
    """Parse a BBOX payload into detections clamped to the original image size."""
    if not payload or len(payload) < 4:
        return []

    declared = struct.unpack_from("<I", payload, 0)[0]
    count = min(declared, (len(payload) - 4) // BBOX_RECORD_SIZE)
    if max_detections > 0:
        count = min(count, max_detections)
    detections: list[dict] = []
    offset = 4
    for _ in range(count):
        x, y, w, h, score, class_id = struct.unpack_from(BBOX_RECORD_FORMAT, payload, offset)
        offset += BBOX_RECORD_SIZE
        if float(score) < min_score:
            continue

        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score), class_id=int(class_id))
        )
    return detections


def draw_boxes(bgr, detections: list[dict], labels: list[str], thickness: int = 2) -> None:
    """Draw class-colored defect boxes and labels on a BGR image in place."""
    import cv2

    for detection in detections:
        x1 = max(0, round_half_up(detection["x1"]))
        y1 = max(0, round_half_up(detection["y1"]))
        x2 = min(bgr.shape[1] - 1, round_half_up(detection["x2"]))
        y2 = min(bgr.shape[0] - 1, round_half_up(detection["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue

        color = class_color(detection["class_id"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)

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
    import numpy as np
    import pyneat

    if bgr is None or bgr.size == 0:
        raise ValueError("empty input image")
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCB defect detection pipeline (YOLO26n)")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration"
    )
    parser.add_argument("--score", type=float, help="Override decode.score_threshold")
    parser.add_argument("--nms", type=float, help="Override decode.nms_iou")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the configuration and exit",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()

    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2

    try:
        raw = load_config(args.config)
    except OSError as error:
        print(f"Error: failed to read config {args.config}: {error}", file=sys.stderr)
        return 2
    except yaml.YAMLError as error:
        print(f"Error: invalid YAML in {args.config}: {error}", file=sys.stderr)
        return 2

    if args.score is not None:
        raw.setdefault("decode", {})["score_threshold"] = args.score
    if args.nms is not None:
        raw.setdefault("decode", {})["nms_iou"] = args.nms

    try:
        cfg = build_app_config(raw)
        validate_config(cfg)
        labels = load_labels(cfg.labels_path)
    except (ValueError, TypeError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.validate_config_only:
        print(
            f"[validate] model={cfg.model_path} classes={len(labels)} "
            f"input_size={cfg.input_size} "
            f"score_threshold={cfg.score_threshold:.2f} nms_iou={cfg.nms_iou:.2f} "
            f"max_detections={cfg.max_detections} timeout_ms={cfg.timeout_ms} "
            f"num_runs={cfg.num_runs} queue_depth={cfg.queue_depth}"
        )
        print("[validate] configuration OK")
        return 0

    if not cfg.input_dir.is_dir():
        print(f"Input directory does not exist: {cfg.input_dir}", file=sys.stderr)
        return 2

    images = discover_images(cfg.input_dir)
    if not images:
        print(f"No images found in {cfg.input_dir}", file=sys.stderr)
        return 3

    import cv2
    import pyneat

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    removed_outputs = clear_output_images(cfg.output_dir, cfg.input_dir)
    if removed_outputs:
        print(f"Cleared {removed_outputs} stale output images", flush=True)
    print(f"Model: {cfg.model_path}", flush=True)
    print(f"Found {len(images)} images in {cfg.input_dir}", flush=True)
    print(f"Classes: {', '.join(labels)}", flush=True)

    try:
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = cfg.score_threshold
        opt.nms_iou_threshold = cfg.nms_iou
        opt.top_k = cfg.max_detections
        opt.num_classes = len(labels)
        model = pyneat.Model(cfg.model_path, opt)

        seed = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if seed is None:
            print(f"Error: failed to read build seed image: {images[0].name}", file=sys.stderr)
            return 4
        # Every frame is letterboxed to this shape, so the graph ingress caps never change.
        seed_lb = letterbox(seed, cfg.input_size)

        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = cfg.queue_depth
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run_opt.preset = pyneat.RunPreset.Balanced
        runner = model.build(
            [to_bgr_tensor(seed_lb.image)],
            route_options=pyneat.ModelRouteOptions(),
            run_options=run_opt,
        )
        runner.run([to_bgr_tensor(seed_lb.image)], timeout_ms=cfg.timeout_ms)
        print("[WARMUP] done")

        all_images = images * cfg.num_runs
        if cfg.num_runs > 1:
            print(
                f"Looping {cfg.num_runs}x over {len(images)} images "
                f"({len(all_images)} total)",
                flush=True,
            )

        pipeline_start = time.perf_counter()
        processed = 0
        images_with_defects = 0
        per_class: Counter = Counter()

        for image_path in all_images:
            image_start = time.perf_counter()

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {image_path.name}", file=sys.stderr)
                continue

            original_h, original_w = bgr.shape[:2]
            lb = letterbox(bgr, cfg.input_size)

            infer_start = time.perf_counter()
            outputs = runner.run([to_bgr_tensor(lb.image)], timeout_ms=cfg.timeout_ms)
            infer_end = time.perf_counter()

            # Detections arrive in letterboxed coordinates; draw them on the original frame.
            payload = extract_bbox_payload(outputs)
            detections = (
                parse_bbox_payload(
                    payload,
                    cfg.input_size,
                    cfg.input_size,
                    cfg.score_threshold,
                    cfg.max_detections,
                )
                if payload
                else []
            )
            detections = to_source_coordinates(detections, lb, original_w, original_h)

            out_path = output_path_for(image_path, cfg.output_dir)
            if cfg.overlay:
                draw_boxes(bgr, detections, labels)
                cv2.imwrite(str(out_path), bgr)
            image_end = time.perf_counter()

            counts = Counter(class_name(d["class_id"], labels) for d in detections)
            per_class.update(counts)
            processed += 1
            if detections:
                images_with_defects += 1

            progress = f"[{processed}/{len(all_images)}] {image_path.name}"
            if cfg.overlay:
                progress += f" -> {out_path.name}"
            print(
                f"{progress} ({len(detections)} defects) {dict(counts)}",
                flush=True,
            )
            if cfg.profile:
                print(
                    f"[PROFILE] {image_path.name}: "
                    f"inference={(infer_end - infer_start) * 1000:.1f}ms "
                    f"overlay+save={(image_end - infer_end) * 1000:.1f}ms "
                    f"total={(image_end - image_start) * 1000:.1f}ms",
                    flush=True,
                )

        runner.close()
    except Exception as error:  # noqa: BLE001 - report any runtime failure to the caller
        print(f"Error: {error}", file=sys.stderr)
        return 4

    elapsed = time.perf_counter() - pipeline_start
    print(
        f"Done: {processed}/{len(all_images)} images in {elapsed:.2f}s | "
        f"images_with_defects={images_with_defects} total_defects={sum(per_class.values())}"
    )
    if per_class:
        print(f"Per-class totals: {dict(per_class)}")
    return 0 if processed > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
