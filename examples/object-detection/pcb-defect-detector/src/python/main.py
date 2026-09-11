"""PCB defect detection over a folder of images using pyneat.

Images of any resolution go straight to the model: Core letterboxes them to the
packaged input size on device. The compiled YOLO26n model pack owns the whole
Neat path:

    letterbox -> color convert + normalize -> MLA inference -> YOLO26 box decode

decode_bbox_tensor returns boxes already in source-image coordinates, so the
application performs no geometry of its own. One annotated image is written per
input image.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
import math
from dataclasses import dataclass, replace
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# Appended to the stem of every annotated image.
OUTPUT_TAG = "_pcb"
# Floor for the one-off priming run of the graph seed.
WARMUP_TIMEOUT_MS = 30000
# Ingress capacity of the graph; boards up to this size share one graph.
DEFAULT_INPUT_MAX_WIDTH = 3840
DEFAULT_INPUT_MAX_HEIGHT = 2160

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
    input_max_width: int
    input_max_height: int
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


def format_counts(counts) -> str:
    """Render per-class counts the same way the C++ twin does."""
    return "{" + ", ".join(f"{name}: {n}" for name, n in sorted(counts.items())) + "}"


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def _section(raw: dict, name: str) -> dict:
    """One config section, rejecting a non-mapping the way the C++ parser does."""
    section = raw.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    return section


def _int(section: dict, key: str, default: int, name: str) -> int:
    """Integer config value; a fractional or non-numeric value is an error."""
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _float(section: dict, key: str, default: float, name: str) -> float:
    """Floating-point config value; a non-numeric value is an error."""
    value = section.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    return float(value)


def build_app_config(raw: dict) -> AppConfig:
    """Map a parsed config mapping onto AppConfig without validating it."""
    model_cfg = _section(raw, "model")
    io_cfg = _section(raw, "io")
    decode_cfg = _section(raw, "decode")
    runtime_cfg = _section(raw, "runtime")
    output_cfg = _section(raw, "output")

    return AppConfig(
        model_path=str(model_cfg.get("path", "")),
        labels_path=Path(str(model_cfg.get(
            "labels",
            "examples/object-detection/pcb-defect-detector/src/common/pcb_label.txt",
        ))),
        input_max_width=_int(model_cfg, "input_max_width", DEFAULT_INPUT_MAX_WIDTH,
                             "model.input_max_width"),
        input_max_height=_int(model_cfg, "input_max_height", DEFAULT_INPUT_MAX_HEIGHT,
                              "model.input_max_height"),
        input_dir=Path(str(io_cfg.get("input_dir", "assets/datasets/pcb"))),
        output_dir=Path(str(io_cfg.get("output_dir", "sandbox/pcb-defect-detector"))),
        score_threshold=_float(decode_cfg, "score_threshold", 0.25, "decode.score_threshold"),
        nms_iou=_float(decode_cfg, "nms_iou", 0.45, "decode.nms_iou"),
        max_detections=_int(decode_cfg, "max_detections", 300, "decode.max_detections"),
        timeout_ms=_int(runtime_cfg, "timeout_ms", 8000, "runtime.timeout_ms"),
        num_runs=_int(runtime_cfg, "num_runs", 1, "runtime.num_runs"),
        queue_depth=_int(runtime_cfg, "queue_depth", 8, "runtime.queue_depth"),
        profile=bool(runtime_cfg.get("profile", False)),
        overlay=bool(output_cfg.get("overlay", True)),
    )


def validate_config(cfg: AppConfig) -> None:
    """Raise ValueError when a resolved configuration cannot be run."""
    if not cfg.model_path:
        raise ValueError("model.path must be set to a compiled model package")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must point to a labels file")
    if cfg.input_max_width < 1:
        raise ValueError(f"model.input_max_width must be >= 1, got {cfg.input_max_width}")
    if cfg.input_max_height < 1:
        raise ValueError(f"model.input_max_height must be >= 1, got {cfg.input_max_height}")
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
    return DEFECT_COLORS[abs(class_id) % len(DEFECT_COLORS)]


def round_half_up(value: float) -> int:
    """Round like C++ std::round. Python's round() is banker's rounding, which
    disagrees on exact .5 values and shifts drawn boxes by a pixel."""
    return int(math.floor(value + 0.5))


def output_path_for(image_path: Path, output_dir: Path) -> Path:
    """Annotated-output path for one input image."""
    return output_dir / f"{image_path.stem}{OUTPUT_TAG}{image_path.suffix}"


def decode_detections(outputs, img_w: int, img_h: int, cfg: AppConfig) -> list[dict]:
    """Decode the model's BBOX payload into source-image coordinates.

    Raises RuntimeError if the response is not a usable detection result;
    returns [] when a valid response found nothing.
    """
    if not outputs:
        raise RuntimeError("model returned no detection tensors")
    if len(outputs) != 1:
        raise RuntimeError(
            f"expected one BBOX tensor from model-managed BoxDecode, got {len(outputs)}"
        )

    bbox_tensor = outputs[0]
    try:
        detection_format = pyneat.detections.read_detection_format(bbox_tensor)
    except Exception:  # noqa: BLE001 - a non-detection tensor has no format tag
        detection_format = ""
    if not detection_format or not pyneat.detections.format_is_bbox(detection_format):
        raise RuntimeError(
            f"model returned no BBOX detection tensor (format: '{detection_format or '<none>'}'); "
            "the model package must expose YOLO26 BoxDecode output"
        )

    # strict=True rejects a truncated or over-long payload instead of decoding it
    # as zero boxes. expected_topk is 0 because strict also throws on it; the cap
    # is applied below.
    try:
        result = pyneat.detections.decode_bbox_tensor(bbox_tensor, img_w, img_h, 0, True)
    except Exception as error:  # noqa: BLE001 - Core raises a plain runtime error
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
        if len(detections) >= cfg.max_detections:
            break
    return detections


def draw_boxes(bgr, detections: list[dict], labels: list[str], thickness: int = 2) -> None:
    """Draw class-colored defect boxes and labels on a BGR image in place."""
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
    except yaml.YAMLError as error:
        print(f"Error: invalid YAML in {args.config}: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Error: failed to read config {args.config}: {error}", file=sys.stderr)
        return 2
    except (ValueError, AttributeError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    try:
        cfg = build_app_config(raw)
        # Overrides are applied to the resolved config, not injected into the raw
        # mapping, so a malformed value in the file is still reported.
        if args.score is not None:
            cfg = replace(cfg, score_threshold=args.score)
        if args.nms is not None:
            cfg = replace(cfg, nms_iou=args.nms)
        validate_config(cfg)
        labels = load_labels(cfg.labels_path)
    except (ValueError, TypeError, AttributeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if args.validate_config_only:
        print(
            f"[validate] model={cfg.model_path} classes={len(labels)} "
            f"score_threshold={cfg.score_threshold:.2f} nms_iou={cfg.nms_iou:.2f} "
            f"max_detections={cfg.max_detections} timeout_ms={cfg.timeout_ms} "
            f"num_runs={cfg.num_runs} queue_depth={cfg.queue_depth}"
        )
        print("[validate] configuration OK")
        return 0

    if not cfg.input_dir.is_dir():
        print(f"Input directory does not exist: {cfg.input_dir}", file=sys.stderr)
        return 2

    try:
        images = discover_images(cfg.input_dir)
    except OSError as error:
        print(f"Error: cannot read {cfg.input_dir}: {error}", file=sys.stderr)
        return 2
    if not images:
        print(f"No images found in {cfg.input_dir}", file=sys.stderr)
        return 3

    global cv2, np, pyneat
    import cv2
    import numpy as np
    import pyneat

    try:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Error: cannot create {cfg.output_dir}: {error}", file=sys.stderr)
        return 2

    print(f"Model: {cfg.model_path}", flush=True)
    print(f"Found {len(images)} images in {cfg.input_dir}", flush=True)

    # Inputs we were asked to inspect but could not complete.
    failed_images: list[str] = []
    # Declared before the try so the abort handler can always report progress,
    # even when the failure happens before the loop starts.
    all_images: list[Path] = []
    processed = 0
    runner = None
    images_with_defects = 0
    per_class: Counter = Counter()
    # The image in flight when a systemic failure aborts the batch, so the summary
    # names it rather than reporting failed=0.
    aborted_on = ""


    try:
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
        model = pyneat.Model(cfg.model_path, opt)

        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = cfg.queue_depth
        # Realtime applies new input caps on the first frame. The other presets
        # wait for a second frame at the same size, which a folder never sends.
        run_opt.preset = pyneat.RunPreset.Realtime

        # The build seed fixes the graph's input caps, so it is a frame at the
        # configured capacity rather than any particular board.
        seed = np.full((cfg.input_max_height, cfg.input_max_width, 3), 114, dtype=np.uint8)

        runner = model.build(
            [to_bgr_tensor(seed)],
            route_options=pyneat.ModelRouteOptions(),
            run_options=run_opt,
        )
        # One-off graph settling and first-touch allocation, which
        # runtime.timeout_ms does not cover.
        runner.run([to_bgr_tensor(seed)], timeout_ms=max(cfg.timeout_ms, WARMUP_TIMEOUT_MS))


        all_images = images * cfg.num_runs
        if cfg.num_runs > 1:
            print(
                f"Looping {cfg.num_runs}x over {len(images)} images "
                f"({len(all_images)} total)",
                flush=True,
            )

        pipeline_start = time.perf_counter()

        for image_path in all_images:
            aborted_on = image_path.name
            image_start = time.perf_counter()

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                # An input we were asked to inspect and could not: record it so
                # the summary and the exit code stay honest.
                print(f"Failed to read: {image_path.name}", file=sys.stderr)
                failed_images.append(image_path.name)
                continue

            original_h, original_w = bgr.shape[:2]
            if original_w > cfg.input_max_width or original_h > cfg.input_max_height:
                print(f"Image {original_w}x{original_h} exceeds model.input_max_width/height "
                      f"({cfg.input_max_width}x{cfg.input_max_height}): {image_path.name}",
                      file=sys.stderr)
                failed_images.append(image_path.name)
                continue

            infer_start = time.perf_counter()
            outputs = runner.run([to_bgr_tensor(bgr)], timeout_ms=cfg.timeout_ms)
            infer_end = time.perf_counter()

            # Raises rather than reporting a clean board if the model produced no
            # usable detection output; a valid empty result returns [].
            detections = decode_detections(outputs, original_w, original_h, cfg)

            out_path = output_path_for(image_path, cfg.output_dir)
            if cfg.overlay:
                draw_boxes(bgr, detections, labels)
                # imwrite reports failure by return value, not by raising.
                try:
                    written = cv2.imwrite(str(out_path), bgr)
                except cv2.error:  # encoder failure, not a batch-wide problem
                    written = False
                if not written:
                    print(f"Failed to write: {out_path}", file=sys.stderr)
                    failed_images.append(image_path.name)
                    continue
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
                f"{progress} ({len(detections)} defects) {format_counts(counts)}",
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
            aborted_on = ""

    except Exception as error:  # noqa: BLE001 - report any runtime failure to the caller
        # A systemic failure repeats on every image, so stop the batch but still
        # report what completed and what failed.
        sys.stdout.flush()
        if aborted_on:
            failed_images.append(aborted_on)
        print(f"Error: {error}", file=sys.stderr)
        print(
            f"Aborted after {processed}/{len(all_images) or len(images)} images | "
            f"images_with_defects={images_with_defects} "
            f"total_defects={sum(per_class.values())} failed={len(failed_images)}",
            file=sys.stderr,
        )
        if failed_images:
            print(f"Previously failed: {', '.join(failed_images[:5])}"
                  f"{', ...' if len(failed_images) > 5 else ''}", file=sys.stderr)
        return 4

    finally:
        if runner is not None:
            runner.close()

    elapsed = time.perf_counter() - pipeline_start
    print(
        f"Done: {processed}/{len(all_images)} images in {elapsed:.2f}s | "
        f"images_with_defects={images_with_defects} total_defects={sum(per_class.values())} "
        f"failed={len(failed_images)}"
    )
    if per_class:
        print(f"Per-class totals: {format_counts(per_class)}")
    sys.stdout.flush()

    # A batch that skipped inputs is not a success.
    if failed_images:
        shown = ", ".join(failed_images[:5])
        suffix = ", ..." if len(failed_images) > 5 else ""
        print(
            f"Error: {len(failed_images)} of {len(all_images)} image(s) could not be "
            f"processed or saved: {shown}{suffix}",
            file=sys.stderr,
        )
        return 4
    if processed != len(all_images):
        print(
            f"Error: processed {processed} of {len(all_images)} images",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
