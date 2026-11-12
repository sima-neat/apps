"""DETR folder object detection through Neat's model-managed BoxDecodeType.Detr decoder."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
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


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(DETR_COCO_LABELS):
        label = DETR_COCO_LABELS[class_id]
        if label != "N/A":
            return label
    return f"class_{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_id) % len(BOX_COLORS)]


def build_model_options(config: Config) -> pyneat.ModelOptions:
    """Stretch, ImageNet-normalize, and decode DETR on the model route."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    # Core's classic DETR decoder requires Stretch (Core docs/reference/boxdecode_decode_types.md).
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    options.decode_type = pyneat.BoxDecodeType.Detr
    options.score_threshold = config.confidence_threshold
    options.top_k = config.max_draw
    return options


def read_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image_bgr


def image_to_tensor(image_bgr: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(image_bgr, dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def detect(runner: pyneat.ModelRunner, image_bgr: np.ndarray, config: Config) -> list[pyneat.Box]:
    """Run one image through the model and return its boxes in original-image pixels."""
    outputs = runner.run([image_to_tensor(image_bgr)])
    if len(outputs) != 1:
        raise RuntimeError(f"expected one BBOX tensor, got {len(outputs)}")
    height, width = image_bgr.shape[:2]
    boxes = pyneat.detections.decode_bbox_tensor(
        outputs[0], width, height, config.max_draw, False
    ).boxes
    if config.person_only:
        return [box for box in boxes if box.class_id == PERSON_CLASS_ID]
    return list(boxes)


def draw_detections(image_bgr: np.ndarray, boxes: list[pyneat.Box]) -> np.ndarray:
    annotated = image_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = (round(v) for v in (box.x1, box.y1, box.x2, box.y2))
        color = class_color(box.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{class_name(box.class_id)} {box.score:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated


def annotate_images(runner: pyneat.ModelRunner, image_paths: list[Path], config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_paths, start=1):
        image_bgr = read_image(image_path)
        boxes = detect(runner, image_bgr, config)

        output_path = config.output_dir / f"{image_path.stem}_detr.png"
        if not cv2.imwrite(str(output_path), draw_detections(image_bgr, boxes)):
            raise RuntimeError(f"Failed to write: {output_path}")
        progress = f"[{index}/{len(image_paths)}] {image_path.name}"
        print(f"{progress}: {len(boxes)} detections -> {output_path.name}")
    print(f"Done: {len(image_paths)} images processed")


def profile_model_run(runner: pyneat.ModelRunner, image_path: Path, config: Config) -> None:
    """Time preprocess, inference, and decode on one image after one untimed warm-up run."""
    image_bgr = read_image(image_path)
    detect(runner, image_bgr, config)

    run_times_s = []
    for _ in range(config.num_runs):
        start = time.perf_counter()
        boxes = detect(runner, image_bgr, config)
        run_times_s.append(time.perf_counter() - start)

    times = np.array(run_times_s)
    print(f"Profiling over {len(times)} runs (image='{image_path}'):")
    print(
        f"  Model run (preprocess+infer+decode): mean={times.mean():.6f}s, "
        f"min={times.min():.6f}s, max={times.max():.6f}s, FPS={len(times) / times.sum():.3f}"
    )
    print(f"Last run detections: {len(boxes)}")
    for i, box in enumerate(boxes[:20]):
        print(
            f"  [{i}] class={class_name(box.class_id)}({box.class_id}) score={box.score:.4f} "
            f"box=[{box.x1:.1f},{box.y1:.1f},{box.x2:.1f},{box.y2:.1f}]"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="DETR folder object detection example")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except OSError as exc:
        print(f"failed to open config file: {args.config} ({exc.strerror})", file=sys.stderr)
        return 2
    except (yaml.YAMLError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    if not config.model_path.is_file():
        print(f"Model file does not exist: {config.model_path}", file=sys.stderr)
        return 2
    if not config.input_dir.is_dir():
        print(f"Input directory does not exist: {config.input_dir}", file=sys.stderr)
        return 2
    image_paths = sorted(p for p in config.input_dir.iterdir() if p.is_file() and is_image(p))
    if not image_paths:
        print(f"No images found in {config.input_dir}", file=sys.stderr)
        return 3

    # Imported after config validation so config errors are reported without a runtime install.
    global cv2, np, pyneat
    import cv2
    import numpy as np
    import pyneat

    try:
        model = pyneat.Model(str(config.model_path), build_model_options(config))
        runner = model.build([image_to_tensor(read_image(image_paths[0]))])
    except Exception as exc:
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 3

    try:
        if config.profile:
            profile_model_run(runner, image_paths[0], config)
        else:
            annotate_images(runner, image_paths, config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 4
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
