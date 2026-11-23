"""Find defects on printed circuit boards in a folder of images with a custom YOLO26n model.

Each image goes to the model at its own size. Core letterboxes it on the device and returns the
boxes in the image's pixels, which are drawn on a copy saved to output.dir.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyneat

# The helpers the examples share live in <apps root>/support/python.
for apps_root in Path(__file__).resolve().parents:
    if (apps_root / "support" / "python" / "sima_examples").is_dir():
        sys.path.insert(0, str(apps_root / "support" / "python"))
        break
else:
    raise SystemExit("support/python not found above this example; reinstall the Apps package")

from sima_examples.config import Config
from sima_examples.labels import load_labels
from utils import images, overlay

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
# Added to each annotated image's name: board.jpg is saved as board_pcb.jpg.
OUTPUT_TAG = "_pcb"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(
    config: Config, labels: list[str], max_size: tuple[int, int], max_detections: int
) -> pyneat.ModelOptions:
    """YOLO26 on BGR images of any size up to max_size."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.input_max_width = max_size[0]
    options.preprocess.input_max_height = max_size[1]
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = config.number("inference.min_score")
    options.nms_iou_threshold = config.number("inference.nms_iou")
    options.top_k = max_detections
    # Loading fails when the label file's class count differs from the model's.
    options.num_classes = len(labels)
    return options


def run_options() -> pyneat.RunOptions:
    options = pyneat.RunOptions()
    # Realtime is the only preset that applies a new input size on the first frame; the others
    # hold a frame until a second one of the same size arrives, which one image at a time never
    # sends. Its KeepLatest overflow policy drops nothing here, because runner.run waits for each
    # result before the next image.
    options.preset = pyneat.RunPreset.Realtime
    # 8 queue slots, not the preset's 2. One image is in flight at a time, so a smaller depth may
    # do as well, but that is not measured.
    options.queue_depth = 8
    return options


def bgr_tensor(image: np.ndarray) -> pyneat.Tensor:
    """A BGR image as a tensor in the memory the model's preprocessing reads."""
    return pyneat.Tensor.from_numpy(
        image, copy=True, image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.EV74
    )


def detect_defects(runner: pyneat.ModelRunner, image: np.ndarray, max_detections: int) -> list:
    """The defect boxes in one image, in its pixels."""
    outputs = runner.run([bgr_tensor(image)])
    height, width = image.shape[:2]
    return pyneat.detections.decode_bbox_tensor(
        outputs[0], width, height, max_detections, strict=True
    ).boxes


def format_counts(counts: Counter) -> str:
    return "{" + ", ".join(f"{name}: {count}" for name, count in sorted(counts.items())) + "}"


def annotate_images(
    runner: pyneat.ModelRunner,
    image_paths: list[Path],
    labels: list[str],
    output_dir: Path,
    max_size: tuple[int, int],
    max_detections: int,
) -> int:
    """Save an annotated copy of each image and print its defects; return the failed count."""
    per_class: Counter = Counter()
    boards_with_defects = 0
    failed = 0
    start = time.perf_counter()
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        problem = images.image_problem(image, max_size)
        if problem:
            print(f"[ERR] {image_path.name}: {problem}", file=sys.stderr)
            failed += 1
            continue
        boxes = detect_defects(runner, image, max_detections)
        overlay.draw_boxes(image, boxes, labels)
        output_path = output_dir / f"{image_path.stem}{OUTPUT_TAG}{image_path.suffix}"
        if not cv2.imwrite(str(output_path), image):
            print(f"[ERR] {output_path}: could not be written", file=sys.stderr)
            failed += 1
            continue
        counts = Counter(labels[box.class_id] for box in boxes)
        per_class.update(counts)
        boards_with_defects += bool(boxes)
        print(
            f"[{index}/{len(image_paths)}] {image_path.name} -> {output_path.name} "
            f"({len(boxes)} defects) {format_counts(counts)}",
            flush=True,
        )
    print(
        f"Done: {len(image_paths) - failed}/{len(image_paths)} images in "
        f"{time.perf_counter() - start:.2f} s | images_with_defects={boards_with_defects} "
        f"total_defects={sum(per_class.values())} failed={failed}"
    )
    print(f"Per-class totals: {format_counts(per_class)}")
    return failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    labels = load_labels(Path(config.string("model.labels")))
    image_paths = images.find_images(Path(config.string("input.source")))
    output_dir = Path(config.string("output.dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    # Used by the model, the build seed, and the size check of each image.
    max_size = (config.integer("app.input_max_width"), config.integer("app.input_max_height"))
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config, labels, max_size, max_detections))
    # The seed fixes the route's input caps, so it is an image of the largest accepted size.
    seed = np.zeros((max_size[1], max_size[0], 3), dtype=np.uint8)
    runner = model.build([bgr_tensor(seed)], run_options=run_options())
    print(f"source={config.string('input.source')} images={len(image_paths)} output={output_dir}")
    try:
        failed = annotate_images(
            runner, image_paths, labels, output_dir, max_size, max_detections
        )
    finally:
        runner.close()
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
