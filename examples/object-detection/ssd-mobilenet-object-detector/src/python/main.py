"""Detect objects in a folder of images with SSD-MobileNet, saved as annotated PNGs and JSON.

The model runs on one image at a time. Each image is saved with its boxes drawn, and every box
goes into one JSON report, unless app.detections_json is empty.
"""

from __future__ import annotations

import argparse
import sys
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
from sima_examples.image_folder import image_paths
from sima_examples.labels import load_labels
from utils import annotate

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(config: Config, num_classes: int, max_detections: int) -> pyneat.ModelOptions:
    """OpenCV BGR images in, boxes out of Core's SSD decoder."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    # The SSD decoder needs a Stretch resize; Core's default is Letterbox.
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    # TensorFlow's SSD-MobileNet V1 and V2 were trained on pixels scaled to -1 to 1: mean 0.5 and
    # stddev 0.5 per channel, on pixels scaled to 0-1.
    options.preprocess.normalize.mean = [0.5, 0.5, 0.5]
    options.preprocess.normalize.stddev = [0.5, 0.5, 0.5]
    options.preprocess.normalize.has_explicit_stats = True
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    options.decode_type = pyneat.BoxDecodeType.Ssd
    options.num_classes = num_classes
    options.score_threshold = config.number("inference.min_score")
    options.nms_iou_threshold = config.number("inference.nms_iou")
    options.top_k = max_detections
    return options


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def image_tensor(image: np.ndarray) -> pyneat.Tensor:
    """An OpenCV image as a BGR tensor in EV74 memory, where the preprocessing reads it."""
    return pyneat.Tensor.from_numpy(
        image, copy=True, image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.EV74
    )


def detect(runner: pyneat.ModelRunner, image: np.ndarray, max_detections: int) -> list:
    """The boxes the model finds in one image, in the image's pixels."""
    bbox = runner.run([image_tensor(image)])[0]
    height, width = image.shape[:2]
    return pyneat.detections.decode_bbox_tensor(bbox, width, height, max_detections, False).boxes


def annotate_images(
    runner: pyneat.ModelRunner,
    paths: list[Path],
    labels: list[str],
    max_detections: int,
    output_dir: Path,
) -> list[dict]:
    """Save each image with its boxes drawn to `output_dir`; return each image's record."""
    records = []
    for index, path in enumerate(paths, start=1):
        image = read_image(path)
        boxes = detect(runner, image, max_detections)
        output_path = output_dir / annotate.output_name(path)
        if not cv2.imwrite(str(output_path), annotate.draw_detections(image, boxes, labels)):
            raise OSError(f"cannot write {output_path}")
        records.append(annotate.detections_record(path, image, boxes, labels))
        print(f"[{index}/{len(paths)}] {path.name} -> {output_path.name} ({len(boxes)} detections)")
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    # One label per class the model scores, background first, so they also give the class count.
    labels = load_labels(Path(config.string("model.labels")))
    paths = image_paths(Path(config.string("input.source")))
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(str(model_path), model_options(config, len(labels), max_detections))
    # The first image seeds the build, which plans the route for images like it.
    runner = model.build([image_tensor(read_image(paths[0]))])
    output_dir = Path(config.string("output.dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        records = annotate_images(runner, paths, labels, max_detections, output_dir)
    finally:
        runner.close()

    # An empty app.detections_json turns the report off.
    report_path = config.string("app.detections_json")
    if report_path:
        annotate.write_report(Path(report_path), records)
        print(f"Done: {len(records)} images processed, detections in {report_path}")
    else:
        print(f"Done: {len(records)} images processed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
