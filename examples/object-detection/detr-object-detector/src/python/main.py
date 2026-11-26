"""Detect objects in a folder of images with DETR, saved as annotated PNGs.

The model runs on one image at a time and decodes DETR's boxes itself. Each image is saved with
its boxes drawn.
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
from sima_examples.detection_overlay import draw_detections
from sima_examples.image_folder import image_paths
from sima_examples.image_io import read_image
from sima_examples.labels import load_labels
from utils import annotate, images

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def model_options(
    config: Config, max_width: int, max_height: int, max_detections: int
) -> pyneat.ModelOptions:
    """OpenCV BGR images in, boxes out of Core's DETR decoder."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    # Neat rejects images larger than 1920x1080 unless these limits are raised.
    options.preprocess.input_max_width = max_width
    options.preprocess.input_max_height = max_height
    # Core's DETR decoder needs a Stretch resize; Core's default is Letterbox.
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    # The registry calibrates and scores DETR on Pillow bilinear resizes, which average every
    # source pixel when shrinking; plain bilinear samples only the nearest four.
    options.preprocess.resize.interpolation = pyneat.ResizeInterpolation.BilinearAntialiased
    options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    options.decode_type = pyneat.BoxDecodeType.Detr
    options.score_threshold = config.number("inference.min_score")
    options.top_k = max_detections
    return options


def image_tensor(image: np.ndarray) -> pyneat.Tensor:
    """An OpenCV image as a BGR tensor in EV74 memory, where the preprocessing reads it."""
    return pyneat.Tensor.from_numpy(
        image, copy=True, image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.EV74
    )


def detect(runner: pyneat.ModelRunner, image: np.ndarray, max_detections: int) -> list:
    """The boxes the model finds in one image, in the image's pixels."""
    bbox = runner.run([image_tensor(image)])[0]
    height, width = image.shape[:2]
    # strict fails on anything but a BBOX payload, such as raw model heads.
    return pyneat.detections.decode_bbox_tensor(bbox, width, height, max_detections, True).boxes


def annotate_images(
    runner: pyneat.ModelRunner,
    paths: list[Path],
    labels: list[str],
    max_detections: int,
    output_dir: Path,
) -> None:
    """Save each image with its boxes drawn to `output_dir`."""
    for index, path in enumerate(paths, start=1):
        image = read_image(path)
        boxes = detect(runner, image, max_detections)
        output_path = output_dir / annotate.output_name(path)
        if not cv2.imwrite(str(output_path), draw_detections(image, boxes, labels)):
            raise OSError(f"cannot write {output_path}")
        print(f"[{index}/{len(paths)}] {path.name}: {len(boxes)} detections -> {output_path.name}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.load(args.config)
    # Used by both the model and the box decode.
    max_detections = config.integer("inference.max_detections")
    # DETR's class IDs index the labels directly.
    labels = load_labels(Path(config.string("model.labels")))
    paths = image_paths(Path(config.string("input.source")))
    max_width, max_height = images.largest_image_size(paths)
    model_path = args.config.absolute().parent / config.string("model.path")
    model = pyneat.Model(
        str(model_path), model_options(config, max_width, max_height, max_detections)
    )
    # The first image seeds the build, which plans the route for images like it.
    runner = model.build([image_tensor(read_image(paths[0]))])
    output_dir = Path(config.string("output.dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        annotate_images(runner, paths, labels, max_detections, output_dir)
    finally:
        runner.close()

    print(f"Done: {len(paths)} images processed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:  # noqa: BLE001 - report the cause in one line, not a traceback
        print(f"[ERR] {error}", file=sys.stderr)
        sys.exit(1)
