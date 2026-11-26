"""Unit tests for the DETR object detector: the checks the application leaves to its config, and
the image size and annotated names it handles. tests/cpp/test_unit.cpp runs the same checks in
C++, and also the cases for the shared C++ image_folder, image_io, and detection_overlay.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]

main_module = load_example_main(EXAMPLE_DIR, "detr_object_detector_main")
annotate = main_module.annotate
image_paths = main_module.image_paths
images = main_module.images

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_model_labels_and_folders():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "detr_resnet_50_w16a16_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    # DETR scores the 91 COCO class IDs; ID 0 and the other unused IDs are N/A.
    labels = main_module.load_labels(APPS_ROOT / config.string("model.labels"))
    assert len(labels) == 91
    assert labels[0] == "N/A" and labels[1] == "person" and labels[90] == "toothbrush"
    # Neat's DETR decoder keeps the package's own threshold and cap when either is 0.
    assert 0 < config.number("inference.min_score") <= 1
    assert config.integer("inference.max_detections") > 0
    # The annotated PNGs go to their own folder under sandbox/, never into the input folder.
    input_dir = (APPS_ROOT / config.string("input.source")).resolve()
    output_dir = (APPS_ROOT / config.string("output.dir")).resolve()
    assert image_paths(input_dir)
    assert input_dir != output_dir
    assert output_dir.is_relative_to(APPS_ROOT / "sandbox")


def test_largest_image_size_takes_the_widest_and_the_tallest_image(tmp_path):
    # Neat rejects any image above its input limits, so they must fit the widest and the tallest
    # image, even when those are different images.
    landscape, portrait = tmp_path / "landscape.png", tmp_path / "portrait.png"
    assert cv2.imwrite(str(landscape), np.zeros((30, 64, 3), np.uint8))
    assert cv2.imwrite(str(portrait), np.zeros((50, 40, 3), np.uint8))
    assert images.largest_image_size([landscape, portrait]) == (64, 50)


def test_output_name_marks_the_detr_result():
    assert annotate.output_name(Path("in/street.jpg")) == "street_detr.png"
