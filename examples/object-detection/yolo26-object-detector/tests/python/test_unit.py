"""Unit tests for the YOLO26 object detector: the checks the application leaves to its config, and
the annotated names it handles. tests/cpp/test_unit.cpp runs the same checks in C++.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]

main_module = load_example_main(EXAMPLE_DIR, "yolo26_object_detector_main")
output_name = main_module.output_name
image_paths = main_module.image_paths

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_model_labels_and_folders():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolo26m_w16a16_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    # YOLO26 scores the 80 COCO classes, IDs 0 to 79.
    labels = main_module.load_labels(APPS_ROOT / config.string("model.labels"))
    assert len(labels) == 80
    assert labels[0] == "person" and labels[79] == "toothbrush"
    # Core's YOLO26 box decode accepts a score threshold below 1 only, and at 0 it keeps noise.
    assert 0 < config.number("inference.min_score") < 1
    # At 0, Core's YOLO26 decoder runs no NMS.
    assert 0 < config.number("inference.nms_iou") <= 1
    assert config.integer("inference.max_detections") > 0
    # The annotated PNGs go to their own folder under sandbox/, never into the input folder.
    input_dir = (APPS_ROOT / config.string("input.source")).resolve()
    output_dir = (APPS_ROOT / config.string("output.dir")).resolve()
    assert image_paths(input_dir)
    assert input_dir != output_dir
    assert output_dir.is_relative_to(APPS_ROOT / "sandbox")


def test_output_name_marks_the_yolo26_result():
    assert output_name(Path("in/street.jpg"), "_yolo26") == "street_yolo26.png"
