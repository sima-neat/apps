"""Unit tests for the SSD-MobileNet object detector: the checks the application leaves to its
config, and the annotated images and report it writes. tests/cpp/test_unit.cpp runs the same
checks in C++.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]

main_module = load_example_main(EXAMPLE_DIR, "ssd_mobilenet_object_detector_main")
annotate = main_module.annotate

pytestmark = pytest.mark.unit


def box(x1, y1, x2, y2, score, class_id):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


def test_shipped_config_names_a_supported_model_labels_and_folders():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    # The SSD decoder scores 91 classes: the background, then the 90 COCO IDs.
    labels = main_module.load_labels(APPS_ROOT / config.string("model.labels"))
    assert len(labels) == 91
    assert labels[0] == "background"
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    # The annotated PNGs go to their own folder, never into the input folder.
    input_dir = (APPS_ROOT / config.string("input.source")).resolve()
    output_dir = (APPS_ROOT / config.string("output.dir")).resolve()
    assert input_dir != output_dir
    assert output_dir.is_relative_to(APPS_ROOT / "sandbox")
    report = (APPS_ROOT / config.string("app.detections_json")).resolve()
    assert report.is_relative_to(APPS_ROOT / "sandbox")
    assert not report.is_relative_to(output_dir)


def test_output_name_keeps_the_source_extension():
    assert annotate.output_name(Path("in/frame.jpg")) == "frame_jpg.png"
    assert annotate.output_name(Path("in/frame.PNG")) == "frame_PNG.png"


def test_detections_record_lists_each_box_with_its_label():
    image = np.zeros((48, 64, 3), np.uint8)
    record = annotate.detections_record(
        Path("in/street.jpg"), image, [box(1.0, 2.0, 30.0, 40.0, 0.75, 3)], ["bg", "a", "b", "car"]
    )
    assert record == {
        "image": "street.jpg",
        "width": 64,
        "height": 48,
        "detections": [
            {"class_id": 3, "label": "car", "score": 0.75, "box": [1.0, 2.0, 30.0, 40.0]}
        ],
    }


def test_draw_detections_draws_the_box_in_its_class_color_on_a_copy():
    image = np.zeros((60, 80, 3), np.uint8)
    annotated = annotate.draw_detections(image, [box(9.6, 20.4, 50.0, 55.0, 0.9, 1)], ["bg", "a"])
    assert not image.any()
    # Class 1 is blue in BGR. x1 = 9.6 rounds to 10, and a 2-pixel line there covers columns 9
    # to 11; truncating to 9 would cover 8 to 10.
    assert tuple(annotated[40, 11]) == (255, 0, 0)
    assert tuple(annotated[40, 8]) == (0, 0, 0)
    assert tuple(annotated[40, 30]) == (0, 0, 0)
