"""Unit tests for the YOLO26 object detector: the checks the application leaves to its config, and
the image folder and annotated images it handles. tests/cpp/test_unit.cpp runs the same checks in
C++.
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

main_module = load_example_main(EXAMPLE_DIR, "yolo26_object_detector_main")
annotate = main_module.annotate
images = main_module.images

pytestmark = pytest.mark.unit


def box(x1, y1, x2, y2, score, class_id):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


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
    assert images.image_paths(input_dir)
    assert input_dir != output_dir
    assert output_dir.is_relative_to(APPS_ROOT / "sandbox")


def test_image_paths_lists_only_images_in_name_order(tmp_path):
    for name in ("b.PNG", "a.jpg", "c.bmp", "notes.txt", "d.jpeg"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "e.jpg").mkdir()
    assert [path.name for path in images.image_paths(tmp_path)] == [
        "a.jpg",
        "b.PNG",
        "c.bmp",
        "d.jpeg",
    ]


def test_image_paths_rejects_a_folder_without_images(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"")
    with pytest.raises(ValueError, match="input.source has no"):
        images.image_paths(tmp_path)


def test_output_name_marks_the_yolo26_result():
    assert annotate.output_name(Path("in/street.jpg")) == "street_yolo26.png"


def test_draw_detections_draws_the_box_in_its_class_color_on_a_copy():
    image = np.zeros((60, 80, 3), np.uint8)
    annotated = annotate.draw_detections(
        image, [box(9.6, 20.4, 50.0, 55.0, 0.9, 1)], ["person", "bicycle"]
    )
    assert not image.any()
    # Class 1 is blue in BGR. x1 = 9.6 rounds to 10, and a 2-pixel line there covers columns 9
    # to 11; truncating to 9 would cover 8 to 10.
    assert tuple(annotated[40, 11]) == (255, 0, 0)
    assert tuple(annotated[40, 8]) == (0, 0, 0)
    assert tuple(annotated[40, 30]) == (0, 0, 0)
