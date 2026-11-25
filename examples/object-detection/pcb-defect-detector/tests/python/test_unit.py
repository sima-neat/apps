"""Unit tests for the PCB defect detector: the checks the application leaves to its config, and
its own logic. tests/cpp/test_unit.cpp runs the same checks in C++.
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

main_module = load_example_main(EXAMPLE_DIR, "pcb_defect_detector_main")
images = main_module.images
overlay = main_module.overlay

pytestmark = pytest.mark.unit


def test_shipped_config_names_the_model_labels_and_folders():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolo26n_plc_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    # The model's class IDs index the labels, and each class has a color.
    labels = main_module.load_labels(APPS_ROOT / config.string("model.labels"))
    assert labels == ["missing_hole", "mouse_bite", "open_circuit", "short", "spur",
                      "spurious_copper"]
    assert len(overlay.CLASS_COLORS) == len(labels)
    # The shipped sample boards run as soon as the model is in place.
    assert main_module.image_folder.image_paths(APPS_ROOT / config.string("input.source"))
    output_dir = config.string("output.dir")
    assert output_dir.startswith("sandbox/")
    # An annotated image written into input.source would be read as a board on the next run.
    assert Path(output_dir) != Path(config.string("input.source"))
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    assert config.integer("inference.max_detections") > 0
    assert config.integer("app.input_max_width") > 0
    assert config.integer("app.input_max_height") > 0


def test_test_scope_downloads_the_configured_model():
    """CI downloads the package the shipped config names."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    scope = main_module.Config.load(EXAMPLE_DIR / "tests" / "test-scope.yaml")
    assert scope.string("models.yolo26n_plc.file") == Path(config.string("model.path")).name


def test_image_problem_names_what_the_model_cannot_take():
    assert images.image_problem(None, (640, 480)) == "not a readable image"
    assert images.image_problem(np.zeros((480, 640, 3), np.uint8), (640, 480)) == ""
    assert images.image_problem(np.zeros((480, 641, 3), np.uint8), (640, 480)) == (
        "641x480 is larger than app.input_max_width x app.input_max_height (640x480)")
    assert images.image_problem(np.zeros((481, 640, 3), np.uint8), (640, 480)) != ""


def test_save_image_reports_a_failed_write_instead_of_raising(tmp_path: Path):
    image = np.zeros((8, 8, 3), np.uint8)
    assert images.save_image(tmp_path / "board_pcb.png", image)
    # OpenCV raises for an encoder failure, such as an extension it has no writer for; one image
    # that cannot be saved must not stop the batch.
    assert not images.save_image(tmp_path / "board_pcb.unknown", image)
    assert not images.save_image(tmp_path / "missing" / "board_pcb.png", image)


def test_pixel_rounds_half_up_and_stays_in_the_image():
    assert overlay.pixel(2.5, 10) == 3
    assert overlay.pixel(2.49, 10) == 2
    assert overlay.pixel(-3.0, 10) == 0
    assert overlay.pixel(12.0, 10) == 9


def test_draw_boxes_draws_each_box_in_its_class_color():
    image = np.zeros((100, 100, 3), np.uint8)
    labels = ["missing_hole", "mouse_bite", "open_circuit"]
    box = SimpleNamespace(x1=10.0, y1=30.0, x2=60.0, y2=80.0, score=0.9, class_id=2)
    overlay.draw_boxes(image, [box], labels)
    # The left edge, below the label, is the class's color; the inside stays untouched.
    assert tuple(int(channel) for channel in image[60, 10]) == overlay.CLASS_COLORS[2]
    assert not image[55, 35].any()


def test_draw_boxes_skips_a_box_with_no_area():
    image = np.zeros((100, 100, 3), np.uint8)
    flat = SimpleNamespace(x1=40.0, y1=30.0, x2=40.2, y2=80.0, score=0.9, class_id=0)
    outside = SimpleNamespace(x1=120.0, y1=30.0, x2=150.0, y2=80.0, score=0.9, class_id=0)
    overlay.draw_boxes(image, [flat, outside], ["missing_hole"])
    assert not image.any()
