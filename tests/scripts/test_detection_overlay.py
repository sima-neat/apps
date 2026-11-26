"""Contract tests for support/python/sima_examples/detection_overlay.py, the box drawing of the
examples that save annotated images. The DETR example's C++ unit test runs the same cases on
support/cpp/sima_examples/detection_overlay.h.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The contract job installs only pytest and PyYAML, so these tests skip there and run wherever
# numpy and OpenCV are installed.
np = pytest.importorskip("numpy", reason="numpy is not installed")
pytest.importorskip("cv2", reason="OpenCV is not installed")

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.detection_overlay import draw_detections


def box(x1, y1, x2, y2, score, class_id):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


def test_draw_detections_draws_the_box_in_its_class_color_on_a_copy():
    image = np.zeros((60, 80, 3), np.uint8)
    annotated = draw_detections(image, [box(9.6, 20.4, 50.0, 55.0, 0.9, 1)], ["N/A", "person"])
    assert not image.any()
    # Class 1 is blue in BGR. x1 = 9.6 rounds to 10, and a 2-pixel line there covers columns 9
    # to 11; truncating to 9 would cover 8 to 10.
    assert tuple(annotated[40, 11]) == (255, 0, 0)
    assert tuple(annotated[40, 8]) == (0, 0, 0)
    assert tuple(annotated[40, 30]) == (0, 0, 0)


def test_draw_detections_repeats_the_colors_every_eight_classes():
    image = np.zeros((60, 80, 3), np.uint8)
    labels = [str(index) for index in range(10)]
    annotated = draw_detections(image, [box(10, 20, 50, 55, 0.9, 9)], labels)
    # Class 9 takes class 1's color.
    assert tuple(annotated[40, 10]) == (255, 0, 0)
