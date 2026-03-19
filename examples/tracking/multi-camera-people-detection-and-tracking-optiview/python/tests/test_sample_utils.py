"""Sample helper tests for the multi-camera people tracking example."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestSampleUtils:
    def test_make_optiview_tracking_detection_uses_track_id_label_text(self):
        from utils.sample_utils import make_optiview_tracking_detection
        from utils.tracker import TrackedDetection

        class FakeOptiViewObject:
            def __init__(self):
                self.x = 0
                self.y = 0
                self.w = 0
                self.h = 0
                self.score = 0.0
                self.class_id = -1

        objects, labels = make_optiview_tracking_detection(
            SimpleNamespace(OptiViewObject=FakeOptiViewObject),
            [
                TrackedDetection(
                    track_id=42,
                    x1=1.0,
                    y1=2.0,
                    x2=11.0,
                    y2=22.0,
                    score=0.9,
                    class_id=0,
                )
            ],
        )

        assert len(objects) == 1
        assert objects[0].x == 1
        assert objects[0].y == 2
        assert objects[0].w == 10
        assert objects[0].h == 20
        assert objects[0].score == pytest.approx(0.9)
        assert objects[0].class_id == 0
        assert labels == ["Track ID: 42"]
