"""Tracker tests for the multi-camera people tracking example."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestTracker:
    def test_reuses_track_id_for_nearby_detection(self):
        from utils.tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=2)

        first = tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 12.0, "y1": 12.0, "x2": 52.0, "y2": 82.0, "score": 0.88, "class_id": 0}],
            frame_index=1,
        )

        assert len(first) == 1
        assert len(second) == 1
        assert second[0].track_id == first[0].track_id

    def test_drops_track_after_missing_budget(self):
        from utils.tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=1)

        tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        tracker.update([], frame_index=1)
        tracker.update([], frame_index=2)

        assert tracker.active_track_count() == 0
