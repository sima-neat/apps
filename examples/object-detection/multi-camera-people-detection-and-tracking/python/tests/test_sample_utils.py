"""Sample helper tests for the multi-camera people tracking example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestSampleUtils:
    def test_make_optiview_tracking_json_uses_track_id_label_text(self):
        from utils.sample_utils import make_optiview_tracking_json
        from utils.tracker import TrackedDetection

        payload = make_optiview_tracking_json(
            1234,
            "frame-7",
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

        data = json.loads(payload)
        obj = data["data"]["objects"][0]

        assert obj["id"] == "track_42"
        assert obj["track_id"] == 42
        assert obj["label"] == "Track ID: 42"
