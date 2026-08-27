"""Unit tests for rfdetr-stream-instance-segmenter (Python)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

_SPEC = importlib.util.spec_from_file_location("rfdetr_stream_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


@pytest.fixture(autouse=True)
def _use_numpy(monkeypatch) -> None:
    """Populate main.np without pulling in pyneat/tvm, which this example's full
    load_runtime_dependencies() also requires and which are only available inside
    the Neat SDK Python environment."""
    monkeypatch.setattr(main, "np", np)


def _use_cv2(monkeypatch) -> None:
    """Populate main.cv2 for tests that exercise image ops (overlay/polygon helpers)."""
    import cv2

    monkeypatch.setattr(main, "cv2", cv2)


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the RTSP segmentation pipeline."""

    def test_help(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/config.yaml"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()


@pytest.mark.unit
class TestConfig:
    def test_validate_config_only_accepts_common_config(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
        )
        assert r.returncode == 0, r.stderr

    def test_invalid_mask_alpha_is_rejected(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
model:
  path: models/rfdetr-seg-432-base
source:
  url: rtsp://127.0.0.1:8554/src1
output:
  mask_alpha: 2
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mask_alpha"):
            main.load_app_config(config)

    def test_missing_source_url_is_rejected(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
model:
  path: models/rfdetr-seg-432-base
output:
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="source.url"):
            main.load_app_config(config)


@pytest.mark.unit
class TestSelectDets:
    def setup_method(self):
        self.np = np

    def test_keeps_only_scores_above_threshold(self):
        logits = np.full((main.NUM_QUERIES, main.NUM_CLASSES), -30.0, dtype=np.float32)
        # A large positive logit for query 0, class 3 sigmoids close to 1.
        logits[0, 3] = 10.0
        boxes = np.zeros((main.NUM_QUERIES, 4), dtype=np.float32)
        fo = main.FrameOut(boxes, logits, np.zeros((108, 108, main.NUM_QUERIES), dtype=np.float32))

        detections = main.select_dets(fo, score_threshold=0.5, max_detections=24)

        assert len(detections) == 1
        assert detections[0]["query"] == 0
        assert detections[0]["class_id"] == 3
        assert detections[0]["score"] > 0.99

    def test_caps_at_max_detections(self):
        logits = np.zeros((main.NUM_QUERIES, main.NUM_CLASSES), dtype=np.float32)
        logits[:, 0] = 10.0  # every query scores above threshold on class 0
        boxes = np.zeros((main.NUM_QUERIES, 4), dtype=np.float32)
        fo = main.FrameOut(boxes, logits, np.zeros((108, 108, main.NUM_QUERIES), dtype=np.float32))

        detections = main.select_dets(fo, score_threshold=0.5, max_detections=5)

        assert len(detections) == 5


@pytest.mark.unit
class TestStretchProjection:
    """The compiled graph resizes with a plain stretch (no letterbox padding), so
    box/mask projection back to frame pixels is a direct per-axis scale."""

    def test_frame_rect_for_query_scales_independently_per_axis(self):
        boxes = np.zeros((main.NUM_QUERIES, 4), dtype=np.float32)
        boxes[0] = [0.5, 0.5, 0.5, 0.25]  # cx, cy, w, h (normalized)
        fo = main.FrameOut(boxes, np.zeros((main.NUM_QUERIES, main.NUM_CLASSES), dtype=np.float32),
                           np.zeros((108, 108, main.NUM_QUERIES), dtype=np.float32))

        x0, y0, x1, y1 = main.frame_rect_for_query(fo, 0, (1080, 1920, 3))

        assert (x0, x1) == (480, 1440)  # (0.5 +/- 0.25) * 1920
        assert (y0, y1) == (405, 675)  # (0.5 +/- 0.125) * 1080

    def test_mask_rect_for_frame_rect_is_proportional(self):
        rect = main.mask_rect_for_frame_rect((0, 0, 216, 216), (432, 432, 3))
        assert rect == (0, 0, 54, 54)  # 216/432 of 108 in each axis


@pytest.mark.unit
class TestMaskOverlay:
    def setup_method(self, method=None):
        pass

    def test_overlay_segmentation_changes_pixels(self, monkeypatch):
        _use_cv2(monkeypatch)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        masks_dev = np.zeros((108, 108, main.NUM_QUERIES), dtype=np.float32)
        masks_dev[20:80, 20:80, 0] = 10.0  # large positive logit -> sigmoid ~= 1
        boxes = np.zeros((main.NUM_QUERIES, 4), dtype=np.float32)
        boxes[0] = [0.5, 0.5, 0.5, 0.5]
        logits = np.zeros((main.NUM_QUERIES, main.NUM_CLASSES), dtype=np.float32)
        fo = main.FrameOut(boxes, logits, masks_dev)
        dets = [{"query": 0, "score": 0.9, "class_id": 1}]
        cfg = main.OutputConfig("", 0, 0.5, 0.5, True)

        out = main.overlay_segmentation(frame, fo, dets, ["person", "bicycle"], cfg)

        assert int(out.sum()) > 0


@pytest.mark.unit
class TestSegmentationMetadata:
    def test_polygon_is_frame_absolute_and_in_bounds(self, monkeypatch):
        _use_cv2(monkeypatch)
        mask108 = np.full((108, 108), 255, dtype=np.uint8)

        polygon = main.mask_polygon(mask108, (1600, 900, 1900, 1070), (1080, 1920, 3), 0.5)

        assert len(polygon) >= 3
        assert all(1600 <= x <= 1900 and 900 <= y <= 1070 for x, y in polygon)

    def test_polygon_is_empty_without_foreground(self, monkeypatch):
        _use_cv2(monkeypatch)
        mask108 = np.zeros((108, 108), dtype=np.uint8)

        assert main.mask_polygon(mask108, (0, 0, 64, 64), (640, 640, 3), 0.5) == []

    def test_build_metadata_segments_shape(self, monkeypatch):
        _use_cv2(monkeypatch)
        masks_dev = np.zeros((108, 108, main.NUM_QUERIES), dtype=np.float32)
        masks_dev[:, :, 0] = 10.0
        boxes = np.zeros((main.NUM_QUERIES, 4), dtype=np.float32)
        boxes[0] = [0.5, 0.5, 0.75, 0.75]
        fo = main.FrameOut(boxes, np.zeros((main.NUM_QUERIES, main.NUM_CLASSES), dtype=np.float32),
                           masks_dev)
        dets = [{"query": 0, "score": 0.9, "class_id": 0}]

        segments = main.build_metadata_segments(fo, dets, ["person"], (64, 64, 3), 0.5)

        assert len(segments) == 1
        assert segments[0]["id"] == "seg_1"
        assert segments[0]["label"] == "person"
        assert segments[0]["mask_format"] == "polygon"
        assert len(segments[0]["mask"]) >= 3

    def test_encode_segments_emits_segments_array(self):
        segment = {
            "id": "seg_1", "label": "person", "confidence": 0.9,
            "bbox": [0, 0, 64, 64], "mask_format": "polygon",
            "mask": [[i % 64, i // 64] for i in range(5)],
        }

        data_json, dropped = main.encode_segments([segment])

        data = json.loads(data_json)
        assert dropped == 0
        assert len(data["segments"]) == 1
        assert data["segments"][0]["mask_format"] == "polygon"

    def test_budget_drops_lowest_confidence_first(self):
        segments = [
            {
                "id": f"seg_{i}", "label": "person", "confidence": 0.5 + 0.01 * i,
                "bbox": [0, 0, 64, 64], "mask_format": "polygon",
                "mask": [[i % 64, i // 64] for i in range(1000)],
            }
            for i in range(12)
        ]

        data_json, dropped = main.encode_segments(segments)

        data = json.loads(data_json)
        assert len(data_json) <= main.METADATA_BYTE_BUDGET
        assert 0 < len(data["segments"]) < len(segments)
        assert len(data["segments"]) + dropped == len(segments)
