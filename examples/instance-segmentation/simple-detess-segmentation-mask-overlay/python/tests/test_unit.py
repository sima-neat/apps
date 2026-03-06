"""Unit tests for simple-detess-segmentation-mask-overlay (Python)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyneat
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"
SPEC = importlib.util.spec_from_file_location("detess_seg_main", MAIN_PY)
APP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(APP)


class FakeTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
        self.dtype = pyneat.TensorDType.Float32
        self.shape = list(self._arr.shape)

    def copy_dense_bytes_tight(self):
        return self._arr.tobytes()


def make_detess_seg_tensors(objectness: float = 0.9, class_score: float = 0.8):
    proto = np.zeros((160, 160, 32), dtype=np.float32)
    xy80 = np.zeros((80, 80, 6), dtype=np.float32)
    wh80 = np.zeros((80, 80, 6), dtype=np.float32)
    co80 = np.zeros((80, 80, 243), dtype=np.float32)
    mk80 = np.zeros((80, 80, 96), dtype=np.float32)
    xy40 = np.zeros((40, 40, 6), dtype=np.float32)
    wh40 = np.zeros((40, 40, 6), dtype=np.float32)
    co40 = np.zeros((40, 40, 243), dtype=np.float32)
    mk40 = np.zeros((40, 40, 96), dtype=np.float32)
    xy20 = np.zeros((20, 20, 6), dtype=np.float32)
    wh20 = np.zeros((20, 20, 6), dtype=np.float32)
    co20 = np.zeros((20, 20, 243), dtype=np.float32)
    mk20 = np.zeros((20, 20, 96), dtype=np.float32)

    xy80[10, 20, 0] = 100.0
    xy80[10, 20, 1] = 120.0
    wh80[10, 20, 0] = 40.0
    wh80[10, 20, 1] = 60.0
    co80[10, 20, 0] = objectness
    co80[10, 20, 1] = class_score

    arrays = [
        proto,
        xy80, wh80, co80, mk80,
        xy40, wh40, co40, mk40,
        xy20, wh20, co20, mk20,
    ]
    return [FakeTensor(arr) for arr in arrays]


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the DeteSS segmentation mask overlay pipeline."""

    def test_missing_all_args(self):
        """Running with no arguments should fail (3 positional args required)."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_two_positional_args(self):
        """Providing only model should fail (input_dir and output_dir missing)."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "model.tar.gz"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_one_positional_arg(self):
        """Providing model and input_dir but not output_dir should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "model.tar.gz", "/tmp/input"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_bad_input_dir(self):
        """A nonexistent input directory should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "model.tar.gz",
             "/nonexistent/path/input", "/tmp/output"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "model.tar.gz", "/tmp/in", "/tmp/out", "--bogus"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()


@pytest.mark.unit
class TestDecodeRegression:
    def test_decode_uses_detess_values_directly(self):
        dets, proto = APP.decode_yolov5_seg(make_detess_seg_tensors(), infer_size=640)

        assert proto.shape == (160, 160, 32)
        assert len(dets) == 1

        det = dets[0]
        assert det["class_id"] == 0
        assert det["score"] == pytest.approx(0.72, rel=1e-4)
        assert det["x1"] == pytest.approx(80.0, rel=1e-4)
        assert det["y1"] == pytest.approx(90.0, rel=1e-4)
        assert det["x2"] == pytest.approx(120.0, rel=1e-4)
        assert det["y2"] == pytest.approx(150.0, rel=1e-4)

    def test_decode_keeps_detections_above_new_threshold(self):
        dets, _ = APP.decode_yolov5_seg(
            make_detess_seg_tensors(objectness=0.5, class_score=0.8), infer_size=640
        )

        assert len(dets) == 1
        assert dets[0]["score"] == pytest.approx(0.4, rel=1e-4)


@pytest.mark.unit
class TestVisualizationBehavior:
    def test_class_color_uses_vivid_palette(self):
        assert APP.class_color(0) == (56, 56, 255)
        assert APP.class_color(1) == (151, 157, 255)
        assert APP.class_color(2) == (31, 112, 255)

    def test_apply_mask_overlay_draws_class_colored_contour(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        proto = np.zeros((16, 16, 32), dtype=np.float32)
        proto[:, :, 0] = 10.0
        dets = [{
            "x1": 16.0, "y1": 16.0, "x2": 48.0, "y2": 48.0,
            "score": 0.9, "class_id": 2,
            "coeff": np.array([1.0] + [0.0] * 31, dtype=np.float32),
        }]

        APP.apply_mask_overlay(bgr, dets, proto, infer_size=64, alpha=0.5)

        expected = np.array(APP.class_color(2), dtype=np.uint8)
        assert np.array_equal(bgr[16, 16], expected)
