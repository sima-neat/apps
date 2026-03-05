"""Unit tests for offline-instance-segmentation-overlay (Python)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import pyneat

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"

_SPEC = importlib.util.spec_from_file_location("offline_instance_seg_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(main)


class FakeTensor:
    def __init__(self, arr: np.ndarray):
        self.dtype = pyneat.TensorDType.Float32
        self.shape = tuple(arr.shape)
        self._arr = np.ascontiguousarray(arr.astype(np.float32))

    def copy_dense_bytes_tight(self) -> bytes:
        return self._arr.tobytes()


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the instance segmentation overlay pipeline."""

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
class TestMaskDecodeAndOverlay:
    def test_class_color_uses_vivid_palette(self):
        assert main.class_color(0) == (56, 56, 255)
        assert main.class_color(1) == (151, 157, 255)
        assert main.class_color(2) == (31, 112, 255)

    def test_decode_returns_mask_coefficients_and_proto(self):
        def reg_cell():
            v = np.full((64,), -8.0, dtype=np.float32)
            for off in (0, 16, 32, 48):
                v[off + 6] = 8.0
            return v

        reg80 = np.zeros((1, 80, 80, 64), dtype=np.float32)
        reg40 = np.zeros((1, 40, 40, 64), dtype=np.float32)
        reg20 = np.zeros((1, 20, 20, 64), dtype=np.float32)
        cls80 = np.full((1, 80, 80, 80), -10.0, dtype=np.float32)
        cls40 = np.full((1, 40, 40, 80), -10.0, dtype=np.float32)
        cls20 = np.full((1, 20, 20, 80), -10.0, dtype=np.float32)
        mk80 = np.zeros((1, 80, 80, 32), dtype=np.float32)
        mk40 = np.zeros((1, 40, 40, 32), dtype=np.float32)
        mk20 = np.zeros((1, 20, 20, 32), dtype=np.float32)
        proto = np.zeros((1, 160, 160, 32), dtype=np.float32)

        reg80[0, 10, 12, :] = reg_cell()
        cls80[0, 10, 12, 3] = 10.0
        mk80[0, 10, 12, 0] = 1.0
        proto[0, 20:60, 20:60, 0] = 10.0

        tensors = [
            FakeTensor(reg80),
            FakeTensor(reg40),
            FakeTensor(reg20),
            FakeTensor(cls80),
            FakeTensor(cls40),
            FakeTensor(cls20),
            FakeTensor(mk80),
            FakeTensor(mk40),
            FakeTensor(mk20),
            FakeTensor(proto),
        ]

        dets, proto_out = main.decode_yolov8_instances(tensors, 640, 0.6, 0.45, 10)
        assert len(dets) == 1
        assert dets[0]["class_id"] == 3
        assert dets[0]["coeff"].shape == (32,)
        assert proto_out.shape == (160, 160, 32)

    def test_apply_mask_overlay_changes_pixels(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        proto = np.zeros((16, 16, 32), dtype=np.float32)
        proto[4:12, 4:12, 0] = 10.0
        coeff = np.zeros((32,), dtype=np.float32)
        coeff[0] = 1.0
        dets = [{"x1": 16.0, "y1": 16.0, "x2": 48.0, "y2": 48.0, "class_id": 1, "coeff": coeff}]

        main.apply_mask_overlay(bgr, dets, proto, infer_size=64, alpha=0.5)
        assert int(bgr.sum()) > 0

    def test_apply_mask_overlay_draws_class_colored_contour(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        proto = np.zeros((16, 16, 32), dtype=np.float32)
        proto[4:12, 4:12, 0] = 10.0
        coeff = np.zeros((32,), dtype=np.float32)
        coeff[0] = 1.0
        dets = [{"x1": 16.0, "y1": 16.0, "x2": 48.0, "y2": 48.0, "class_id": 2, "coeff": coeff}]

        main.apply_mask_overlay(bgr, dets, proto, infer_size=64, alpha=0.5)
        expected = np.array(main.class_color(2), dtype=np.uint8)
        assert np.any(np.all(bgr == expected, axis=2))

    def test_draw_boxes_uses_class_color(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        boxes = [{"x1": 10.0, "y1": 10.0, "x2": 40.0, "y2": 40.0, "score": 0.9, "class_id": 1}]
        main.draw_boxes(bgr, boxes, infer_size=64)

        expected = np.array(main.class_color(1), dtype=np.uint8)
        assert np.array_equal(bgr[10, 10], expected)
