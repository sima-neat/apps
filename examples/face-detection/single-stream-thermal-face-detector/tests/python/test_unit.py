"""Unit tests for single-stream-thermal-face-detector (Python)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"

# Pyramid grid sizes for the 800x800 canvas the model was compiled for.
LEVEL_SIZES = (100, 50, 25)


def load_example():
    spec = importlib.util.spec_from_file_location("thermal_face_example", MAIN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # main.py declares module-level dataclasses, which resolve annotations
    # through sys.modules, so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.np = np  # main.py binds numpy lazily at runtime
    return module


def build_nhwc_heads(example):
    """Six [1,H,W,C] split heads with one planted high-confidence face per level."""
    rng = np.random.default_rng(0)
    heads = []
    for size in LEVEL_SIZES:
        for channels in (example.BOX_CHANNELS, example.LM_CHANNELS):
            head = rng.standard_normal((1, size, size, channels)).astype(np.float32) * 0.5
            if channels == example.BOX_CHANNELS:
                head[0, size // 3, size // 4, 4] = 4.0  # objectness logit
                head[0, size // 3, size // 4, 5] = 4.0  # class logit
            heads.append(head)
    return heads


@pytest.mark.unit
class TestSplitHeadLayouts:
    """The decoder must accept every layout a raw split head can arrive in.

    NEAT delivers raw heads as [1,H,C,W] (the RetinaFace example documents the
    same), so a decoder that only probed the last and first axes rejected the
    real runtime tensors and published no metadata at all.
    """

    def decode(self, example, heads):
        return example.decode_yolov5face_split(
            heads, conf_threshold=0.25, iou_threshold=0.45, max_detections=50)

    def test_nhwc_layout_decodes(self):
        example = load_example()
        boxes, scores, landmarks = self.decode(example, build_nhwc_heads(example))
        assert len(boxes) > 0
        assert scores.shape == (len(boxes),)
        assert landmarks.shape == (len(boxes), example.NUM_LANDMARKS, 2)

    @pytest.mark.parametrize("name,axes", [
        ("neat_hcw", (0, 1, 3, 2)),  # [1,H,W,C] -> [1,H,C,W], NEAT's raw layout
        ("nchw", (0, 3, 1, 2)),      # [1,H,W,C] -> [1,C,H,W], ONNX-native
    ])
    def test_permuted_layouts_match_nhwc(self, name, axes):
        """A permuted head must decode to exactly the NHWC result."""
        example = load_example()
        nhwc = build_nhwc_heads(example)
        permuted = [np.ascontiguousarray(h.transpose(*axes)) for h in nhwc]

        expected = self.decode(example, nhwc)
        actual = self.decode(example, permuted)

        for got, want, field in zip(actual, expected, ("boxes", "scores", "landmarks")):
            assert np.allclose(got, want, atol=1e-5), f"{name}: {field} differ"

    def test_unrecognized_shape_is_rejected(self):
        example = load_example()
        bogus = [np.zeros((1, 7, 7, 7), dtype=np.float32)]
        with pytest.raises(ValueError, match="Unrecognized split output shape"):
            self.decode(example, bogus)


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the single-stream yolov5s-face pipeline."""

    def test_help(self):
        """--help should describe the config-driven CLI."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        """A missing config file should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/single-stream-thermal-face-detector-config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_validate_config_only(self):
        """--validate-config-only should parse the shipped config without touching hardware."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(COMMON_CONFIG), "--validate-config-only"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "validated" in r.stdout.lower()

    def test_validate_config_rejects_empty_labels_path(self, tmp_path):
        """An explicit empty path must not be normalized to the current directory."""
        config = COMMON_CONFIG.read_text(encoding="utf-8")
        config = config.replace(
            "labels: examples/face-detection/single-stream-thermal-face-detector/src/common/face_label.txt",
            'labels: ""',
        )
        config_path = tmp_path / "empty-labels.yaml"
        config_path.write_text(config, encoding="utf-8")

        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode != 0
        assert "model.labels must be set" in r.stderr
