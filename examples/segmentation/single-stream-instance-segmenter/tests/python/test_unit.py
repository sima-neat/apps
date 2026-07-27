"""Unit tests for single-stream-instance-segmenter (Python)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

_SPEC = importlib.util.spec_from_file_location("instance_seg_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


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
    def test_hevc_alias(self):
        assert main.parse_source_codec("hevc") == "h265"

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
  path: model.tar.gz
source:
  rtsp_url: rtsp://127.0.0.1:8554/src1
output:
  mask_alpha: 2
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mask_alpha"):
            main.load_app_config(config)


@pytest.mark.unit
class TestMaskOverlay:
    def setup_method(self):
        main.load_runtime_dependencies()

    def test_class_color_uses_vivid_palette(self):
        assert main.class_color(0) == (56, 56, 255)
        assert main.class_color(1) == (151, 157, 255)
        assert main.class_color(2) == (31, 112, 255)

    def test_overlay_segmentation_changes_pixels(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[40:120, 40:120] = 255
        dets = [
            {
                "x1": 16.0,
                "y1": 16.0,
                "x2": 48.0,
                "y2": 48.0,
                "score": 0.9,
                "class_id": 1,
                "mask": mask,
            }
        ]
        cfg = main.OutputConfig("", 0, 0.5, 0.5, True)

        out = main.overlay_segmentation(frame, dets, 0.55, cfg, ["person", "bicycle"])

        assert int(out.sum()) > 0

    def test_overlay_segmentation_can_skip_boxes(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[40:120, 40:120] = 255
        dets = [
            {
                "x1": 16.0,
                "y1": 16.0,
                "x2": 48.0,
                "y2": 48.0,
                "score": 0.9,
                "class_id": 2,
                "mask": mask,
            }
        ]
        cfg = main.OutputConfig("", 0, 0.5, 0.5, False)

        out = main.overlay_segmentation(frame, dets, 0.55, cfg, ["person", "bicycle", "car"])

        assert int(out.sum()) > 0
