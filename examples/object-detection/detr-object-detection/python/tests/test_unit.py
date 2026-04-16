"""Unit tests for detr-object-detection (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the DETR example."""

    def test_missing_all_args(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_model_file(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "assets/test_images/image.png", "--model", "does_not_exist.tar.gz"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert r.returncode == 2
        assert "model file does not exist" in r.stderr.lower()

    def test_missing_image_file(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "/does/not/exist.png", "--model", "assets/models/fake.tar.gz"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert r.returncode == 2
        assert "model file does not exist" in r.stderr.lower()

    def test_unknown_flag(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "assets/test_images/image.png", "--bogus"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()
