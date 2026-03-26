"""Unit tests for retinaface-face-detection (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the RetinaFace example."""

    def test_missing_all_args(self):
        """Running with no arguments should fail because image is required."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_model_file(self):
        """Passing an image but a non-existent --model path should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "some_image.jpg", "--model", "does_not_exist.tar.gz"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # main() prints a friendly error and returns 2 when the model is missing
        assert r.returncode == 2
        assert "model file does not exist" in r.stderr.lower()

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "some_image.jpg", "--bogus"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

