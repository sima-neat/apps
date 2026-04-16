# python/tests/test_unit.py
"""Unit tests for reid re-identification (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the ReID example."""

    def test_missing_all_args(self):
        """Running with no arguments should fail because images are required."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_second_image(self):
        """Passing only one image should fail because two are required."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "some_image.jpg"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_model_file(self):
        """Passing images but a non-existent --model path should fail."""
        r = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "some_image.jpg", "other_image.jpg",
                "--model", "does_not_exist.tar.gz",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "model file does not exist" in r.stderr.lower()

    def test_invalid_metric(self):
        """An unsupported --metric value should cause argparse to exit with code 2."""
        r = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "a.jpg", "b.jpg",
                "--metric", "manhattan",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "a.jpg", "b.jpg", "--bogus"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()