"""Unit tests for offline-depth-map-generation (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the offline depth map generation pipeline."""

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
