"""Unit tests for single-rtsp-object-detection-optiview (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the single RTSP OptiView detection pipeline."""

    def test_missing_all_args(self):
        """Running with no arguments should fail (--rtsp is required)."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_rtsp_flag(self):
        """Providing --model but not --rtsp should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--model", "model.tar.gz"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "--rtsp" in r.stderr or "required" in r.stderr.lower()

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--rtsp", "rtsp://x", "--bogus"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()
