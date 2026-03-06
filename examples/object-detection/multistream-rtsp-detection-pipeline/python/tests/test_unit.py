"""Unit tests for multistream-rtsp-detection-pipeline (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the multistream RTSP detection pipeline."""

    def test_missing_all_args(self):
        """Running with no arguments should fail (--model, --output, --rtsp required)."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_output_and_rtsp(self):
        """Providing --model but not --output or --rtsp should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--model", "model.tar.gz"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "required" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_missing_rtsp(self):
        """Providing --model and --output but not --rtsp should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz", "--output", "/tmp/out"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "--rtsp" in r.stderr or "required" in r.stderr.lower()

    def test_missing_model(self):
        """Providing --rtsp and --output but not --model should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--rtsp", "rtsp://localhost:8554/stream", "--output", "/tmp/out"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "--model" in r.stderr or "required" in r.stderr.lower()

    def test_missing_output(self):
        """Providing --model and --rtsp but not --output should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz",
             "--rtsp", "rtsp://localhost:8554/stream"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "--output" in r.stderr or "required" in r.stderr.lower()

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "m.tar.gz",
             "--rtsp", "rtsp://x",
             "--output", "/tmp/out",
             "--bogus"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()
