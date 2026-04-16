"""Unit tests for simple-newest-yolo-object-detection-overlay-pipeline (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the yolo26m object detection pipeline."""

    def test_missing_all_args(self):
        """Running with no arguments should fail (required flags missing)."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_required_flags(self):
        """Providing only --model should fail (--labels, --input-dir, --output-dir missing)."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--model", "model.tar.gz"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_two_required_flags(self):
        """Providing --model and --labels but not --input-dir/--output-dir should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz", "--labels", "labels.txt"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_missing_one_required_flag(self):
        """Providing --model, --labels, --input-dir but not --output-dir should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz", "--labels", "labels.txt",
             "--input-dir", "/tmp/input"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "error" in r.stderr.lower()

    def test_bad_input_dir(self):
        """A nonexistent input directory should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz", "--labels", "labels.txt",
             "--input-dir", "/nonexistent/path/input",
             "--output-dir", "/tmp/output"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz", "--labels", "labels.txt",
             "--input-dir", "/tmp/in", "--output-dir", "/tmp/out",
             "--bogus"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_profile_flag_accepted(self):
        """--profile should be recognized; failure should be from bad dir, not argparse."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY),
             "--model", "model.tar.gz", "--labels", "labels.txt",
             "--input-dir", "/nonexistent/path/input",
             "--output-dir", "/tmp/output",
             "--profile"],
            capture_output=True, text=True, timeout=10,
        )
        # Should fail due to bad input dir (exit 2), not argparse error
        assert r.returncode != 0
        # argparse errors always say "unrecognized arguments"; this should not
        assert "unrecognized" not in r.stderr.lower()
