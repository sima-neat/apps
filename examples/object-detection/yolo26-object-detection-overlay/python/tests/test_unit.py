"""Unit tests for yolo26-object-detection-overlay (Python)."""
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

    @staticmethod
    def _run_with_extra(extra):
        """Run main.py with minimal valid required flags plus caller's extras."""
        base = [
            sys.executable, str(MAIN_PY),
            "--model", "model.tar.gz", "--labels", "labels.txt",
            "--input-dir", "/nonexistent/path/input",
            "--output-dir", "/tmp/output",
        ]
        return subprocess.run(base + extra, capture_output=True, text=True, timeout=10)

    @pytest.mark.parametrize("extra", [
        ["--profile"],
        ["--no-overlay"],
        ["--num-runs", "3"],
        ["--min-score", "0.3"],
        ["--nms-iou", "0.5"],
    ])
    def test_optional_flag_recognized(self, extra):
        """Each optional flag should parse cleanly; failure is from bad dir, not argparse."""
        r = self._run_with_extra(extra)
        assert r.returncode != 0
        assert "unrecognized" not in r.stderr.lower()

    @pytest.mark.parametrize("extra", [
        ["--num-runs", "0"],
        ["--num-runs", "-1"],
        ["--min-score", "2.0"],
        ["--min-score", "-0.1"],
        ["--nms-iou", "1.5"],
    ])
    def test_invalid_values_rejected(self, extra):
        """Out-of-range values must exit non-zero with an error message."""
        base = [
            sys.executable, str(MAIN_PY),
            "--model", "model.tar.gz", "--labels", "labels.txt",
            "--input-dir", "/tmp", "--output-dir", "/tmp/out",
        ]
        r = subprocess.run(base + extra, capture_output=True, text=True, timeout=10)
        assert r.returncode != 0
        assert "error" in r.stderr.lower()
