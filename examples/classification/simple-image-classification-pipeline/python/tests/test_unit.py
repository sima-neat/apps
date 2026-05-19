"""Unit tests for simple-image-classification-pipeline (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the classification pipeline."""

    def test_help(self):
        """--help should print usage."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0
        assert "usage" in r.stdout.lower()

    def test_bad_config_path(self):
        """A missing config should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/config.yaml"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()
