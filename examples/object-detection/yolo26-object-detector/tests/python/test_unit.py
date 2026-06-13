"""Unit tests for yolo26-object-detector (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the yolo26m object detection pipeline."""

    def test_help(self):
        """--help should describe the config-driven CLI."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        """A missing config file should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/yolo26-config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()
