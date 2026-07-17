"""Unit tests for single-stream-thermal-face-detector (Python)."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the single-stream yolov5s-face pipeline."""

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
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/single-stream-thermal-face-detector-config.yaml"],
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

    def test_validate_config_only(self):
        """--validate-config-only should parse the shipped config without touching hardware."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(COMMON_CONFIG), "--validate-config-only"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "validated" in r.stdout.lower()
