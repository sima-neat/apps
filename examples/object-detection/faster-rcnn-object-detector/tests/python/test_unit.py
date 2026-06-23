"""Unit tests for faster-rcnn-object-detector (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_bad_config_path(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/faster-rcnn-config.yaml"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode != 0

    def test_unknown_flag(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 2
        assert "unrecognized" in result.stderr.lower() or "error" in result.stderr.lower()
