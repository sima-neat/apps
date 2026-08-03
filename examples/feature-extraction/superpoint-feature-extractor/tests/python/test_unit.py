"""CLI tests for the Python SuperPoint example."""

import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.unit
class TestCli:
    def run(self, *args: str):
        return subprocess.run(
            [sys.executable, str(MAIN_PY), *args],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_help(self):
        result = self.run("--help")
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_unknown_argument_is_rejected(self):
        result = self.run("--bogus")
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr

    def test_missing_config_value_is_rejected(self):
        result = self.run("--config")
        assert result.returncode == 2
        assert "expected one argument" in result.stderr

    def test_missing_config_is_reported(self):
        result = self.run("--config", "/does/not/exist.yaml")
        assert result.returncode == 2
        assert "Error:" in result.stderr
