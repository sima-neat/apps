#!/usr/bin/env python3
"""Unit tests for mae-medical-anomaly-detection (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the MAE anomaly-detection pipeline."""

    def test_missing_all_args(self) -> None:
        """Running with no arguments should fail because positional args are required."""
        result = subprocess.run(
            [sys.executable, str(MAIN_PY)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        # argparse exits with code 2 for usage errors.
        assert result.returncode == 2
        assert "usage" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_unknown_flag(self) -> None:
        """An unrecognized flag should cause argparse to exit with code 2."""
        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--bogus",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 2
        assert "unrecognized" in result.stderr.lower() or "error" in result.stderr.lower()

