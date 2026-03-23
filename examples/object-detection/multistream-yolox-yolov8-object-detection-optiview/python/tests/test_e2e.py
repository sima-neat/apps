"""Starter smoke test for multistream-yolox-yolov8-object-detection-optiview (Python)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.e2e
def test_validate_config_only_smoke() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(EXAMPLE_DIR),
    )
    assert result.returncode == 0
