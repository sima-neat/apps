"""Hardware-gated MIPI camera capture smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DEFAULT_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("SIMA_NEAT_TEST_MIPI_CAMERA") != "1",
    reason="set SIMA_NEAT_TEST_MIPI_CAMERA=1 on a board with a configured camera",
)
def test_strict_camera_capture(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["camera"]["name"] = os.environ.get("SIMA_NEAT_MIPI_CAMERA_NAME", "")
    raw["capture"]["duration_seconds"] = 3
    raw["capture"]["sample_times_seconds"] = [1]
    raw["output"]["directory"] = str(tmp_path / "output")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, (
        f"MIPI capture failed with {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )
    summary = json.loads((tmp_path / "output" / "summary.json").read_text(encoding="utf-8"))
    assert summary["error"] is None
    assert summary["strict_zero_copy"] is True
    assert summary["capture_buffers"] == 32
    assert summary["frames_pulled"] > 0
    assert summary["timeouts"] == 0
    assert len(summary["captures"]) == 1
