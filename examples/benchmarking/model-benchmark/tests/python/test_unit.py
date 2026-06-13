"""Unit tests for model-benchmark (Python)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.unit
def test_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


@pytest.mark.unit
def test_missing_model_fails_before_pyneat_import(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  path: missing.tar.gz
benchmark:
  frames: 1
output:
  report_json: report.json
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "model file does not exist" in result.stderr


@pytest.mark.unit
def test_zero_frames_fails(tmp_path: Path) -> None:
    model = tmp_path / "model.tar.gz"
    model.write_text("fake model", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--model", str(model), "--frames", "0"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "benchmark.frames must be > 0" in result.stderr


@pytest.mark.unit
def test_writes_json_report_with_benchmark_metrics(tmp_path: Path) -> None:
    fake_pyneat = tmp_path / "pyneat.py"
    fake_pyneat.write_text(
        """
class Report:
    latency_ms = 1.25
    fps = 800.0
    avg_power_watts = 2.5
    energy_joules = 0.75


class Model:
    def __init__(self, path):
        self.path = path

    def input_specs(self):
        return ["input:uint8[1,3,224,224]"]

    def output_specs(self):
        return ["output:float32[1,1000]"]

    def benchmark(self, frames):
        assert frames == 7
        return Report()
""",
        encoding="utf-8",
    )
    model = tmp_path / "model.tar.gz"
    model.write_text("fake model", encoding="utf-8")
    report = tmp_path / "report.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            str(MAIN_PY),
            "--model",
            str(model),
            "--frames",
            "7",
            "--output-json",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["benchmark"]["type"] == "model.synthetic"
    assert data["benchmark"]["frames"] == 7
    assert data["model"]["file"] == "model.tar.gz"
    assert data["metrics"] == {
        "latency_ms": 1.25,
        "fps": 800.0,
        "avg_power_watts": 2.5,
        "energy_joules": 0.75,
    }
