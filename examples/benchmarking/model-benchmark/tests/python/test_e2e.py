"""E2E tests for model-benchmark (Python)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.e2e
def test_generates_benchmark_report(
    e2e_model_path,
    tmp_output_dir,
    test_timeout_ms,
    e2e_config_writer,
) -> None:
    report_path = tmp_output_dir / "report.json"
    config_path = e2e_config_writer(
        {
            "benchmark": {"frames": 3},
            "output": {"report_json": str(report_path)},
        }
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=str(EXAMPLE_DIR),
    )

    assert result.returncode == 0, (
        f"main.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["benchmark"]["frames"] == 3
    assert data["model"]["file"] == e2e_model_path.name
    assert data["metrics"]["latency_ms"] > 0
    assert data["metrics"]["fps"] > 0
    assert data["metrics"]["avg_power_watts"] >= 0
    assert data["metrics"]["energy_joules"] >= 0
