"""E2E tests for multi-stream-object-detector (Python)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_validate_config_only_smoke_runs(self, e2e_config_writer):
        config_path = e2e_config_writer(
            {
                "model": {"path": "assets/models/yolo_v8m_mpk.tar.gz"},
                "streams": ["rtsp://127.0.0.1:8554/src1"],
                "runtime": {"worker_count": 2},
                "output": {"insight": {"host": "127.0.0.1"}},
            }
        )

        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--config",
                str(config_path),
                "--validate-config-only",
            ],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=10,
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "Config validated" in result.stdout
