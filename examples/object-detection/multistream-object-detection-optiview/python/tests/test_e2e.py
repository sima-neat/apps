"""E2E tests for multistream-object-detection-optiview (Python)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_validate_config_only_smoke_runs(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: assets/models/yolo_v8m_mpk.tar.gz
                  family: yolov8
                streams:
                  - rtsp://127.0.0.1:8554/src1
                runtime:
                  worker_count: 2
                output:
                  optiview:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
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
