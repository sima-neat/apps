"""E2E tests for single-rtsp-object-detection-optiview (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "main.py"


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    for f in models_dir.iterdir():
        if pattern in f.name and "seg" not in f.name and f.name.endswith(".tar.gz"):
            return f
    return None


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(self, models_dir, rtsp_url, test_timeout_ms, skip_unless_e2e_ready):
        model = _find_model(models_dir, "yolo_v8s")
        skip_unless_e2e_ready(model is not None, "yolo (non-seg) model not found in models_dir")

        cmd = [
            sys.executable, str(MAIN_PY),
            "--rtsp", rtsp_url,
            "--frames", "10",
            "--model", str(model),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        if result.returncode != 0 and "failed to open UDP H.264 writer" in result.stderr:
            skip_unless_e2e_ready(
                False, "OpenCV GStreamer x264 writer is unavailable for OptiView Python e2e"
            )

        # OptiView output is UDP, so verify the process exits cleanly.
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
