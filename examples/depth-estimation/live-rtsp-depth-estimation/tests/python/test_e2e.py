"""E2E tests for live-rtsp-depth-estimation (Python)."""

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
        if pattern in f.name and f.name.endswith(".tar.gz"):
            return f
    return None


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self, models_dir, tmp_output_dir, rtsp_url, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "midas")
        skip_unless_e2e_ready(model is not None, "depth/midas model not found in models_dir")

        output_file = tmp_output_dir / "depth_overlay.mp4"

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "--model", str(model),
                "--rtsp", rtsp_url,
                "--frames", "5",
                "--fps", "30",
                "--tcp",
                "--output-file", str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        assert output_file.exists(), f"Output file was not created: {output_file}"
        assert output_file.stat().st_size > 0, f"Output file is empty: {output_file}"
