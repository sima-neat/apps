"""E2E tests for multistream-rtsp-detection-pipeline (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    for f in models_dir.iterdir():
        if pattern in f.name and "seg" not in f.name and f.name.endswith(".tar.gz"):
            return f
    return None


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self, models_dir, tmp_output_dir, rtsp_url, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "yolo_v8m")
        skip_unless_e2e_ready(model is not None, "yolo (non-seg) model not found in models_dir")

        cmd = [
            sys.executable, str(MAIN_PY),
            "--model", str(model),
            "--output", str(tmp_output_dir),
            "--rtsp", rtsp_url,
            "--frames", "10",
            "--fps", "30",
            "--width", "1280",
            "--height", "720",
            "--save-every", "1",
            "--tcp",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # The pipeline creates stream_N/ subdirectories with saved frames.
        all_files = list(tmp_output_dir.rglob("*"))
        output_files = [f for f in all_files if f.is_file()]
        assert len(output_files) > 0, "Expected output files but output directory is empty"

    def test_full_pipeline_multi_url(
        self, models_dir, tmp_output_dir, rtsp_urls, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "yolo_v8m")
        skip_unless_e2e_ready(model is not None, "yolo (non-seg) model not found in models_dir")

        cmd = [
            sys.executable, str(MAIN_PY),
            "--model", str(model),
            "--output", str(tmp_output_dir),
            "--frames", "10",
            "--fps", "30",
            "--width", "1280",
            "--height", "720",
            "--save-every", "1",
            "--tcp",
        ]
        for url in rtsp_urls:
            cmd.extend(["--rtsp", url])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        all_files = list(tmp_output_dir.rglob("*"))
        output_files = [f for f in all_files if f.is_file()]
        assert len(output_files) > 0, "Expected output files but output directory is empty"
