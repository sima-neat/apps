"""E2E tests for simple-detess-segmentation-mask-overlay (Python)."""

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
        self, models_dir, tmp_output_dir, test_images_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "yolov5")
        skip_unless_e2e_ready(model is not None, "yolo seg model not found in models_dir")

        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                str(model),
                str(test_images_dir),
                str(tmp_output_dir),
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

        output_files = list(tmp_output_dir.iterdir())
        assert len(output_files) > 0, "Expected output files but output directory is empty"
