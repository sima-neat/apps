"""E2E tests for retinaface-face-detection (Python)."""

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
        if pattern in f.name and f.name.endswith(".tar.gz"):
            return f
    return None


def _find_image(input_dir: Path, pattern: str) -> Path | None:
    if not input_dir.exists():
        return None
    for f in input_dir.iterdir():
        if pattern in f.name and f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return f
    return None


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        models_dir,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
    ):
        model = _find_model(models_dir, "retinaface_mobilenet25")
        skip_unless_e2e_ready(model is not None, "retinaface model not found in models_dir")

        image = _find_image(test_images_dir, "face")
        skip_unless_e2e_ready(image is not None, "no suitable face test image found in test_images_dir")

        out_path = tmp_output_dir / "retinaface_output.png"

        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                str(image),
                "--model",
                str(model),
                "--conf",
                "0.4",
                "--nms",
                "0.9",
                "--output",
                str(out_path),
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

        assert out_path.exists(), "Expected an annotated output image to be written"

