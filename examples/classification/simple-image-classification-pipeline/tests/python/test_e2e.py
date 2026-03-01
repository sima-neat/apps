"""E2E tests for simple-image-classification-pipeline (Python)."""

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
        self, models_dir, test_images_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "resnet")
        skip_unless_e2e_ready(model is not None, "resnet model not found in models_dir")
        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )
        image = None
        for name in ("goldfish.jpeg", "goldfish.jpg", "n01443537_goldfish.JPEG"):
            candidate = test_images_dir / name
            if candidate.exists():
                image = candidate
                break
        skip_unless_e2e_ready(
            image is not None,
            "test_images_dir must contain goldfish.jpeg (or n01443537_goldfish.JPEG)",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--model",
                str(model),
                "--image",
                str(image),
                "--min-prob",
                "0.0",
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
