"""E2E tests for image-classifier (Python)."""

import os
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


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        models_dir,
        apps_root,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        model = _find_model(models_dir, "resnet")
        skip_unless_e2e_ready(model is not None, "resnet model not found in models_dir")

        image_env = Path(
            os.environ.get(
                "SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
                str(apps_root / "assets" / "test_images_classification" / "goldfish.jpeg"),
            )
        )
        skip_unless_e2e_ready(
            image_env.exists(),
            "classification image missing; set SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
        )
        validation = e2e_config_section("image-classifier", "validation")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(model)},
                "io": {"image": str(image_env), "fallback_image_url": None},
                "validation": validation,
            }
        )

        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--config",
                str(config_path),
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
