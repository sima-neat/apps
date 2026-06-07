"""E2E tests for image-classifier (Python)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        e2e_model_path,
        apps_root,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
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
        config_path = e2e_config_writer(
            {
                "io": {"image": str(image_env), "fallback_image_url": None},
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
