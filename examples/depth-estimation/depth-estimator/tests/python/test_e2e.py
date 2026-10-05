"""E2E tests for depth-estimator (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.output_assertions import (
    assert_saved_frames_are_usable,
    supported_image_files,
)

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_images_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            test_images_dir.exists() and bool(supported_image_files(test_images_dir)),
            "test_images_dir has no images to process",
        )

        config_path = e2e_config_writer(
            {
                "io": {"input_dir": str(test_images_dir), "output_dir": str(tmp_output_dir)},
            }
        )

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "--config", str(config_path),
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

        assert_saved_frames_are_usable(
            tmp_output_dir, len(supported_image_files(test_images_dir))
        )
