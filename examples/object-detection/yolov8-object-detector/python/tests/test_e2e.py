"""E2E tests for yolov8-object-detector (Python)."""

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
        tmp_output_dir,
        test_images_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
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

        output_files = [
            path
            for path in tmp_output_dir.iterdir()
            if path.is_file() and path.name != "config.yaml"
        ]
        assert output_files, "Expected output files but output directory is empty"
        assert all(path.stat().st_size > 0 for path in output_files)
