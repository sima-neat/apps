"""E2E tests for pcb-defect-detector (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        e2e_model_path,
        apps_root,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        # PCB defects are not present in the shared COCO fixtures, so this
        # example uses its own test images instead of test_images_dir.
        input_dir = apps_root / "assets" / "datasets-test" / "pcb"
        skip_unless_e2e_ready(
            input_dir.is_dir() and any(input_dir.iterdir()),
            f"PCB test images are missing or empty: {input_dir}",
        )

        config_path = e2e_config_writer(
            {
                "io": {"input_dir": str(input_dir), "output_dir": str(tmp_output_dir)},
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

        expected = len([path for path in input_dir.iterdir() if path.is_file()])
        output_files = [
            path
            for path in tmp_output_dir.iterdir()
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(output_files) == expected, (
            f"Expected {expected} annotated images, found {len(output_files)}"
        )
        assert all(path.stat().st_size > 0 for path in output_files)
        assert all(path.suffix == ".png" for path in output_files)
