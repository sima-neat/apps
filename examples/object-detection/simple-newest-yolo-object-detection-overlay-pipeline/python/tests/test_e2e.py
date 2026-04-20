"""E2E tests for simple-newest-yolo-object-detection-overlay-pipeline (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"
LABELS_FILE = EXAMPLE_DIR / "common" / "coco_label.txt"
EXPECTED_MODEL = "yolo26m_mod_mpk.tar.gz"


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self, models_dir, tmp_output_dir, test_images_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = models_dir / EXPECTED_MODEL
        skip_unless_e2e_ready(
            model.is_file(),
            f"expected model '{EXPECTED_MODEL}' not found in {models_dir}",
        )

        skip_unless_e2e_ready(LABELS_FILE.exists(), f"Labels file not found: {LABELS_FILE}")

        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "--model", str(model),
                "--labels", str(LABELS_FILE),
                "--input-dir", str(test_images_dir),
                "--output-dir", str(tmp_output_dir),
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
