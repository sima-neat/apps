"""E2E tests for yolo26-object-detector (Python)."""

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
        self,
        models_dir,
        tmp_output_dir,
        test_images_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
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

        decode = e2e_config_section("yolo26-object-detector", "decode")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(model), "labels": str(LABELS_FILE)},
                "io": {"input_dir": str(test_images_dir), "output_dir": str(tmp_output_dir)},
                "decode": decode,
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
        assert len(output_files) > 0, "Expected output files but output directory is empty"
