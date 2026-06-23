"""Optional e2e tests for faster-rcnn-object-detector (Python)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(self, models_dir, test_images_dir, tmp_output_dir, test_timeout_ms, skip_unless_e2e_ready):
        backbone = models_dir / "backbone_rpn_head_640_640_mpk.tar.gz"
        head = models_dir / "box_head_predictor_640_640_mpk.tar.gz"
        skip_unless_e2e_ready(backbone.is_file(), f"missing model package: {backbone}")
        skip_unless_e2e_ready(head.is_file(), f"missing model package: {head}")
        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            f"test_images_dir is missing or empty: {test_images_dir}",
        )

        config_path = tmp_output_dir.parent / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "models": {
                        "backbone_rpn": {"path": str(backbone)},
                        "head_predictor": {"path": str(head)},
                    },
                    "io": {"input_dir": str(test_images_dir), "output_dir": str(tmp_output_dir)},
                    "runtime": {"num_runs": 1, "timeout_ms": int(os.environ.get("SIMANEAT_APPS_TEST_TIMEOUT_MS", test_timeout_ms))},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}
stdout:
{result.stdout}
stderr:
{result.stderr}"
        )
        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file() and path.name != "config.yaml"]
        assert output_files, "Expected annotated output images to be written"
        assert all(path.stat().st_size > 0 for path in output_files), "Output image is empty"
