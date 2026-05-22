"""E2E tests for simple-image-classification-pipeline (Python)."""

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
        self, models_dir, apps_root, tmp_output_dir, test_timeout_ms, skip_unless_e2e_ready
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
        config_path = tmp_output_dir.parent / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    f"  path: {model}",
                    "io:",
                    f"  image: {image_env}",
                    "  fallback_image_url: null",
                    "runtime:",
                    "  input_width: 224",
                    "  input_height: 224",
                    "  timeout_ms: 2000",
                    "validation:",
                    "  expected_class_id: 1",
                    "  min_probability: 0.0",
                    "",
                ]
            ),
            encoding="utf-8",
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
