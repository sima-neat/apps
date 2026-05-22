"""E2E tests for simple-pose-estimation-overlay-pipeline (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MAIN_PY = WORKSPACE_DIR / "main.py"


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
        self, models_dir, tmp_output_dir, test_images_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "pose")
        skip_unless_e2e_ready(model is not None, "pose model not found in models_dir")

        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        config_path = tmp_output_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"path": str(model)},
                    "io": {
                        "input_dir": str(test_images_dir),
                        "output_dir": str(tmp_output_dir),
                    },
                    "runtime": {
                        "infer_size": 640,
                        "timeout_ms": 1000,
                        "upsample_factor": 4.0,
                    },
                    "decode": {
                        "keypoint_score": 0.1,
                        "nms_radius": 6,
                        "paf_score": 0.05,
                        "paf_success_ratio": 0.8,
                        "paf_samples": 10,
                        "min_valid_joints": 3,
                        "min_avg_person_score": 0.2,
                    },
                }
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
            cwd=str(WORKSPACE_DIR),
        )

        assert result.returncode == 0, (
            f"main_pose.py exited with code {result.returncode}\\n"
            f"stdout:\\n{result.stdout}\\nstderr:\\n{result.stderr}"
        )

        output_files = [
            path
            for path in tmp_output_dir.iterdir()
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(output_files) > 0, "Expected output files but output directory is empty"
