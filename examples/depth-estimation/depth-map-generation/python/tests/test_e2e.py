"""E2E tests for depth-map-generation (Python)."""

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
        self, models_dir, tmp_output_dir, test_images_dir, test_timeout_ms, skip_unless_e2e_ready
    ):
        model = _find_model(models_dir, "depth_anything")
        skip_unless_e2e_ready(model is not None, "depth_anything model not found in models_dir")

        skip_unless_e2e_ready(
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            "test_images_dir is missing or empty",
        )

        config_path = tmp_output_dir.parent / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    f"  path: {model}",
                    "io:",
                    f"  input_dir: {test_images_dir}",
                    f"  output_dir: {tmp_output_dir}",
                    "runtime:",
                    "  infer_size: 518",
                    "  timeout_ms: 5000",
                    "  queue_depth: 4",
                    "",
                ]
            ),
            encoding="utf-8",
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

        for f in output_files:
            assert f.stat().st_size > 0, f"Output file is empty: {f.name}"
