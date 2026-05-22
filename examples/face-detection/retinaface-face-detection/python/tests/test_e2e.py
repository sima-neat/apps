"""E2E tests for retinaface-face-detection (Python)."""

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


def _find_image(input_dir: Path, pattern: str) -> Path | None:
    if not input_dir.exists():
        return None
    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for f in images:
        if pattern in f.name.lower():
            return f
    if images:
        return images[0]
    return None


def _resolve_input_dir(default_dir: Path) -> Path:
    raw = os.environ.get("SIMANEAT_APPS_TEST_INPUT_DIR", "").strip()
    if raw:
        return Path(raw)
    return default_dir


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        models_dir,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
    ):
        model = _find_model(models_dir, "retinaface_mobilenet25")
        skip_unless_e2e_ready(model is not None, "retinaface model not found in models_dir")

        input_dir = _resolve_input_dir(test_images_dir)
        image = _find_image(input_dir, "face")
        skip_unless_e2e_ready(
            image is not None,
            f"no suitable test image found in input_dir={input_dir}",
        )

        out_path = tmp_output_dir / "retinaface_output.png"
        config_path = tmp_output_dir.parent / "config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "model:",
                    f"  path: {model}",
                    "io:",
                    f"  image: {image}",
                    f"  output: {out_path}",
                    "decode:",
                    "  confidence_threshold: 0.40",
                    "  nms_iou: 0.90",
                    "  top_k: 5000",
                    "  keep_top_k: 750",
                    "  max_draw: 50",
                    "  landmarks: true",
                    "runtime:",
                    "  timeout_ms: 5000",
                    "  profile: false",
                    "  num_runs: 1",
                    "  verbose: false",
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

        assert out_path.exists(), "Expected an annotated output image to be written"
