# python/tests/test_e2e.py
"""E2E tests for reid re-identification (Python)."""

import json
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


def _find_image_pair(images_dir: Path) -> tuple[Path, Path] | None:
    """Find two distinct images to use as a pair."""
    if not images_dir.exists():
        return None
    images = sorted(
        f for f in images_dir.rglob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 2:
        return None
    return images[0], images[1]


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline_cosine(
        self,
        models_dir,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        model = _find_model(models_dir, "reid")
        skip_unless_e2e_ready(model is not None, "reid model not found in models_dir")

        pair = _find_image_pair(test_images_dir)
        skip_unless_e2e_ready(pair is not None, f"no images found in {test_images_dir}")

        image_a, image_b = pair
        comparison = e2e_config_section("person-re-identification", "comparison.cosine")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(model)},
                "io": {
                    "image1": str(image_a),
                    "image2": str(image_b),
                    "output_dir": str(tmp_output_dir),
                },
                "output": {"type": "both"},
                "comparison": {"metric": "cosine", "threshold": comparison["threshold"]},
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
        assert (tmp_output_dir / "comparison.jpg").exists(), (
            "Expected comparison.jpg to be written to output dir"
        )
        result_json = tmp_output_dir / "result.json"
        assert result_json.exists(), "Expected result.json to be written to output dir"
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        assert payload["metric"] == "cosine"
        assert isinstance(payload["score"], float)
        assert payload["threshold"] == comparison["threshold"]
        assert payload["decision"] in {"SAME", "DIFFERENT"}

    def test_full_pipeline_euclidean(
        self,
        models_dir,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        model = _find_model(models_dir, "reid")
        skip_unless_e2e_ready(model is not None, "reid model not found in models_dir")

        pair = _find_image_pair(test_images_dir)
        skip_unless_e2e_ready(pair is not None, f"need at least 2 images in {test_images_dir}")
        image_a, image_b = pair
        comparison = e2e_config_section("person-re-identification", "comparison.euclidean")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(model)},
                "io": {
                    "image1": str(image_a),
                    "image2": str(image_b),
                    "output_dir": str(tmp_output_dir),
                },
                "output": {"type": "both"},
                "comparison": {"metric": "euclidean", "threshold": comparison["threshold"]},
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
        assert (tmp_output_dir / "comparison.jpg").exists(), (
            "Expected comparison.jpg to be written to output dir"
        )
        result_json = tmp_output_dir / "result.json"
        assert result_json.exists(), "Expected result.json to be written to output dir"
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        assert payload["metric"] == "euclidean"
        assert isinstance(payload["score"], float)
        assert payload["decision"] in {"SAME", "DIFFERENT"}

    def test_output_dir_is_created(
        self,
        models_dir,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        """Output directory should be created automatically if it does not exist."""
        model = _find_model(models_dir, "reid")
        skip_unless_e2e_ready(model is not None, "reid model not found in models_dir")

        pair = _find_image_pair(test_images_dir)
        skip_unless_e2e_ready(pair is not None, f"need at least 2 images in {test_images_dir}")
        image_a, image_b = pair

        new_out_dir = tmp_output_dir / "brand_new_dir"
        assert not new_out_dir.exists()
        comparison = e2e_config_section("person-re-identification", "comparison.cosine")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(model)},
                "io": {
                    "image1": str(image_a),
                    "image2": str(image_b),
                    "output_dir": str(new_out_dir),
                },
                "output": {"type": "both"},
                "comparison": {"metric": "cosine", "threshold": comparison["threshold"]},
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

        assert result.returncode == 0
        assert new_out_dir.exists(), "Output directory should have been created by main.py"

    def test_output_type_json_only(
        self,
        models_dir,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        model = _find_model(models_dir, "reid")
        skip_unless_e2e_ready(model is not None, "reid model not found in models_dir")

        pair = _find_image_pair(test_images_dir)
        skip_unless_e2e_ready(pair is not None, f"need at least 2 images in {test_images_dir}")
        image_a, image_b = pair

        out_dir = tmp_output_dir / "json_only"
        comparison = e2e_config_section("person-re-identification", "comparison.cosine")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(model)},
                "io": {
                    "image1": str(image_a),
                    "image2": str(image_b),
                    "output_dir": str(out_dir),
                },
                "output": {"type": "json"},
                "comparison": {"metric": "cosine", "threshold": comparison["threshold"]},
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
        assert (out_dir / "result.json").exists(), "Expected result.json in json-only mode"
        assert not (out_dir / "comparison.jpg").exists(), "comparison.jpg should not be created in json-only mode"
