"""E2E tests for image-classification-explorer (Python)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"


def _validation_expectations() -> tuple[int, float]:
    """Expected top-1 class id and minimum probability for the bundled goldfish
    fixture, read from the shipped config's `validation` block."""
    validation = yaml.safe_load(COMMON_CONFIG.read_text())["validation"]
    return int(validation["expected_class_id"]), float(validation["min_probability"])


def _assert_goldfish(image_entry: dict, model_names) -> None:
    """A nonempty top-k is not enough: a regression in normalization, output
    interpretation or label mapping would still produce one. Pin the known
    class for the known image."""
    expected_id, min_probability = _validation_expectations()
    for model_name in model_names:
        top1 = image_entry["predictions"][model_name]["top_k"][0]
        assert top1["class_id"] == expected_id, (
            f"{model_name}: expected class {expected_id}, got {top1['class_id']} "
            f"({top1['label']})"
        )
        assert top1["probability"] >= min_probability, (
            f"{model_name}: goldfish probability {top1['probability']:.4f} below "
            f"{min_probability}"
        )


MODEL_NAMES = ("resnet_50", "resnet_18", "efficientnet_b0", "densenet_121")


def _model_paths(models_dir: Path) -> dict[str, Path]:
    return {name: models_dir / f"{name}_mpk.tar.gz" for name in MODEL_NAMES}


@pytest.mark.e2e
class TestE2E:
    def test_single_image_two_models(
        self,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        tmp_output_dir,
    ):
        default_image = apps_root / "assets" / "datasets-test" / "imagenet" / "goldfish.jpeg"
        image_env = Path(
            os.environ.get("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE", str(default_image))
        )
        skip_unless_e2e_ready(
            image_env.exists(),
            "classification image missing; set SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
        )
        paths = _model_paths(models_dir)
        skip_unless_e2e_ready(
            all(p.is_file() for p in paths.values()),
            f"{', '.join(MODEL_NAMES)} model packages not found under {models_dir}",
        )

        output_dir = tmp_output_dir.parent / "report"
        config_path = e2e_config_writer(
            {
                "io": {"input": str(image_env), "output_dir": str(output_dir)},
                "models": {name: {"path": str(paths[name])} for name in MODEL_NAMES},
            }
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        report_json = output_dir / "report.json"
        assert report_json.is_file()
        assert (output_dir / "report.csv").is_file()
        assert (output_dir / "report.html").is_file()

        payload = json.loads(report_json.read_text())
        assert set(payload["models"]) == set(MODEL_NAMES)
        assert len(payload["images"]) == 1
        image_entry = payload["images"][0]
        assert "errors" not in image_entry, f"image had errors: {image_entry.get('errors')}"
        for model_name in MODEL_NAMES:
            assert image_entry["predictions"][model_name]["top_k"], (
                f"{model_name} produced no predictions"
            )
        if image_env == default_image:
            _assert_goldfish(image_entry, MODEL_NAMES)

    def test_directory_input(
        self,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        tmp_output_dir,
        test_images_dir,
    ):
        skip_unless_e2e_ready(
            test_images_dir.is_dir() and any(test_images_dir.glob("*.jpg")),
            f"no test images found under {test_images_dir}",
        )
        paths = _model_paths(models_dir)
        skip_unless_e2e_ready(
            all(p.is_file() for p in paths.values()),
            f"{', '.join(MODEL_NAMES)} model packages not found under {models_dir}",
        )

        output_dir = tmp_output_dir.parent / "report"
        config_path = e2e_config_writer(
            {
                "io": {"input": str(test_images_dir), "output_dir": str(output_dir)},
                "models": {name: {"path": str(paths[name])} for name in MODEL_NAMES},
            }
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        payload = json.loads((output_dir / "report.json").read_text())
        expected_count = len(list(test_images_dir.glob("*.jpg")))
        assert len(payload["images"]) == expected_count
        assert payload["timing"]["image_count"] == expected_count
        for image_entry in payload["images"]:
            assert "errors" not in image_entry, (
                f"{image_entry['path']} had errors: {image_entry.get('errors')}"
            )
            for model_name in MODEL_NAMES:
                assert image_entry["predictions"][model_name]["top_k"], (
                    f"{image_entry['path']}: {model_name} produced no predictions"
                )

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_each_model_individually(
        self,
        model_name,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        tmp_output_dir,
    ):
        """Each supported model must work on its own, not just alongside the others."""
        default_image = apps_root / "assets" / "datasets-test" / "imagenet" / "goldfish.jpeg"
        image_env = Path(
            os.environ.get("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE", str(default_image))
        )
        skip_unless_e2e_ready(
            image_env.exists(),
            "classification image missing; set SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
        )
        model_path = _model_paths(models_dir)[model_name]
        skip_unless_e2e_ready(
            model_path.is_file(), f"{model_name} model package not found under {models_dir}"
        )

        output_dir = tmp_output_dir.parent / "report"
        config_path = tmp_output_dir.parent / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "io": {"input": str(image_env), "output_dir": str(output_dir)},
            "models": {model_name: {"path": str(model_path)}},
        }))

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        payload = json.loads((output_dir / "report.json").read_text())
        assert payload["models"] == [model_name]
        assert payload["images"][0]["predictions"][model_name]["top_k"]
        if image_env == default_image:
            _assert_goldfish(payload["images"][0], [model_name])
