"""Regression tests for repository-owned Apps asset paths."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from tests.utils import pytest_fixtures


APPS_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_COCO_IMAGES = {
    "000000081061.jpg",
    "000000116439.jpg",
    "000000129492.jpg",
    "000000210273.jpg",
    "000000227985.jpg",
    "000000237864.jpg",
    "000000265108.jpg",
    "000000300276.jpg",
    "000000331075.jpg",
    "000000424551.jpg",
    "000000437331.jpg",
    "000000449579.jpg",
    "000000493864.jpg",
    "000000494759.jpg",
    "000000506004.jpg",
    "000000516173.jpg",
    "000000519611.jpg",
    "000000550349.jpg",
    "000000560266.jpg",
    "000000566758.jpg",
    "000000570756.jpg",
}
RUNTIME_PCB_IMAGES = {
    "pcb_01_missing_hole.jpg",
    "pcb_02_open_circuit.jpg",
    "pcb_03_spur.jpg",
    "pcb_04_spurious_copper.jpg",
    "pcb_05_spurious_copper_rotated.jpg",
}
TEST_PCB_IMAGES = {
    "pcb_01_missing_hole.jpg",
    "pcb_03_spur.jpg",
    "pcb_04_spurious_copper.jpg",
}
TEST_COCO_IMAGES = {
    "000000081061.jpg",
    "000000116439.jpg",
    "000000237864.jpg",
    "000000265108.jpg",
    "000000331075.jpg",
    "000000437331.jpg",
    "000000449579.jpg",
    "000000516173.jpg",
    "000000550349.jpg",
    "000000566758.jpg",
}


def _image_names(relative_dir: str) -> set[str]:
    return {path.name for path in (APPS_ROOT / relative_dir).glob("*.jpg")}


def test_runtime_and_test_coco_datasets_have_expected_images():
    assert _image_names("assets/datasets/coco") == RUNTIME_COCO_IMAGES
    assert _image_names("assets/datasets-test/coco") == TEST_COCO_IMAGES


def test_runtime_and_test_pcb_datasets_have_expected_images():
    assert _image_names("assets/datasets/pcb") == RUNTIME_PCB_IMAGES
    assert _image_names("assets/datasets-test/pcb") == TEST_PCB_IMAGES


def test_classification_image_is_test_only():
    assert (APPS_ROOT / "assets/datasets-test/imagenet/goldfish.jpeg").is_file()
    assert not (APPS_ROOT / "assets/datasets/imagenet/goldfish.jpeg").exists()


def test_superpoint_video_is_a_licensed_runtime_dataset():
    dataset = APPS_ROOT / "assets" / "datasets" / "tum-rgbd"
    video = dataset / "freiburg1-desk.mp4"
    license_file = dataset / "LICENSE.md"

    assert video.is_file()
    assert license_file.is_file()
    assert hashlib.sha256(video.read_bytes()).hexdigest() == (
        "fe6189df4c6f3fa468fea1439a31d2ce306110988cfba4854fb1d24d37a978fe"
    )
    assert "CC BY 4.0" in license_file.read_text(encoding="utf-8")


def test_direct_pytest_defaults_to_test_assets(monkeypatch):
    monkeypatch.delenv("SIMANEAT_APPS_TEST_MODELS_DIR", raising=False)
    monkeypatch.delenv("SIMANEAT_APPS_TEST_INPUT_DIR", raising=False)

    assert pytest_fixtures.models_dir.__wrapped__() == APPS_ROOT / "models"
    assert pytest_fixtures.test_images_dir.__wrapped__() == (
        APPS_ROOT / "assets/datasets-test/coco"
    )


def test_direct_classification_pytest_uses_test_fixture():
    e2e_path = (
        APPS_ROOT / "examples/classification/image-classifier/tests/python/test_e2e.py"
    )
    source = e2e_path.read_text(encoding="utf-8")

    assert '"datasets-test"' in source
    assert '"imagenet"' in source
    assert "test_images_classification" not in source


def test_benchmark_refresh_finds_top_level_model(tmp_path, monkeypatch):
    package = "example-package.tar.gz"
    model_path = tmp_path / "models" / package
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")

    refresh_path = (
        APPS_ROOT / "examples/benchmarking/model-benchmark/scripts/refresh_results.py"
    )
    module_name = "model_benchmark_refresh_contract"
    spec = importlib.util.spec_from_file_location(module_name, refresh_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "APPS_ROOT", tmp_path)

    row = module.ModelRow(
        "example",
        package,
        "example-app",
        f"models/nested/{package}",
    )

    assert module.resolve_model_path(row) == model_path


def test_obsolete_asset_directories_are_absent():
    for relative_dir in (
        Path("assets", "coco-images"),
        Path("assets", "images"),
        Path("assets", "portal"),
        Path("assets", "test_images"),
        Path("assets", "test_images_classification"),
    ):
        assert not (APPS_ROOT / relative_dir).exists()
