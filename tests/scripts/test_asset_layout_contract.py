"""Regression tests for repository-owned Apps asset paths."""

from __future__ import annotations

from pathlib import Path


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


def test_classification_image_is_test_only():
    assert (APPS_ROOT / "assets/datasets-test/imagenet/goldfish.jpeg").is_file()
    assert not (APPS_ROOT / "assets/datasets/imagenet/goldfish.jpeg").exists()


def test_obsolete_asset_directories_are_absent():
    for relative_dir in (
        Path("assets", "coco-images"),
        Path("assets", "images"),
        Path("assets", "portal"),
        Path("assets", "test_images"),
        Path("assets", "test_images_classification"),
    ):
        assert not (APPS_ROOT / relative_dir).exists()
