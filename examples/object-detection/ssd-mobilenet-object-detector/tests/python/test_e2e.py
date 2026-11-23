"""E2E tests for ssd-mobilenet-object-detector (Python)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.utils.output_assertions import (
    assert_saved_frames_are_usable,
    supported_image_files,
)

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
APPS_ROOT = EXAMPLE_DIR.parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def load_labels(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_valid_detections(report, expected_images, labels, min_score):
    entries = report["images"]
    assert {entry["image"] for entry in entries} == expected_images
    detection_count = 0
    for entry in entries:
        width, height = entry["width"], entry["height"]
        assert width > 0 and height > 0
        for detection in entry["detections"]:
            detection_count += 1
            x1, y1, x2, y2 = detection["box"]
            assert 0 <= x1 < x2 <= width
            assert 0 <= y1 < y2 <= height
            assert min_score <= detection["score"] <= 1
            # Class 0 is the background, which the SSD decoder never reports.
            assert 0 < detection["class_id"] < len(labels)
            assert detection["label"] == labels[detection["class_id"]]
    assert detection_count > 0, "expected at least one decoded detection"


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        e2e_model_path,
        test_images_dir,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            test_images_dir.exists() and bool(supported_image_files(test_images_dir)),
            f"test_images_dir has no images to process: {test_images_dir}",
        )

        detections_path = tmp_output_dir.parent / "detections.json"
        config_path = e2e_config_writer(
            {
                "input": {"source": str(test_images_dir)},
                "output": {"dir": str(tmp_output_dir)},
                "app": {"detections_json": str(detections_path)},
            }
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
            check=False,
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert_saved_frames_are_usable(tmp_output_dir, len(supported_image_files(test_images_dir)))

        assert detections_path.is_file(), f"Expected a detections report at {detections_path}"
        reported = json.loads(detections_path.read_text(encoding="utf-8"))
        expected_images = {path.name for path in supported_image_files(test_images_dir)}
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        labels = load_labels(APPS_ROOT / config["model"]["labels"])
        _assert_valid_detections(
            reported, expected_images, labels, config["inference"]["min_score"]
        )
