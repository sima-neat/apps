"""E2E tests for ssd-mobilenet-object-detector (Python)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def _assert_valid_detections(report, expected_images):
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
            assert 0 <= detection["score"] <= 1
            assert 0 < detection["class_id"] < 91
            assert detection["label"]
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
            test_images_dir.exists() and any(test_images_dir.iterdir()),
            f"test_images_dir is missing or empty: {test_images_dir}",
        )

        detections_path = tmp_output_dir.parent / "detections.json"
        config_path = e2e_config_writer(
            {
                "io": {
                    "input_dir": str(test_images_dir),
                    "output_dir": str(tmp_output_dir),
                    "detections_json": str(detections_path),
                }
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
        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
        assert output_files, "Expected annotated output images to be written"
        assert all(path.stat().st_size > 0 for path in output_files), (
            "Output image is empty"
        )

        assert detections_path.is_file(), (
            f"Expected a detections report at {detections_path}"
        )
        reported = json.loads(detections_path.read_text(encoding="utf-8"))
        expected_images = {
            path.name
            for path in test_images_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        }
        _assert_valid_detections(reported, expected_images)
