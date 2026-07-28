"""E2E tests for ssd-mobilenet-object-detector (Python)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
GOLDEN_PATH = EXAMPLE_DIR / "tests" / "golden_detections.json"


def _iou(a, b):
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    denom = (
        max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        - inter
    )
    return inter / denom if denom > 0.0 else 0.0


def _assert_golden_detections(reported, golden):
    """Every golden detection must be matched by a reported detection of the same class."""
    min_score = float(golden["match"]["min_score"])
    min_iou = float(golden["match"]["min_iou"])
    actual_by_image = {entry["image"]: entry["detections"] for entry in reported["images"]}

    asserted = 0
    failures = []
    for image, expected in golden["images"].items():
        if image not in actual_by_image:
            # The harness may point at a different image folder; only assert what it ran.
            continue
        candidates = [
            det
            for det in actual_by_image[image]
            if det["score"] >= min_score
        ]
        used = set()
        for exp in expected:
            asserted += 1
            best_iou, best_idx = 0.0, None
            for idx, det in enumerate(candidates):
                if idx in used or det["class_id"] != exp["class_id"]:
                    continue
                iou = _iou(exp["box"], det["box"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx is None or best_iou < min_iou:
                failures.append(
                    f"{image}: golden {exp['label']}({exp['class_id']}) {exp['box']} "
                    f"unmatched (best_iou={best_iou:.2f})"
                )
            else:
                used.add(best_idx)

    assert asserted > 0, "no golden detections were asserted; input folder has none of the golden images"
    assert not failures, "golden detection mismatch:\n" + "\n".join(failures)


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
                },
                "runtime": {"num_runs": 1},
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
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
        assert output_files, "Expected annotated output images to be written"
        assert all(path.stat().st_size > 0 for path in output_files), "Output image is empty"

        assert detections_path.is_file(), f"Expected a detections report at {detections_path}"
        reported = json.loads(detections_path.read_text(encoding="utf-8"))
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        _assert_golden_detections(reported, golden)
