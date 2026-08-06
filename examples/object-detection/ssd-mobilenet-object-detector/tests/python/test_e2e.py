"""E2E tests for ssd-mobilenet-object-detector (Python)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
GOLDEN_PATH = EXAMPLE_DIR / "tests" / "golden_detections.json"
ACCURACY_REFERENCE_PATH = EXAMPLE_DIR / "tests" / "ssd_accuracy_reference.json"
ACCURACY_MODELS = (
    ("ssd_mobilenet_v1_modalix_bf16_tess_mla_mpk.tar.gz", "v1", "tensorflow_ssd"),
    (
        "ssd_mobilenet_v1_modalix_bf16_tess_off_mla_mpk.tar.gz",
        "v1",
        "tensorflow_ssd",
    ),
    ("ssd_mobilenet_v1_modalix_int8_tess_mla_mpk.tar.gz", "v1", "tensorflow_ssd"),
    (
        "ssd_mobilenet_v1_modalix_int8_tess_off_mla_mpk.tar.gz",
        "v1",
        "tensorflow_ssd",
    ),
    ("ssd_mobilenet_v2_modalix_bf16_tess_mla_mpk.tar.gz", "v2", "tensorflow_ssd"),
    (
        "ssd_mobilenet_v2_modalix_bf16_tess_off_mla_mpk.tar.gz",
        "v2",
        "tensorflow_ssd",
    ),
    ("ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz", "v2", "tensorflow_ssd"),
    (
        "ssd_mobilenet_v2_modalix_int8_tess_off_mla_mpk.tar.gz",
        "v2",
        "tensorflow_ssd",
    ),
    ("ssd_mobilenet_v3_modalix_bf16_tess_mla_mpk.tar.gz", "v3", "tensorflow_ssd"),
    (
        "ssd_mobilenet_v3_modalix_bf16_tess_off_mla_mpk.tar.gz",
        "v3",
        "tensorflow_ssd",
    ),
    (
        "ssd_mobilenet_v3_modalix_int8_tess_mla_mpk.tar.gz",
        "v3",
        "torchvision_ssdlite",
    ),
    (
        "ssd_mobilenet_v3_modalix_int8_tess_off_mla_mpk.tar.gz",
        "v3",
        "torchvision_ssdlite",
    ),
)


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
    actual_by_image = {
        entry["image"]: entry["detections"] for entry in reported["images"]
    }

    asserted = 0
    failures = []
    for image, expected in golden["images"].items():
        if image not in actual_by_image:
            # The harness may point at a different image folder; only assert what it ran.
            continue
        candidates = [
            det for det in actual_by_image[image] if det["score"] >= min_score
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

    assert (
        asserted > 0
    ), "no golden detections were asserted; input folder has none of the golden images"
    assert not failures, "golden detection mismatch:\n" + "\n".join(failures)


def _assert_forbidden_display_regions(reported, golden):
    """Display-policy goldens must be retained in raw JSON but hidden from overlays."""
    failures = []
    reported_by_image = {entry["image"]: entry for entry in reported["images"]}
    for image, forbidden in golden.get("forbidden", {}).items():
        entry = reported_by_image.get(image)
        if entry is None:
            continue
        frame_area = float(entry["width"] * entry["height"])
        for rule in forbidden:
            retained_hidden_box = False
            for det in entry["detections"]:
                if det["class_id"] != rule["class_id"]:
                    continue
                x1, y1, x2, y2 = det["box"]
                area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
                if area_fraction < rule["min_area_fraction"]:
                    continue
                if det.get("displayed", True):
                    failures.append(
                        f"{image}: forbidden {rule['label']} covers {area_fraction:.1%} "
                        f"of the frame: {det['box']}"
                    )
                else:
                    retained_hidden_box = True
            if not retained_hidden_box:
                failures.append(
                    f"{image}: {rule['label']} was not retained as a hidden raw detection"
                )

    assert not failures, "display policy mismatch:\n" + "\n".join(failures)


def _source_parity_metrics(reported, reference):
    actual_by_image = {
        entry["image"]: entry["detections"] for entry in reported["images"]
    }
    matched_ious = []
    reference_count = 0
    for image, expected in reference["images"].items():
        candidates = actual_by_image.get(image, [])
        used = set()
        for exp in expected:
            reference_count += 1
            best_iou, best_index = 0.0, None
            for index, actual in enumerate(candidates):
                if index in used or actual["class_id"] != exp["class_id"]:
                    continue
                overlap = _iou(exp["box"], actual["box"])
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index is not None and best_iou >= float(reference["min_iou"]):
                used.add(best_index)
                matched_ious.append(best_iou)
    assert reference_count > 0
    recall = len(matched_ious) / reference_count
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return recall, mean_iou, len(matched_ious), reference_count


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
        assert all(
            path.stat().st_size > 0 for path in output_files
        ), "Output image is empty"

        assert (
            detections_path.is_file()
        ), f"Expected a detections report at {detections_path}"
        reported = json.loads(detections_path.read_text(encoding="utf-8"))
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        _assert_golden_detections(reported, golden)
        _assert_forbidden_display_regions(reported, golden)


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("model_file", "family", "preprocessing_profile"),
    ACCURACY_MODELS,
    ids=[entry[0].removesuffix("_mpk.tar.gz") for entry in ACCURACY_MODELS],
)
def test_precision_matrix_source_accuracy(
    model_file,
    family,
    preprocessing_profile,
    models_dir,
    test_images_dir,
    tmp_output_dir,
    test_timeout_ms,
    skip_unless_e2e_ready,
    e2e_config_writer,
):
    model_path = models_dir / model_file
    skip_unless_e2e_ready(model_path.is_file(), f"model not found: {model_path}")
    skip_unless_e2e_ready(
        test_images_dir.exists() and any(test_images_dir.iterdir()),
        f"test_images_dir is missing or empty: {test_images_dir}",
    )
    detections_path = tmp_output_dir.parent / "accuracy-detections.json"
    config_path = e2e_config_writer(
        {
            "model": {
                "path": str(model_path),
                "preprocessing_profile": preprocessing_profile,
            },
            "io": {
                "input_dir": str(test_images_dir),
                "output_dir": str(tmp_output_dir),
                "detections_json": str(detections_path),
            },
            "decode": {"score_threshold": 0.30, "nms_iou": 0.60, "max_detections": 100},
            "runtime": {"num_runs": 1},
            "output": {"overlay": False, "aggregate_suppression": False},
        }
    )
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=max(test_timeout_ms / 1000, 120),
        cwd=str(EXAMPLE_DIR),
    )
    assert result.returncode == 0, (
        f"{model_file}: main.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    reported = json.loads(detections_path.read_text(encoding="utf-8"))
    references = json.loads(ACCURACY_REFERENCE_PATH.read_text(encoding="utf-8"))
    reference = references["families"][family]
    recall, mean_iou, matched, total = _source_parity_metrics(reported, reference)
    print(
        f"model={model_file} family={family} matched={matched}/{total} "
        f"source_recall_at_iou_0_45={recall:.6f} mean_matched_iou={mean_iou:.6f}"
    )
    assert recall >= float(
        reference["min_recall"]
    ), f"{model_file}: source recall {recall:.6f} below {reference['min_recall']}"
    assert mean_iou >= float(reference["min_mean_matched_iou"]), (
        f"{model_file}: mean matched IoU {mean_iou:.6f} below "
        f"{reference['min_mean_matched_iou']}"
    )
