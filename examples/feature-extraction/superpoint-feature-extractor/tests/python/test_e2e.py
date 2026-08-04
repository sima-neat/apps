"""End-to-end tests for the Python SuperPoint example."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
REFERENCE = EXAMPLE_DIR / "tests" / "data" / "fp32-a65-tum-desk.npz"
ACCURACY_MODELS = (
    "superpoint_modalix_int8_tessellation_mla_mpk.tar.gz",
    "superpoint_modalix_int8_tessellation_ev74_mpk.tar.gz",
    "superpoint_modalix_bf16_tessellation_mla_mpk.tar.gz",
    "superpoint_modalix_bf16_tessellation_ev74_mpk.tar.gz",
)


def load_example():
    spec = importlib.util.spec_from_file_location("superpoint_example", MAIN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean_keypoint_parity(reference, actual):
    if not len(reference) or not len(actual):
        return 0.0, 0.0
    distances = np.sum((reference[:, None] - actual[None, :]) ** 2, axis=2)
    recall = float(np.mean(np.min(distances, axis=1) <= 9.0))
    precision = float(np.mean(np.min(distances, axis=0) <= 9.0))
    return recall, precision


def common_descriptor_cosine(
    reference_points, reference_descriptors, points, descriptors
):
    actual_by_point = {
        tuple(point): descriptor for point, descriptor in zip(points, descriptors)
    }
    pairs = [
        (reference, actual_by_point[tuple(point)])
        for point, reference in zip(reference_points, reference_descriptors)
        if tuple(point) in actual_by_point
    ]
    if not pairs:
        return 0.0, 0.0
    expected, actual = (np.stack(values) for values in zip(*pairs))
    denominator = np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1)
    cosine = np.sum(expected * actual, axis=1) / np.maximum(denominator, 1.0e-12)
    return len(pairs) / len(reference_points), float(np.mean(cosine))


@pytest.mark.e2e
def test_video_pipeline(
    apps_root,
    e2e_model_path,
    e2e_config_writer,
    tmp_output_dir,
    test_timeout_ms,
):
    input_video = apps_root / "assets" / "datasets" / "tum-rgbd" / "freiburg1-desk.mp4"
    assert input_video.is_file()
    output_video = tmp_output_dir / "annotated.mp4"
    config = e2e_config_writer(
        {
            "model": {"path": str(e2e_model_path)},
            "io": {"input": str(input_video), "output": str(output_video)},
            "runtime": {"frames": 8},
        }
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=apps_root,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    match = re.search(
        r"frames=8 average_points=([0-9]+(?:\.[0-9]+)?) descriptor_dim=256 ",
        result.stdout,
    )
    assert match, result.stdout
    assert 0 < float(match.group(1)) <= 600

    video = cv2.VideoCapture(str(output_video))
    ok, frame = video.read()
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_value = int(video.get(cv2.CAP_PROP_FOURCC))
    video.release()
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    assert ok
    assert frame.shape[:2] == (480, 640)
    assert frame_count == 8
    assert fourcc == "avc1"


@pytest.mark.e2e
@pytest.mark.parametrize("model_file", ACCURACY_MODELS)
def test_fp32_a65_accuracy(models_dir, skip_unless_e2e_ready, model_file):
    import pyneat

    example = load_example()
    model_path = models_dir / model_file
    skip_unless_e2e_ready(model_path.is_file(), f"model not found: {model_path}")
    with np.load(REFERENCE) as reference:
        images = reference["image"]
        offsets = reference["offsets"]
        expected_points = reference["keypoints"]
        sample_offsets = reference["sample_offsets"]
        sample_points = reference["sample_keypoints"]
        sample_descriptors = reference["sample_descriptors"]

        first_frame = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)
        model = pyneat.Model(str(model_path), example.model_options(pyneat))
        input_specs = model.input_specs()
        assert len(input_specs) == 1
        input_dtype = example.select_input_dtype(input_specs[0], pyneat)
        model_input = example.input_tensor(first_frame, input_dtype, cv2, np, pyneat)
        runner = model.build(
            [model_input],
            route_options=pyneat.ModelRouteOptions(),
            run_options=pyneat.RunOptions(),
        )

        recalls = []
        precisions = []
        descriptor_coverages = []
        descriptor_cosines = []
        try:
            for index, gray in enumerate(images):
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                model_input = example.input_tensor(frame, input_dtype, cv2, np, pyneat)
                output = runner.run([model_input], timeout_ms=20000)
                decoded = pyneat.decode_superpoint(list(output))
                assert len(decoded) == 1
                points = np.asarray(
                    decoded[0].keypoints.to_numpy(copy=True), dtype=np.float32
                )
                descriptors = np.asarray(
                    decoded[0].descriptors.to_numpy(copy=True), dtype=np.float32
                )

                begin, end = offsets[index : index + 2]
                recall, precision = mean_keypoint_parity(
                    expected_points[begin:end], points
                )
                recalls.append(recall)
                precisions.append(precision)

                begin, end = sample_offsets[index : index + 2]
                coverage, cosine = common_descriptor_cosine(
                    sample_points[begin:end],
                    sample_descriptors[begin:end],
                    points,
                    descriptors,
                )
                descriptor_coverages.append(coverage)
                descriptor_cosines.append(cosine)
        finally:
            runner.close()

    recall = float(np.mean(recalls))
    precision = float(np.mean(precisions))
    descriptor_coverage = float(np.mean(descriptor_coverages))
    descriptor_cosine = float(np.mean(descriptor_cosines))
    print(
        f"model={model_file} "
        f"keypoint_recall_at_3px={recall:.6f} "
        f"keypoint_precision_at_3px={precision:.6f} "
        f"descriptor_coverage={descriptor_coverage:.6f} "
        f"common_descriptor_cosine={descriptor_cosine:.6f}"
    )
    assert recall >= 0.90, f"{model_file}: keypoint recall {recall:.6f}"
    assert precision >= 0.90, f"{model_file}: keypoint precision {precision:.6f}"
    assert (
        descriptor_coverage >= 0.70
    ), f"{model_file}: descriptor coverage {descriptor_coverage:.6f}"
    assert (
        descriptor_cosine >= 0.995
    ), f"{model_file}: descriptor cosine {descriptor_cosine:.6f}"
