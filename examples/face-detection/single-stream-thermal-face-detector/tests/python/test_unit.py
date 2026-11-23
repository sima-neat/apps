"""Unit tests for the thermal face detector: the checks the application leaves to its config,
and the yolov5s-face decode in utils/yolov5_face.py. tests/cpp/test_unit.cpp runs the same
checks in C++.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]

main_module = load_example_main(EXAMPLE_DIR, "single_stream_thermal_face_detector_main")
yolov5_face = main_module.yolov5_face

pytestmark = pytest.mark.unit

# The planted faces sit on the 100x100 grid (stride 8), in cell x 20, y 10.
CELL_X, CELL_Y = 20, 10
# Logits whose sigmoids are 0.8 and 0.5: a face score of 0.4.
SCORE_LOGITS = (math.log(4.0), 0.0)


def heads_with_faces(faces: list[tuple[int, tuple[float, float]]]) -> list[np.ndarray]:
    """Six [1, H, W, C] heads, largest grid first, with a face planted at (CELL_X, CELL_Y) for
    each (anchor, (object logit, class logit)). Every other score is about 2e-9."""
    heads = []
    for grid in (100, 50, 25):
        box = np.zeros((1, grid, grid, 18), dtype=np.float32)
        box[..., 4::6] = -10.0
        box[..., 5::6] = -10.0
        landmarks = np.zeros((1, grid, grid, 30), dtype=np.float32)
        if grid == 100:
            for anchor, (object_logit, class_logit) in faces:
                box[0, CELL_Y, CELL_X, anchor * 6 + 4] = object_logit
                box[0, CELL_Y, CELL_X, anchor * 6 + 5] = class_logit
                # Landmark k is (k, -k) anchor sizes from the cell's corner.
                for k in range(5):
                    landmarks[0, CELL_Y, CELL_X, anchor * 10 + 2 * k] = k
                    landmarks[0, CELL_Y, CELL_X, anchor * 10 + 2 * k + 1] = -k
        heads += [box, landmarks]
    return heads


def test_shipped_config_names_the_model_labels_and_usable_values():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolov5s_face_raw_split_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    # The model has one class, and its name labels every pose.
    assert main_module.load_labels(APPS_ROOT / config.string("model.labels")) == ["face"]
    assert config.string("input.source").startswith("rtsp://")
    for key in (
        "input.width",
        "input.height",
        "input.fps",
        "input.rtsp.latency_ms",
        "inference.frames",
    ):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def test_decode_faces_places_the_box_and_landmarks_of_a_planted_face():
    """Anchor 1 of the first level is 8x10 pixels: a zero box offset centers the face on
    (20.5, 10.5) cells of 8 pixels, the size of the anchor."""
    faces = yolov5_face.decode_faces(heads_with_faces([(1, SCORE_LOGITS)]), 0.25, 0.45, 50)
    assert_close(faces.boxes, [[160.0, 79.0, 168.0, 89.0]])
    assert_close(faces.scores, [0.4])
    expected_landmarks = [[160.0 + 8 * k, 80.0 - 10 * k] for k in range(5)]
    assert_close(faces.landmarks, [expected_landmarks])


def test_decode_faces_drops_faces_at_or_below_min_score():
    heads = heads_with_faces([(1, SCORE_LOGITS)])
    assert len(yolov5_face.decode_faces(heads, 0.45, 0.45, 50).scores) == 0


@pytest.mark.parametrize("axes", [(0, 1, 3, 2), (0, 3, 1, 2)], ids=["neat-hcw", "onnx-chw"])
def test_decode_faces_reads_every_head_layout_alike(axes):
    """Neat delivers raw heads as [1, H, C, W]; ONNX runs give [1, C, H, W]."""
    heads = heads_with_faces([(1, SCORE_LOGITS), (2, (5.0, 5.0))])
    permuted = [np.ascontiguousarray(head.transpose(axes)) for head in heads]
    expected = yolov5_face.decode_faces(heads, 0.25, 0.45, 50)
    actual = yolov5_face.decode_faces(permuted, 0.25, 0.45, 50)
    assert len(expected.scores) == 2
    for got, want in zip(actual, expected):
        assert_close(got, want)


def test_decode_faces_keeps_the_higher_score_of_two_overlapping_faces():
    """Two faces on one cell, 8x10 (score 0.4) and 13x16 (score near 1), overlap by
    80 / 208 = 0.385 of their union."""
    heads = heads_with_faces([(1, SCORE_LOGITS), (2, (10.0, 10.0))])
    suppressed = yolov5_face.decode_faces(heads, 0.25, 0.3, 50)
    assert_close(suppressed.boxes, [[157.5, 76.0, 170.5, 92.0]])
    kept = yolov5_face.decode_faces(heads, 0.25, 0.45, 50)
    assert_close(kept.scores, [sigmoid(10.0) ** 2, 0.4])
    capped = yolov5_face.decode_faces(heads, 0.25, 0.45, 1)
    assert_close(capped.scores, [sigmoid(10.0) ** 2])


def test_decode_faces_rejects_heads_it_cannot_pair():
    heads = heads_with_faces([])
    with pytest.raises(ValueError, match="expected 6 split heads"):
        yolov5_face.decode_faces(heads[:5], 0.25, 0.45, 50)
    with pytest.raises(ValueError, match="no axis of 18 or 30 channels"):
        yolov5_face.decode_faces(heads[:5] + [np.zeros((1, 7, 7, 7))], 0.25, 0.45, 50)
    with pytest.raises(ValueError, match="a box and a landmark head on 3 grids"):
        yolov5_face.decode_faces(heads[:4] + [heads[0], heads[5]], 0.25, 0.45, 50)


def test_letterbox_centers_the_frame_on_the_canvas():
    assert yolov5_face.letterbox(1600, 900) == (0.5, 0, 175)
    assert yolov5_face.letterbox(640, 512) == (1.25, 0, 80)


def test_to_frame_maps_canvas_pixels_to_the_frame_and_clips_only_boxes():
    faces = yolov5_face.Faces(
        boxes=np.array([[100, 275, 300, 475], [700, 150, 900, 300]], dtype=np.float32),
        scores=np.array([0.9, 0.8], dtype=np.float32),
        landmarks=np.array([[[400, 575]] * 5, [[-10, 100]] * 5], dtype=np.float32),
    )
    mapped = yolov5_face.to_frame(faces, yolov5_face.letterbox(1600, 900), 1600, 900)
    assert mapped.boxes.tolist() == [[200, 200, 600, 600], [1400, 0, 1600, 250]]
    assert mapped.landmarks[:, 0].tolist() == [[800, 800], [-20, -150]]
    assert_close(mapped.scores, [0.9, 0.8])


def test_poses_json_gives_each_face_five_named_keypoints_with_its_score():
    faces = yolov5_face.Faces(
        boxes=np.zeros((1, 4), dtype=np.float32),
        scores=np.array([0.5], dtype=np.float32),
        landmarks=np.arange(10, dtype=np.float32).reshape(1, 5, 2),
    )
    names = ["eye_l", "eye_r", "nose_tip", "mouth_l", "mouth_r"]
    assert json.loads(yolov5_face.poses_json(faces, "face")) == {
        "poses": [
            {
                "id": "face_1",
                "label": "face",
                "keypoints": [
                    {"name": name, "x": 2.0 * k, "y": 2.0 * k + 1, "confidence": 0.5}
                    for k, name in enumerate(names)
                ],
            }
        ]
    }


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def assert_close(actual, expected) -> None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-4)
