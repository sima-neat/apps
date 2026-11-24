"""Unit tests for the SuperPoint feature extractor: the checks the application leaves to its config
and the mapping of feature points onto the source frame. tests/cpp/test_unit.cpp runs the same
checks in C++.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "superpoint_feature_extractor_main")
feature_points = main_module.feature_points

pytestmark = pytest.mark.unit

MODEL_SIZE = (640, 480)
SOURCE_SIZE = (1280, 720)


def test_shipped_config_names_the_model_and_usable_values():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "superpoint_w8a8_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models/superpoint/w8a8/ from the config file.
    assert Path(os.path.normpath(model_path.absolute())) == (
        EXAMPLE_DIR.parents[2].absolute() / "models" / "superpoint" / "w8a8" / model_path.name
    )
    assert config.string("input.source").startswith("rtsp://")
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames"):
        assert config.integer(key) >= 0, key
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def test_source_points_scale_each_axis_onto_the_source_frame():
    points = feature_points.source_points(
        np.array([[600.0, 400.0], [10.0, 20.0]]), MODEL_SIZE, SOURCE_SIZE
    )

    np.testing.assert_array_equal(points, [[1200, 600], [20, 30]])


def test_source_points_round_half_up_to_whole_pixels():
    points = feature_points.source_points(
        np.array([[6.25, 3.0], [6.0, 3.3]]), MODEL_SIZE, SOURCE_SIZE
    )

    np.testing.assert_array_equal(points, [[13, 5], [12, 5]])


def test_source_points_drop_points_on_or_beyond_an_edge():
    points = feature_points.source_points(
        np.array(
            [[600.0, 400.0], [640.0, 200.0], [639.8, 200.0], [20.0, 480.0], [-1.0, 5.0],
             [np.nan, 10.0]]
        ),
        MODEL_SIZE,
        SOURCE_SIZE,
    )

    # 639.8 maps to 1279.6, which rounds onto the right edge.
    np.testing.assert_array_equal(points, [[1200, 600]])


def test_source_points_of_a_frame_without_features_are_empty():
    points = feature_points.source_points(np.zeros((0, 2)), MODEL_SIZE, SOURCE_SIZE)

    assert points.shape == (0, 2)


def test_points_json_sends_the_points_as_one_pose_of_unnamed_keypoints():
    """Insight draws each keypoint as a dot; without names and confidences it draws no skeleton
    line and no box."""
    data = json.loads(feature_points.points_json(np.array([[13, 7], [1279, 0]])))

    assert data == {"poses": [{"keypoints": [{"x": 13, "y": 7}, {"x": 1279, "y": 0}]}]}


def test_points_json_of_a_frame_without_features_has_no_keypoints():
    assert json.loads(feature_points.points_json(np.zeros((0, 2), np.int64))) == {
        "poses": [{"keypoints": []}]
    }


def test_points_json_of_600_points_fits_one_metadata_message():
    """The metadata sender rejects messages over 65,507 bytes; the decoder returns at most 600
    points per frame."""
    payload = feature_points.points_json(np.full((600, 2), 1279))

    assert len(payload.encode()) < 65_507 - 1_024  # Leaves room for the message envelope.
