"""Unit tests for the multi-stream YOLO26-to-BlazePose application."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
PYTHON_DIR = MAIN_PY.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import main

main.np = np
pytestmark = pytest.mark.unit


def write_config(tmp_path: Path, streams: list[dict]) -> Path:
    config = {
        "models": {"detector_path": "detector.tar.gz", "pose_path": "pose.tar.gz"},
        "streams": streams,
        "output": {"insight": {"host": "127.0.0.1"}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def stream(
    index: int, *, stream_id: str | None = None, channel: int | None = None
) -> dict:
    return {
        "id": stream_id or f"camera{index}",
        "url": f"rtsp://127.0.0.1/src{index}",
        "codec": "hevc" if index == 1 else "h264",
        "insight_channel": index if channel is None else channel,
    }


def test_cli_help_and_missing_config():
    help_result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    missing = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", "does-not-exist.yaml"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "config file not found" in missing.stderr


def test_stream_configuration_rejects_more_than_four(tmp_path: Path):
    with pytest.raises(ValueError, match="streams must contain between 1 and 4 entries"):
        main.load_app_config(write_config(tmp_path, [stream(index) for index in range(5)]))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda values: values.__setitem__(1, stream(1, stream_id="camera0")),
            "ids must be unique",
        ),
        (
            lambda values: values.__setitem__(1, stream(1, channel=0)),
            "channels must be unique",
        ),
    ],
)
def test_duplicate_stream_fields_are_rejected(tmp_path: Path, mutator, message: str):
    streams = [stream(0), stream(1)]
    mutator(streams)
    with pytest.raises(ValueError, match=message):
        main.load_app_config(write_config(tmp_path, streams))


def test_roi_landmark_and_metadata_contract():
    box = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0, "score": 0.9, "class_id": 0}
    roi = main.square_roi(box, 1.5)
    assert roi == (-10, 10, 60, 60)
    assert main.crop_plan(roi, 1280, 720) == ((0, 10, 50, 60), (-10, 0, 60, 60))
    assert main.crop_plan((1280, 10, 20, 20), 1280, 720) is None
    assert main.batch_crop_plan(
        [roi, (1280, 10, 20, 20), (100, 100, 20, 20)], 1280, 720
    ) == ((0, 10, 120, 110), [(0, (-10, 0, 60, 60)), (2, (100, 90, 20, 20))])
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[0] = [4.0, 8.0, 0.0, 2.0, -2.0]
    raw_world = np.zeros((39, 3), dtype=np.float32)
    raw_world[0] = [0.1, -0.2, 0.3]
    affine = main.offset_affine((2.0, 0.0, 10.0, 0.0, 3.0, 10.0), 0, 10)
    pose = main.decode_pose(raw, raw_world, affine, box, 2)
    assert pose["keypoints"][0]["x"] == pytest.approx(18.0)
    assert pose["keypoints"][0]["y"] == pytest.approx(44.0)
    assert pose["keypoints"][0]["confidence"] == pytest.approx(main.sigmoid(-2.0))
    world_point = pose["world_keypoints"][0]
    assert world_point["name"] == "nose"
    assert world_point["x"] == pytest.approx(0.1)
    assert world_point["y"] == pytest.approx(-0.2)
    assert world_point["z"] == pytest.approx(0.3)
    assert world_point["confidence"] == pytest.approx(main.sigmoid(-2.0))
    data = main.poses_data([pose], "camera0")
    assert data["stream_id"] == "camera0"
    assert data["poses"][0]["id"] == "pose_3"
    assert len(data["poses"][0]["keypoints"]) == 33
    assert data["poses"][0]["keypoints"][0]["name"] == "nose"
    assert "world_keypoints" not in data["poses"][0]
    auxiliary = main.world_pose_auxiliary_data([pose], "camera0")
    assert auxiliary["stream_id"] == "camera0"
    assert auxiliary["schema_version"] == 1
    assert auxiliary["id"] == "world-pose"
    assert auxiliary["renderer"] == "blazepose-3d"
    assert len(auxiliary["payload"]["poses"][0]["keypoints"]) == 33
    assert auxiliary["payload"]["poses"][0]["keypoints"][0]["name"] == "nose"
    point_cloud = {
        "points": [{"x": 0.1, "y": 0.2, "z": 0.3, "value": 7}],
        "axes": ["east", "north", "up"],
    }
    generic = main.auxiliary_visualization_data(
        "depth-cloud", "point-cloud-3d", point_cloud
    )
    assert generic == {
        "schema_version": 1,
        "id": "depth-cloud",
        "renderer": "point-cloud-3d",
        "payload": point_cloud,
    }
    json.dumps(data)
    json.dumps(auxiliary)
    json.dumps(generic)


def test_frame_identity_falls_back_through_source_sequence_fields():
    assert main.select_frame_id(9, 8, 7, 6) == 9
    assert main.select_frame_id(-1, 8, 7, 6) == 8
    assert main.select_frame_id(-1, -1, 7, 6) == 7
    assert main.select_frame_id(-1, -1, -1, 6) == 6


def test_publish_metadata_sends_paired_overlay_and_auxiliary_messages():
    calls = []

    class Sender:
        def send_metadata(self, *args):
            calls.append(args)

    stream_runtime = SimpleNamespace(
        metadata_lock=threading.Lock(),
        metadata_sender=Sender(),
        pose_smoother=main.PoseSmoother(),
        pose_temporal_filter_enabled=True,
        metadata_frames=0,
    )
    identity = main.FrameIdentity("camera0", 7, 1_234_000_000, -1, -1, 7, 7)

    main.publish_metadata(stream_runtime, identity, [])

    assert [call[0] for call in calls] == [
        "pose-estimation",
        "auxiliary-visualization",
    ]
    assert all(call[2:] == (1234, "7") for call in calls)
    assert json.loads(calls[0][1]) == {"stream_id": "camera0", "poses": []}
    assert json.loads(calls[1][1])["stream_id"] == "camera0"
    assert json.loads(calls[1][1])["payload"] == {"poses": []}
    assert stream_runtime.metadata_frames == 1


def pose_sample(x: float, confidence: float, world_x: float, box_x: float = 0.0) -> dict:
    return {
        "roi_index": 0,
        "box": {
            "x1": box_x,
            "y1": 0.0,
            "x2": box_x + 100.0,
            "y2": 100.0,
            "score": 0.9,
            "class_id": 0,
        },
        "keypoints": [{"name": "nose", "x": x, "y": 50.0, "confidence": confidence}],
        "world_keypoints": [
            {"name": "nose", "x": world_x, "y": 0.0, "z": 0.0, "confidence": confidence}
        ],
    }


def test_pose_smoother_filters_2d_world_and_confidence_together_without_buffering():
    smoother = main.PoseSmoother()
    first = smoother.filter([pose_sample(50.0, 0.2, 0.0)], 1_000_000_000)[0]
    assert first["keypoints"][0]["x"] == 50.0

    second = smoother.filter([pose_sample(54.0, 0.4, 0.04)], 1_040_000_000)[0]
    image_fraction = (second["keypoints"][0]["x"] - 50.0) / 4.0
    world_fraction = second["world_keypoints"][0]["x"] / 0.04
    assert 0.45 < image_fraction < 0.90
    assert world_fraction == pytest.approx(image_fraction)
    assert second["keypoints"][0]["confidence"] == pytest.approx(0.24)
    assert second["world_keypoints"][0]["confidence"] == pytest.approx(0.24)

    fast = smoother.filter([pose_sample(154.0, 0.4, 1.04)], 1_080_000_000)[0]
    assert fast["keypoints"][0]["x"] > 140.0

    raw_after_gap = pose_sample(30.0, 0.9, -0.2)
    reset = smoother.filter([copy.deepcopy(raw_after_gap)], 1_400_000_000)[0]
    assert reset == raw_after_gap


def test_pose_smoother_state_is_independent_per_stream():
    left = main.PoseSmoother()
    right = main.PoseSmoother()
    left.filter([pose_sample(10.0, 1.0, 0.0)], 1_000_000_000)
    right.filter([pose_sample(90.0, 1.0, 1.0)], 1_000_000_000)
    left_result = left.filter([pose_sample(12.0, 1.0, 0.02)], 1_040_000_000)[0]
    right_result = right.filter([pose_sample(88.0, 1.0, 0.98)], 1_040_000_000)[0]
    assert left_result["keypoints"][0]["x"] < 12.0
    assert right_result["keypoints"][0]["x"] > 88.0


def test_pose_smoother_bridges_two_missing_results_without_buffering():
    smoother = main.PoseSmoother()
    original = pose_sample(50.0, 0.9, 0.0)
    smoother.filter([copy.deepcopy(original)], 1_000_000_000)

    first_gap = smoother.filter([], 1_040_000_000)
    second_gap = smoother.filter([], 1_080_000_000)
    expired = smoother.filter([], 1_120_000_000)

    assert first_gap[0]["keypoints"][0]["x"] == 50.0
    assert second_gap[0]["keypoints"][0]["x"] == 50.0
    assert first_gap[0]["keypoints"][0]["confidence"] < 0.9
    assert second_gap[0]["keypoints"][0]["confidence"] < first_gap[0]["keypoints"][0]["confidence"]
    assert expired == []
