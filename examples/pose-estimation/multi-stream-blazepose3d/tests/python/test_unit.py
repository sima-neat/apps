"""Unit tests for the multi-stream YOLO26-to-BlazePose application."""

from __future__ import annotations

import copy
import json
import sys
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
        "codec": "h265" if index == 1 else "h264",
        "insight_channel": index if channel is None else channel,
    }


def test_stream_configuration_rejects_more_than_four(tmp_path: Path):
    with pytest.raises(
        ValueError, match="streams must contain between 1 and 4 entries"
    ):
        main.load_app_config(
            write_config(tmp_path, [stream(index) for index in range(5)])
        )


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


def test_insight_video_and_metadata_ports_must_not_overlap(tmp_path: Path):
    path = write_config(tmp_path, [stream(0), stream(1, channel=100)])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["output"]["insight"].update(
        {"video_port_base": 9000, "metadata_port_base": 9100}
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="video and metadata ports must not overlap"):
        main.load_app_config(path)


def test_stream_mapping_order_and_explicit_caps_match_cpp(tmp_path: Path):
    ordered_stream = {
        "url": "rtsp://127.0.0.1/ordered",
        "width": 1920,
        "id": "camera0",
        "fps": 30,
        "insight_channel": 0,
        "height": 1080,
        "codec": "h264",
    }
    cfg = main.load_app_config(write_config(tmp_path, [ordered_stream]))
    assert cfg.streams[0] == main.StreamConfig(
        "camera0", "rtsp://127.0.0.1/ordered", "h264", 0, 1920, 1080, 30
    )


def test_stream_caps_must_be_complete(tmp_path: Path):
    incomplete = stream(0)
    incomplete["width"] = 1920
    with pytest.raises(ValueError, match="must either all be omitted or all be > 0"):
        main.load_app_config(write_config(tmp_path, [incomplete]))


def test_pose_count_is_bounded_by_metadata_transport(tmp_path: Path):
    path = write_config(tmp_path, [stream(0)])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["pose"] = {"max_people_per_frame": 11}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="between 1 and 10"):
        main.load_app_config(path)


def test_roi_landmark_and_metadata_contract():
    box = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0, "score": 0.9, "class_id": 0}
    roi = main.square_roi(box, 1.5)
    assert roi == (-10, 10, 60, 60)
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[0] = [4.0, 8.0, 0.0, 2.0, -2.0]
    raw_world = np.zeros((39, 3), dtype=np.float32)
    raw_world[0] = [0.1, -0.2, 0.3]
    affine = (2.0, 0.0, 10.0, 0.0, 3.0, 20.0)
    pose = main.decode_pose(raw, raw_world, affine, box, 0.93)
    assert pose["presence"] == pytest.approx(0.93)
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
    assert data["poses"][0]["id"] == "pose_1"
    assert data["poses"][0]["presence"] == pytest.approx(0.93)
    assert len(data["poses"][0]["keypoints"]) == 33
    assert data["poses"][0]["keypoints"][0]["name"] == "nose"
    assert len(data["poses"][0]["world_keypoints"]) == 33
    assert data["poses"][0]["world_keypoints"][0]["name"] == "nose"
    assert data["poses"][0]["world_keypoints"][0]["z"] == pytest.approx(0.3)
    auxiliary = main.world_pose_auxiliary_data(data)
    assert auxiliary["stream_id"] == "camera0"
    assert auxiliary["schema_version"] == 1
    assert auxiliary["id"] == "world-pose"
    assert auxiliary["renderer"] == "blazepose-3d"
    assert auxiliary["payload"]["poses"][0]["presence"] == pytest.approx(0.93)
    assert len(auxiliary["payload"]["poses"][0]["keypoints"]) == 33
    assert auxiliary["payload"]["poses"][0]["keypoints"][0]["name"] == "nose"
    assert (
        auxiliary["payload"]["poses"][0]["keypoints"]
        == data["poses"][0]["world_keypoints"]
    )
    json.dumps(data)
    json.dumps(auxiliary)


def test_pose_presence_logit_is_activated_before_thresholding(monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def to_numpy(self, *, copy):
            return self.values.copy() if copy else self.values

    tensors = [Tensor(np.zeros(195)), Tensor([0.0]), Tensor(np.zeros(117))]
    monkeypatch.setattr(main, "tensors_from_sample", lambda *_args: tensors)
    context = main.PoseInputContext(
        {
            "x1": 0.0,
            "y1": 0.0,
            "x2": 100.0,
            "y2": 100.0,
            "score": 0.9,
            "class_id": 0,
        },
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    cfg = SimpleNamespace(pose_presence_threshold=0.5)

    pose = main.parse_pose_output(object(), context, cfg)
    assert pose["presence"] == pytest.approx(0.5)

    tensors[1] = Tensor([-0.01])
    assert main.parse_pose_output(object(), context, cfg) is None


def test_pose_preprocess_copies_readonly_tensor_exports():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image.flags.writeable = False
    tensor = SimpleNamespace(to_numpy=lambda *, copy: image)

    exported = main.writable_rgb_view(tensor)

    assert exported.flags.writeable
    assert not np.shares_memory(exported, image)


def test_incomplete_closed_stream_is_not_a_successful_finite_run():
    runtime = SimpleNamespace(
        streams=[
            SimpleNamespace(
                config=SimpleNamespace(id="offline"), closed=True, metadata_frames=0
            ),
            SimpleNamespace(
                config=SimpleNamespace(id="healthy"), closed=False, metadata_frames=8
            ),
        ]
    )
    with pytest.raises(RuntimeError, match="offline"):
        main.require_successful_completion(runtime, 8)


def test_all_closed_streams_fail_an_unbounded_run():
    runtime = SimpleNamespace(
        streams=[
            SimpleNamespace(
                config=SimpleNamespace(id="camera0"), closed=True, metadata_frames=2
            )
        ]
    )
    with pytest.raises(RuntimeError, match="all source streams stopped"):
        main.require_successful_completion(runtime, 0)


def pose_sample(
    x: float, confidence: float, world_x: float, box_x: float = 0.0
) -> dict:
    return {
        "presence": confidence,
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
    assert image_fraction == pytest.approx(0.45)
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


def test_model_pipeline_preserves_frame_order_through_backpressure_and_empty_frames():
    runtime = SimpleNamespace(state=main.SharedState(1))
    requests = iter(
        [
            (0, "a", float("inf")),
            (1, None, 0),
            (2, "b", float("inf")),
            (3, "c", float("inf")),
        ]
    )
    accepted, results = [], []

    class Run:
        attempts = 0
        received = 0
        pulls = 0

        def try_push(self, _name, samples):
            self.attempts += 1
            if self.attempts == 1:
                return False
            accepted.extend(samples)
            return True

        def can_push(self):
            return True

        def pull(self, _name, _timeout):
            self.pulls += 1
            assert self.pulls <= 8, "pipeline made no progress"
            # Hold the first output until two inputs have been accepted. A
            # serial implementation cannot make progress through this check.
            if len(accepted) < 2:
                return None
            result = accepted[self.received].upper()
            self.received += 1
            return result

        def can_pull(self):
            return True

    def consume(context, output):
        results.append((context, output))
        runtime.state.stopping = len(results) == 4

    main.pump_requests(
        runtime, Run(), "in", "out", 3, lambda: next(requests, None), consume
    )
    assert accepted == ["a", "b", "c"]
    assert results == [(0, "A"), (1, None), (2, "B"), (3, "C")]


@pytest.mark.parametrize("mode", ["expired", "timeout", "closed"])
def test_pipeline_expiry_cannot_attach_a_late_output_to_another_frame(
    monkeypatch, mode
):
    runtime = SimpleNamespace(state=main.SharedState(1))
    clock = [0.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])
    accepted = []
    requests = iter([(0, object(), -1.0 if mode == "expired" else 1.0)])

    class Run:
        def try_push(self, _name, samples):
            accepted.extend(samples)
            return True

        def pull(self, _name, _timeout):
            clock[0] = 2.0

        def can_pull(self):
            return mode != "closed"

        def last_error(self):
            return "closed"

    def consume(context, output):
        assert (context, output) == (0, None)
        runtime.state.stopping = True

    def pump():
        main.pump_requests(
            runtime, Run(), "in", "out", 2, lambda: next(requests, None), consume
        )

    if mode == "expired":
        pump()
        assert accepted == []
    else:
        with pytest.raises(RuntimeError, match="timed out|closed"):
            pump()
        assert len(accepted) == 1


def test_mailboxes_are_fair_and_do_not_wait_for_an_offline_source():
    state = main.SharedState(3)
    runtime = SimpleNamespace(state=state)
    state.detector_mailboxes = [None, "camera1-old", "camera2"]
    state.detector_mailboxes[1] = "camera1-latest"
    take = lambda: main.take_next_job(
        runtime, state.detector_mailboxes, "next_detector_stream"
    )
    assert take() == "camera1-latest"
    state.detector_mailboxes[1] = "camera1-next"
    assert take() == "camera2"
    assert take() == "camera1-next"
    state.stopping = True
    assert take() is None
