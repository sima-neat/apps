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


def runtime_stream(sender, *, outstanding: int = 1, temporal_filter: bool = False):
    return SimpleNamespace(
        config=SimpleNamespace(id="camera0"),
        metadata_lock=threading.Lock(),
        metadata_sender=sender,
        pose_smoother=main.PoseSmoother(),
        pose_temporal_filter_enabled=temporal_filter,
        pending_publications={},
        next_publication_sequence=1,
        metadata_frames=0,
        metadata_send_failures=0,
        outstanding_frames=outstanding,
        timed_out_jobs=0,
        detector_frames=0,
        completed_rois=0,
    )


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


def test_insight_video_and_metadata_ports_must_not_overlap(tmp_path: Path):
    path = write_config(tmp_path, [stream(0), stream(1, channel=100)])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["output"]["insight"].update(
        {"video_port_base": 9000, "metadata_port_base": 9100}
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="video and metadata ports must not overlap"):
        main.load_app_config(path)


@pytest.mark.parametrize("video_enabled", [False, True])
def test_video_port_range_applies_only_when_video_is_enabled(tmp_path: Path, video_enabled):
    path = write_config(tmp_path, [stream(0, channel=60000)])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["output"]["insight"].update({"video_port_base": 9000, "metadata_port_base": 1})
    config["output"]["video_enabled"] = video_enabled
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if video_enabled:
        with pytest.raises(ValueError, match="stream video port must be <= 65535"):
            main.load_app_config(path)
    else:
        cfg = main.load_app_config(path)
        assert cfg.metadata_port_base + cfg.streams[0].insight_channel == 60001


def test_metadata_port_range_applies_without_video(tmp_path: Path):
    path = write_config(tmp_path, [stream(0, channel=60000)])
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["output"]["insight"].update({"video_port_base": 1, "metadata_port_base": 9100})
    config["output"]["video_enabled"] = False
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="stream metadata port must be <= 65535"):
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


def test_unknown_stream_setting_is_rejected_like_cpp(tmp_path: Path):
    entry = stream(0)
    entry["enabled"] = True
    with pytest.raises(ValueError, match="unknown stream setting: enabled"):
        main.load_app_config(write_config(tmp_path, [entry]))


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


class _Options:
    """Records attribute writes; unset attributes read as the -1 Core default."""

    def __init__(self):
        self.output_caps = SimpleNamespace(fps=-1)

    def __getattr__(self, name):
        return -1


class _Graph:
    def __init__(self, name=""):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def connect(self, *nodes):
        self.nodes.extend(nodes)


def _recording_pyneat():
    return SimpleNamespace(
        RtspDecodedInputOptions=_Options,
        RtspEncodedInputOptions=_Options,
        SimaDecodeOptions=_Options,
        InputOptions=_Options,
        Graph=_Graph,
        RtspCodec=SimpleNamespace(H264="h264", H265="h265"),
        SimaDecodeType=SimpleNamespace(H264="h264", H265="h265"),
        Format=SimpleNamespace(NV12="NV12", H264="H264", H265="H265"),
        CapsMemory=SimpleNamespace(Any="any"),
        PayloadType=SimpleNamespace(Encoded="encoded"),
        InputMemoryPolicy=SimpleNamespace(Ev74="ev74"),
        groups=SimpleNamespace(rtsp_encoded_input=lambda options: ("rtsp", options)),
        nodes=SimpleNamespace(
            input=lambda *args: ("input", args),
            sima_decode=lambda options: ("decode", options),
            caps_raw=lambda *args: ("caps", args),
            output=lambda *args: ("output", args),
        ),
    )


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_source_frame_rate_is_a_decoder_hint_not_a_caps_pin(monkeypatch, codec: str):
    # A 29.97 fps camera probes as 30 but negotiates 30000/1001; pinning 30/1 into
    # the encoded or raw caps stops that stream with an incompatible-caps error.
    monkeypatch.setattr(main, "pyneat", _recording_pyneat())
    stream_cfg = main.StreamConfig("camera0", "rtsp://127.0.0.1/src0", codec, 0)
    cfg = main.AppConfig("detector.tar.gz", "pose.tar.gz", [stream_cfg])
    options = main.build_source_options(cfg, stream_cfg, 1280, 720, 30)
    assert options.dec_fps == 30
    assert options.source_fps == -1
    assert options.output_caps.fps == -1
    assert options.fallback_h264_fps == (30 if codec == "h264" else -1)

    _, encoded = main.make_encoded_source(options).nodes[0]
    assert encoded.source_fps == -1
    assert encoded.fallback_h264_fps == options.fallback_h264_fps

    decoder = main.make_decoder(options).nodes
    _, decode = decoder[1]
    assert decode.dec_fps == 30
    caps = [args for kind, args in decoder if kind == "caps"]
    assert caps == [("NV12", 1280, 720, -1, "any")]


def test_standalone_sequence_dash_starts_a_stream_like_cpp(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n  detector_path: detector.tar.gz\n  pose_path: pose.tar.gz\n"
        "streams:\n"
        "  -\n    id: camera0\n    url: rtsp://127.0.0.1/src0\n"
        "    codec: h264\n    insight_channel: 0\n"
        "  - # second stream\n    id: camera1\n    url: rtsp://127.0.0.1/src1\n"
        "    insight_channel: 1\n"
        "output:\n  insight:\n    host: 127.0.0.1\n",
        encoding="utf-8",
    )
    cfg = main.load_app_config(path)
    assert [stream.id for stream in cfg.streams] == ["camera0", "camera1"]
    assert [stream.insight_channel for stream in cfg.streams] == [0, 1]


def test_null_codec_and_trailing_quotes_match_cpp(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "models:\n  detector_path: detector.tar.gz\n  pose_path: pose.tar.gz\n"
        "streams:\n  - id: camera'\n    url: rtsp://127.0.0.1/src0\n"
        "    codec: null\n    insight_channel: 0\n"
        "output:\n  insight:\n    host: 127.0.0.1\n",
        encoding="utf-8",
    )
    cfg = main.load_app_config(path)
    assert cfg.streams[0].id == "camera'"
    assert cfg.streams[0].codec == main.parse_codec("h264")


@pytest.mark.parametrize("roi_scale", [".nan", ".inf", "-.inf", "0", "-1.5"])
def test_roi_scale_must_be_finite_and_positive(tmp_path: Path, roi_scale: str):
    path = write_config(tmp_path, [stream(0)])
    path.write_text(
        path.read_text(encoding="utf-8") + f"pose:\n  roi_scale: {roi_scale}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pose.roi_scale must be finite and > 0"):
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
    pose = main.decode_pose(raw, raw_world, affine, box, 0.93, 2)
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
    assert data["poses"][0]["id"] == "pose_3"
    assert data["poses"][0]["presence"] == pytest.approx(0.93)
    assert len(data["poses"][0]["keypoints"]) == 33
    assert data["poses"][0]["keypoints"][0]["name"] == "nose"
    assert len(data["poses"][0]["world_keypoints"]) == 33
    assert data["poses"][0]["world_keypoints"][0]["name"] == "nose"
    assert data["poses"][0]["world_keypoints"][0]["z"] == pytest.approx(0.3)
    auxiliary = main.world_pose_auxiliary_data([pose], "camera0")
    assert auxiliary["stream_id"] == "camera0"
    assert auxiliary["schema_version"] == 1
    assert auxiliary["id"] == "world-pose"
    assert auxiliary["renderer"] == "blazepose-3d"
    assert auxiliary["payload"]["poses"][0]["presence"] == pytest.approx(0.93)
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


def test_pose_presence_logit_is_activated_before_thresholding(monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def to_numpy(self, *, copy):
            return self.values.copy() if copy else self.values

    tensors = [Tensor(np.zeros(195)), Tensor([0.0]), Tensor(np.zeros(117))]
    monkeypatch.setattr(main, "tensors_from_sample", lambda *_args: tensors)
    identity = main.FrameIdentity("camera0", 1, 0, -1, -1, 1, 1)
    context = main.PoseInputContext(
        1,
        0,
        0,
        1,
        {
            "x1": 0.0,
            "y1": 0.0,
            "x2": 100.0,
            "y2": 100.0,
            "score": 0.9,
            "class_id": 0,
        },
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        identity,
    )
    cfg = SimpleNamespace(pose_presence_threshold=0.5)

    pose = main.parse_pose_output(object(), context, cfg)
    assert pose["presence"] == pytest.approx(0.5)

    tensors[1] = Tensor([-0.01])
    assert main.parse_pose_output(object(), context, cfg) is None


@pytest.mark.parametrize(
    ("tensor_index", "position", "value"),
    [(0, 0, np.nan), (0, 194, np.inf), (2, 0, np.nan), (2, 116, -np.inf)],
    ids=["screen-nan", "screen-inf", "world-nan", "world-negative-inf"],
)
def test_non_finite_landmarks_discard_only_that_pose(
    monkeypatch, tensor_index: int, position: int, value: float
):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def to_numpy(self, *, copy):
            return self.values.copy() if copy else self.values

    screen = np.zeros(195, dtype=np.float32)
    world = np.zeros(117, dtype=np.float32)
    tensors = [Tensor(screen), Tensor([4.0]), Tensor(world)]
    monkeypatch.setattr(main, "tensors_from_sample", lambda *_args: tensors)
    identity = main.FrameIdentity("camera0", 1, 0, -1, -1, 1, 1)
    box = {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0, "score": 0.9, "class_id": 0}
    context = main.PoseInputContext(
        1, 0, 0, 1, box, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), identity
    )
    cfg = SimpleNamespace(pose_presence_threshold=0.5)
    assert main.parse_pose_output(object(), context, cfg) is not None

    tensors[tensor_index].values[position] = value

    assert main.parse_pose_output(object(), context, cfg) is None


def test_pose_preprocess_copies_readonly_tensor_exports():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image.flags.writeable = False
    tensor = SimpleNamespace(to_numpy=lambda *, copy: image)

    exported = main.writable_rgb_view(tensor)

    assert exported.flags.writeable
    assert not np.shares_memory(exported, image)


def test_frame_identity_falls_back_through_source_sequence_fields():
    assert main.select_frame_id(9, 8, 7, 6) == 9
    assert main.select_frame_id(-1, 8, 7, 6) == 8
    assert main.select_frame_id(-1, -1, 7, 6) == 7
    assert main.select_frame_id(-1, -1, -1, 6) == 6


def test_closed_stream_does_not_block_healthy_stream_frame_limit():
    runtime = SimpleNamespace(
        streams=[
            SimpleNamespace(closed=True, metadata_frames=0, outstanding_frames=0),
            SimpleNamespace(closed=False, metadata_frames=8, outstanding_frames=0),
        ]
    )
    assert main.all_streams_done(runtime, 8)


def test_closed_stream_drains_admitted_frames_before_completion():
    runtime = SimpleNamespace(
        streams=[
            SimpleNamespace(closed=True, metadata_frames=0, outstanding_frames=1),
            SimpleNamespace(closed=False, metadata_frames=8, outstanding_frames=0),
        ]
    )
    assert not main.all_streams_done(runtime, 8)
    runtime.streams[0].outstanding_frames = 0
    assert main.all_streams_done(runtime, 8)


def test_frame_admission_reserves_only_the_remaining_per_stream_limit():
    stream_runtime = SimpleNamespace(
        metadata_lock=threading.Lock(), metadata_frames=6, outstanding_frames=1
    )
    assert main.stream_can_admit_frame(stream_runtime, 8)
    stream_runtime.outstanding_frames = 2
    assert not main.stream_can_admit_frame(stream_runtime, 8)
    stream_runtime.metadata_frames = 8
    stream_runtime.outstanding_frames = 0
    assert not main.stream_can_admit_frame(stream_runtime, 8)
    assert main.stream_can_admit_frame(stream_runtime, 0)


def test_model_push_retries_without_using_the_blocking_python_binding():
    class Run:
        def __init__(self):
            self.attempts = 0

        def try_push(self, _name, _samples):
            self.attempts += 1
            return self.attempts == 3

        @staticmethod
        def can_push():
            return True

        @staticmethod
        def push(*_args):
            raise AssertionError("blocking push must not be used")

    runtime = SimpleNamespace(state=main.SharedState(1))
    pending = main.deque()
    context = object()
    run = Run()

    assert (
        main.try_push_with_context(
            runtime, run, "input", object(), pending, context, "closed"
        )
        is main.NonblockingPushResult.ACCEPTED
    )
    assert run.attempts == 3
    assert list(pending) == [context]


def test_pose_dispatch_aborts_expired_rejected_push_and_continues(monkeypatch):
    calls = []

    class Sender:
        def send_metadata(self, *args):
            calls.append(args)
            return True

    class Tensor:
        def clone(self):
            return self

        def cvu(self):
            return self

    stream_runtime = runtime_stream(Sender(), outstanding=2)
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    now = main.time.monotonic()
    first_job = main.FrameJob(1, 1, 0, object(), [{}], identity, now + 1.0)
    jobs = iter(
        [
            first_job,
            main.FrameJob(2, 2, 0, object(), [], identity, now + 1.0),
            None,
        ]
    )

    class Run:
        attempts = 0

        @classmethod
        def try_push(cls, _name, _samples):
            cls.attempts += 1
            first_job.deadline = main.time.monotonic() - 1.0
            state.aggregates[first_job.job_id].deadline = first_job.deadline
            return False

        @staticmethod
        def can_push():
            return True

    state = main.SharedState(1)
    runtime = SimpleNamespace(
        state=state,
        streams=[stream_runtime],
        pose_run=Run(),
        pose_model=object(),
    )
    monkeypatch.setattr(main, "take_next_job", lambda *_args: next(jobs))
    monkeypatch.setattr(main, "writable_rgb_view", lambda _tensor: object())
    monkeypatch.setattr(main, "square_roi", lambda *_args: (0, 0, 1, 1))
    monkeypatch.setattr(main, "affine_from_tensor", lambda _tensor: (1, 0, 0, 0, 1, 0))
    monkeypatch.setattr(main, "pose_input_sample", lambda *_args: object())
    monkeypatch.setattr(
        main,
        "pyneat",
        SimpleNamespace(
            stages=SimpleNamespace(preproc=lambda *_args, **_kwargs: [Tensor()]),
            PreprocessRoi=lambda *_args: object(),
            PixelFormat=SimpleNamespace(RGB="rgb"),
        ),
    )

    main.dispatch_pose_jobs(
        runtime, SimpleNamespace(roi_scale=1.0, max_pending_jobs=1)
    )

    assert state.error is None
    assert not state.pending_pose_outputs
    assert not state.aggregates
    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 2
    assert stream_runtime.outstanding_frames == 0
    assert len(calls) == 4
    assert Run.attempts == 1


def test_rejected_detector_push_accepts_concurrent_expiry_tombstone(monkeypatch):
    class Tensor:
        @staticmethod
        def cvu():
            return object()

    stream_runtime = runtime_stream(SimpleNamespace(send_metadata=lambda *_args: True))
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    job = main.FrameJob(1, 1, 0, Tensor(), [], identity, main.time.monotonic() + 1.0)
    state = main.SharedState(1)
    state.detector_mailboxes[0] = job

    class Run:
        @staticmethod
        def try_push(_name, _samples):
            job.deadline = main.time.monotonic() - 1.0
            main.expire_detector_jobs(runtime)
            return False

    runtime = SimpleNamespace(state=state, streams=[stream_runtime], detector_run=Run())
    monkeypatch.setattr(main, "image_input_sample", lambda *_args: object())
    worker = threading.Thread(
        target=main.dispatch_detector_jobs,
        args=(runtime, SimpleNamespace(max_pending_jobs=1)),
    )

    worker.start()
    for _ in range(100):
        if stream_runtime.outstanding_frames == 0:
            break
        main.time.sleep(0.002)

    assert worker.is_alive()
    assert not state.pending_detector_outputs
    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 0
    assert state.error is None

    with state.condition:
        state.stopping = True
        state.condition.notify_all()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_detector_dispatch_aborts_expired_rejected_push_and_continues(monkeypatch):
    calls = []

    class Tensor:
        @staticmethod
        def cvu():
            return object()

    stream_runtime = runtime_stream(
        SimpleNamespace(send_metadata=lambda *args: calls.append(args) or True), outstanding=2
    )
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    now = main.time.monotonic()
    first_job = main.FrameJob(1, 1, 0, Tensor(), [], identity, now + 1.0)
    jobs = iter(
        [
            first_job,
            main.FrameJob(2, 2, 0, Tensor(), [], identity, now + 1.0),
            None,
        ]
    )

    class Run:
        attempts = 0

        @classmethod
        def try_push(cls, _name, _samples):
            cls.attempts += 1
            first_job.deadline = main.time.monotonic() - 1.0
            return cls.attempts > 1

        @staticmethod
        def can_push():
            return True

    state = main.SharedState(1)
    runtime = SimpleNamespace(state=state, streams=[stream_runtime], detector_run=Run())
    monkeypatch.setattr(main, "take_next_job", lambda *_args: next(jobs))
    monkeypatch.setattr(main, "image_input_sample", lambda *_args: object())

    main.dispatch_detector_jobs(runtime, SimpleNamespace(max_pending_jobs=2))

    assert state.error is None
    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 1
    assert len(state.pending_detector_outputs) == 1
    assert state.pending_detector_outputs[0].job_id == 2
    assert Run.attempts == 2


def test_detector_timeout_completes_without_output_and_discards_late_result(monkeypatch):
    calls = []

    class Sender:
        def send_metadata(self, *args):
            calls.append(args)
            return True

    stream_runtime = runtime_stream(Sender())
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    job = main.FrameJob(1, 1, 0, object(), [], identity, main.time.monotonic() - 1.0)
    state = main.SharedState(1)
    state.pending_detector_outputs.append(job)

    class DelayedRun:
        def __init__(self):
            self.pulls = 0

        def pull(self, _name, _timeout):
            self.pulls += 1
            if self.pulls == 1:
                return None
            with state.condition:
                state.stopping = True
            return object()

        @staticmethod
        def can_pull():
            return True

    runtime = SimpleNamespace(
        state=state,
        streams=[stream_runtime],
        detector_run=DelayedRun(),
    )
    monkeypatch.setattr(
        main,
        "select_people",
        lambda *_args: pytest.fail("late output must not be parsed as a newer frame"),
    )

    main.pull_detector_outputs(runtime, SimpleNamespace())

    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 0
    assert len(calls) == 2
    assert not state.pending_detector_outputs
    assert state.error is None


def test_detector_dispatch_expires_while_tombstones_hold_capacity(monkeypatch):
    class Tensor:
        @staticmethod
        def cvu():
            return object()

    class Run:
        @staticmethod
        def try_push(*_args):
            raise AssertionError("an expired frame must not reach the detector")

    stream_runtime = runtime_stream(SimpleNamespace(send_metadata=lambda *_args: True))
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    state = main.SharedState(1)
    state.pending_detector_outputs.append(None)
    state.detector_mailboxes[0] = main.FrameJob(
        1,
        1,
        0,
        Tensor(),
        [],
        identity,
        main.time.monotonic() + 0.02,
    )
    runtime = SimpleNamespace(state=state, streams=[stream_runtime], detector_run=Run())
    monkeypatch.setattr(main, "image_input_sample", lambda *_args: object())
    cfg = SimpleNamespace(max_pending_jobs=1)
    worker = threading.Thread(target=main.dispatch_detector_jobs, args=(runtime, cfg))

    worker.start()
    for _ in range(100):
        if stream_runtime.outstanding_frames == 0:
            break
        main.time.sleep(0.002)
    with state.condition:
        state.stopping = True
        state.condition.notify_all()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 0
    assert list(state.pending_detector_outputs) == [None]
    assert state.error is None


def test_pose_timeout_retains_correlation_until_late_roi_output(monkeypatch):
    stream_runtime = runtime_stream(SimpleNamespace(send_metadata=lambda *_args: True))
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    context = main.PoseInputContext(1, 0, 0, 1, {}, (1, 0, 0, 0, 1, 0), identity)
    state = main.SharedState(1)
    state.pending_pose_outputs.append(context)
    state.aggregates[1] = main.PoseAggregate(
        0, 1, 1, identity, main.time.monotonic() - 1.0
    )

    class DelayedRun:
        def __init__(self):
            self.pulls = 0

        def pull(self, _name, _timeout):
            self.pulls += 1
            if self.pulls == 1:
                return None
            with state.condition:
                state.stopping = True
            return object()

        @staticmethod
        def can_pull():
            return True

    runtime = SimpleNamespace(state=state, streams=[stream_runtime], pose_run=DelayedRun())
    monkeypatch.setattr(
        main,
        "parse_pose_output",
        lambda *_args: pytest.fail("late ROI output must not be parsed"),
    )

    main.pull_pose_outputs(runtime, SimpleNamespace())

    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 0
    assert stream_runtime.completed_rois == 1
    assert not state.pending_pose_outputs
    assert not state.aggregates
    assert state.error is None


def test_completed_pose_aggregate_is_claimed_before_expiry_can_publish_it(monkeypatch):
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
    context = main.PoseInputContext(1, 0, 0, 1, {}, (1, 0, 0, 0, 1, 0), identity)
    state = main.SharedState(1)
    state.pending_pose_outputs.append(context)
    state.aggregates[1] = main.PoseAggregate(0, 1, 1, identity, main.time.monotonic() + 60.0)
    runtime = SimpleNamespace(state=state, streams=[], pose_run=None)
    expiry_runs = []

    class RacingStream(SimpleNamespace):
        armed = False

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "completed_rois" and RacingStream.armed and not expiry_runs:
                # The final ROI output has been counted under the state lock and the
                # lock released. Before the puller publishes, the frame's deadline
                # passes and the dispatcher runs an expiry pass.
                with state.condition:
                    for aggregate in state.aggregates.values():
                        aggregate.deadline = 0.0
                main.expire_pose_jobs(runtime)
                expiry_runs.append(True)

    stream = RacingStream(**vars(runtime_stream(SimpleNamespace(send_metadata=lambda *_: True))))
    runtime.streams.append(stream)

    class OneOutputRun:
        @staticmethod
        def pull(_name, _timeout):
            with state.condition:
                state.stopping = True
            return object()

        @staticmethod
        def can_pull():
            return True

    runtime.pose_run = OneOutputRun()
    monkeypatch.setattr(main, "parse_pose_output", lambda *_args: None)
    RacingStream.armed = True

    main.pull_pose_outputs(runtime, SimpleNamespace())

    assert expiry_runs == [True]
    assert state.error is None
    assert stream.metadata_frames == 1
    assert stream.outstanding_frames == 0
    assert stream.timed_out_jobs == 0
    assert not state.aggregates


@pytest.mark.parametrize("failure", ["returns-false", "raises"])
def test_failed_metadata_pair_completes_the_frame_without_counting_it(failure: str):
    class Sender:
        def __init__(self):
            self.calls = []
            self.fail = True

        def send_metadata(self, metadata_type, *_args):
            self.calls.append(metadata_type)
            if not self.fail or metadata_type == "pose-estimation":
                return True
            if failure == "raises":
                raise RuntimeError("metadata datagram could not be queued")
            return False

    sender = Sender()
    stream = runtime_stream(sender, outstanding=2)
    identity = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)

    main.complete_frame(stream, 1, identity, [])

    assert sender.calls == ["pose-estimation", "auxiliary-visualization"]
    assert stream.metadata_frames == 0
    assert stream.metadata_send_failures == 1
    assert stream.outstanding_frames == 1

    sender.fail = False
    main.complete_frame(stream, 2, identity, [])

    assert stream.metadata_frames == 1
    assert stream.metadata_send_failures == 1
    assert stream.outstanding_frames == 0
    stream.closed = True
    runtime = SimpleNamespace(streams=[stream])
    assert main.all_streams_done(runtime, 2)
    with pytest.raises(RuntimeError, match="before reaching runtime.frames=2: camera0"):
        main.require_successful_completion(runtime, 2)


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


def test_one_source_owner_detects_each_independent_run_closing():
    owner_threads = set()
    barrier = threading.Barrier(2)

    class ClosedRun:
        def __init__(self):
            self.pull_count = 0

        def pull(self, _output, timeout_ms):
            assert timeout_ms == -1
            owner_threads.add(threading.get_ident())
            self.pull_count += 1
            barrier.wait(timeout=1)
            return None

    runs = [ClosedRun(), ClosedRun()]
    streams = [
        SimpleNamespace(
            config=SimpleNamespace(id=f"camera{index}"),
            source_run=source_run,
            closed=False,
            metadata_lock=threading.Lock(),
            metadata_frames=0,
            outstanding_frames=0,
        )
        for index, source_run in enumerate(runs)
    ]
    runtime = SimpleNamespace(streams=streams, state=main.SharedState(2))
    cfg = SimpleNamespace(frame_limit=0)
    pullers = [
        threading.Thread(target=main.pull_source_frames, args=(runtime, cfg, index))
        for index in range(2)
    ]

    for puller in pullers:
        puller.start()
    for puller in pullers:
        puller.join(timeout=2)

    assert all(not puller.is_alive() for puller in pullers)
    assert all(stream.closed for stream in streams)
    assert [source_run.pull_count for source_run in runs] == [1, 1]
    assert len(owner_threads) == 2


def test_source_runs_start_concurrently(monkeypatch):
    build_threads = set()
    barrier = threading.Barrier(2)

    class ClosedRun:
        def pull(self, _output, timeout_ms):
            assert timeout_ms == -1
            return None

        @staticmethod
        def close():
            return None

    class SourceGraph:
        def build(self, _options):
            build_threads.add(threading.get_ident())
            barrier.wait(timeout=1)
            return ClosedRun()

    fake_pyneat = SimpleNamespace(
        RunOptions=type("RunOptions", (), {}),
        RunPreset=SimpleNamespace(Realtime="realtime"),
        OutputMemory=SimpleNamespace(ZeroCopy="zero-copy"),
    )
    monkeypatch.setattr(main, "pyneat", fake_pyneat)
    streams = [
        SimpleNamespace(
            index=index,
            config=SimpleNamespace(id=f"camera{index}"),
            source_graph=SourceGraph(),
            source_run=None,
            closed=False,
        )
        for index in range(2)
    ]
    runtime = SimpleNamespace(streams=streams, state=main.SharedState(2))
    pullers = [
        threading.Thread(
            target=main.run_source_stream,
            args=(runtime, SimpleNamespace(), index),
        )
        for index in range(2)
    ]

    for puller in pullers:
        puller.start()
    for puller in pullers:
        puller.join(timeout=2)

    assert all(not puller.is_alive() for puller in pullers)
    assert len(build_threads) == 2
    assert all(stream.closed for stream in streams)


def test_publish_metadata_sends_paired_overlay_and_auxiliary_messages():
    calls = []

    class Sender:
        def send_metadata(self, *args):
            calls.append(args)
            return True

    stream_runtime = runtime_stream(Sender(), temporal_filter=True)
    identity = main.FrameIdentity("camera0", 7, 1_234_000_000, -1, -1, 7, 7)

    main.complete_frame(stream_runtime, 1, identity, [])

    assert [call[0] for call in calls] == [
        "pose-estimation",
        "auxiliary-visualization",
    ]
    assert all(call[2:] == (1234, "7") for call in calls)
    assert json.loads(calls[0][1]) == {"stream_id": "camera0", "poses": []}
    assert json.loads(calls[1][1])["stream_id"] == "camera0"
    assert json.loads(calls[1][1])["payload"] == {"poses": []}
    assert stream_runtime.metadata_frames == 1


def test_frame_publication_waits_for_prior_sequence_and_skips_dropped_work():
    calls = []

    class Sender:
        def send_metadata(self, _type, _data, _timestamp, frame_id):
            calls.append(frame_id)
            return True

    stream_runtime = runtime_stream(Sender(), outstanding=3)
    first = main.FrameIdentity("camera0", 10, 10_000_000, -1, -1, 10, 10)
    second = main.FrameIdentity("camera0", 11, 11_000_000, -1, -1, 11, 11)
    third = main.FrameIdentity("camera0", 12, 12_000_000, -1, -1, 12, 12)

    main.complete_frame(stream_runtime, 2, second, [])
    assert calls == []
    main.complete_frame(stream_runtime, 1, first, [])
    main.complete_frame(stream_runtime, 3, third, None)

    assert calls == ["10", "10", "11", "11"]
    assert stream_runtime.metadata_frames == 2
    assert stream_runtime.next_publication_sequence == 4


def test_shutdown_does_not_wait_for_an_uninterruptible_source_build():
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    runtime = SimpleNamespace(
        streams=[SimpleNamespace(config=SimpleNamespace(id="offline"))]
    )

    started = main.time.monotonic()
    unfinished = main.join_source_workers(runtime, [worker], timeout_s=0.01)

    assert unfinished == ["offline"]
    assert main.time.monotonic() - started < 0.25
    release.set()
    worker.join(timeout=1)


def pose_sample(x: float, confidence: float, world_x: float, box_x: float = 0.0) -> dict:
    return {
        "roi_index": 0,
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


def test_pose_smoother_drops_stale_subject_after_coasting_without_pts():
    smoother = main.PoseSmoother()
    smoother.filter([pose_sample(50.0, 0.9, 0.0)], -1)
    assert smoother.filter([], -1) and smoother.filter([], -1)
    assert smoother.filter([], -1) == []

    later = smoother.filter([pose_sample(54.0, 0.4, 0.0)], -1)

    assert later[0]["keypoints"][0]["x"] == 54.0
