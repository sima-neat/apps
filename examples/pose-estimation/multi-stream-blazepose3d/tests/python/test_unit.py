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

IDENTITY = main.FrameIdentity("camera0", 1, 1_000_000, -1, -1, 1, 1)
BOX = {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0, "score": 0.9, "class_id": 0}
UNIT_AFFINE = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
CONFIG_HEAD = "models:\n  detector_path: detector.tar.gz\n  pose_path: pose.tar.gz\n"
CONFIG_TAIL = "output:\n  insight:\n    host: 127.0.0.1\n"


def stream(index: int, *, channel: int | None = None, **fields) -> dict:
    return {
        "id": f"camera{index}",
        "url": f"rtsp://127.0.0.1/src{index}",
        "codec": "hevc" if index == 1 else "h264",
        "insight_channel": index if channel is None else channel,
        **fields,
    }


def config_text(streams=None, insight=None, output=None, **sections) -> str:
    config = {
        "models": {"detector_path": "detector.tar.gz", "pose_path": "pose.tar.gz"},
        "streams": [stream(0)] if streams is None else streams,
        **sections,
        "output": {"insight": {"host": "127.0.0.1", **(insight or {})}, **(output or {})},
    }
    return yaml.safe_dump(config, sort_keys=False)


BASE = config_text()


def replaced(old: str, new: str) -> str:
    return BASE.replace(old, new, 1)


def load_config(tmp_path: Path, text: str) -> main.AppConfig:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return main.load_app_config(path)


def stream_config(index: int, codec: str = "h264", channel: int | None = None):
    return main.StreamConfig(
        f"camera{index}",
        f"rtsp://127.0.0.1/src{index}",
        codec,
        index if channel is None else channel,
    )


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


class RecordingSender:
    def __init__(self):
        self.calls = []

    def send_metadata(self, *args):
        self.calls.append(args)
        return True


class FakeFrame:
    def clone(self):
        return self

    def cvu(self):
        return self


class LateOutputRun:
    """Times out `timeouts` times, then returns one output and stops the app."""

    def __init__(self, state, timeouts: int = 1):
        self.state = state
        self.timeouts = timeouts

    def pull(self, _name, _timeout):
        if self.timeouts > 0:
            self.timeouts -= 1
            return None
        with self.state.condition:
            self.state.stopping = True
        return object()

    @staticmethod
    def can_pull():
        return True


def wait_until(predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        main.time.sleep(0.002)


def stop_and_join(state, worker: threading.Thread) -> None:
    with state.condition:
        state.stopping = True
        state.condition.notify_all()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_cli_help_and_missing_config():
    help_result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"], capture_output=True, text=True, check=False
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


FLOW_ERROR = "flow-style YAML collections are not supported"
FLOW_STREAM = "{id: camera0, url: rtsp://127.0.0.1/src0, insight_channel: 0}"
BLOCK_STREAM = "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    insight_channel: 0\n"

# test_config_validation() in test_unit.cpp runs the same cases against the C++ app.
REJECTED_CONFIGS = [
    ("five-streams", config_text([stream(i) for i in range(5)]), "between 1 and 4 entries"),
    ("duplicate-id", config_text([stream(0), stream(1, id="camera0")]), "ids must be unique"),
    (
        "duplicate-channel",
        config_text([stream(0), stream(1, channel=0)]),
        "channels must be unique",
    ),
    (
        "port-overlap",
        config_text([stream(0), stream(1, channel=100)], {"video_port_base": 9000}),
        "video and metadata ports must not overlap",
    ),
    (
        "video-port-range",
        config_text([stream(0, channel=60000)], {"metadata_port_base": 1}, {"video_enabled": True}),
        "stream video port must be <= 65535",
    ),
    (
        "metadata-port-range",
        config_text([stream(0, channel=60000)], {"video_port_base": 1}, {"video_enabled": False}),
        "stream metadata port must be <= 65535",
    ),
    ("unknown-stream-key", config_text([stream(0, enabled=True)]), "unknown stream setting"),
    ("partial-caps", config_text([stream(0, width=1920)]), "all be omitted or all be > 0"),
    ("unknown-codec", config_text([stream(0, codec="vp9")]), "codec must be h264/avc or h265/hevc"),
    ("eleven-people", config_text(pose={"max_people_per_frame": 11}), "between 1 and 10"),
    ("max-detections", config_text(detector={"max_detections": 0}), "max_detections must be > 0"),
    ("max-inflight", config_text(detector={"max_inflight_per_stream": 0}), "must be -1 or > 0"),
    ("max-pending-jobs", config_text(pose={"max_pending_jobs": 0}), "max_pending_jobs must be > 0"),
    ("temporal-filter", config_text(pose={"temporal_filter_enabled": 1}), "must be true or false"),
    ("null-bool", config_text(output={"video_enabled": None}), "must be true or false"),
    ("null-int", config_text(detector={"max_detections": None}), "must be an integer"),
    ("null-double", config_text(pose={"roi_scale": None}), "must be numeric"),
    ("null-url", replaced("url: rtsp://127.0.0.1/src0", "url: null"), "url must be set"),
    ("numeric-id", replaced("id: camera0", "id: 17"), "must be a string"),
    ("binary-id", replaced("id: camera0", "id: 0b101"), "must be a string"),
    ("sexagesimal-id", replaced("id: camera0", "id: 1:20"), "must be a string"),
    ("date-id", replaced("id: camera0", "id: 2026-10-01"), "must be a string"),
    ("boolean-url", replaced("url: rtsp://127.0.0.1/src0", "url: true"), "must be a string"),
    ("numeric-detector", replaced("detector.tar.gz", "123"), "must be a string"),
    ("boolean-pose", replaced("pose.tar.gz", "false"), "must be a string"),
    ("numeric-host", replaced("host: 127.0.0.1", "host: 127"), "must be a string"),
    ("flow-streams", CONFIG_HEAD + f"streams: [{FLOW_STREAM}]\n", FLOW_ERROR),
    ("flow-stream-entry", CONFIG_HEAD + f"streams:\n  - {FLOW_STREAM}\n", FLOW_ERROR),
    (
        "flow-section",
        CONFIG_HEAD + "streams:\n" + BLOCK_STREAM + "output: {insight: {host: 127.0.0.1}}\n",
        FLOW_ERROR,
    ),
] + [
    (f"roi-scale-{value}", BASE + f"pose:\n  roi_scale: {value}\n", "finite and > 0")
    for value in (".nan", ".inf", "-.inf", "0", "-1.5")
]


@pytest.mark.parametrize(
    ("text", "message"),
    [case[1:] for case in REJECTED_CONFIGS],
    ids=[case[0] for case in REJECTED_CONFIGS],
)
def test_config_validation(tmp_path: Path, text: str, message: str):
    with pytest.raises((TypeError, ValueError), match=message):
        load_config(tmp_path, text)


ACCEPTED_CONFIGS = [
    (
        "standalone-sequence-dash",
        CONFIG_HEAD
        + "streams:\n  -\n    id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n"
        + "    insight_channel: 0\n  - # second stream\n    id: camera1\n"
        + "    url: rtsp://127.0.0.1/src1\n    insight_channel: 1\n"
        + CONFIG_TAIL,
        [stream_config(0), stream_config(1)],
    ),
    (
        "yaml-spellings",
        replaced("url: rtsp://127.0.0.1/src0", "url: rtsp://host/cam's # camera label")
        .replace("insight_channel: 0", "insight_channel: 0b10")
        + "runtime:\n  frames: 010\ndetector:\n  min_score: 0.5_0\n"
        + "pose:\n  job_timeout_ms: 1_000\n  roi_scale: 1:20.5\ntest:\n  nan: .NaN\n",
        [main.StreamConfig("camera0", "rtsp://host/cam's", "h264", 2)],
    ),
    ("empty-flow-mapping", BASE + "pose: {}\n", [stream_config(0)]),
    (
        "null-codec-trailing-quote",
        replaced("id: camera0", "id: camera'").replace("codec: h264", "codec: null"),
        [main.StreamConfig("camera'", "rtsp://127.0.0.1/src0", "h264", 0)],
    ),
    (
        "mapping-order-and-caps",
        config_text(
            [
                {
                    "url": "rtsp://127.0.0.1/ordered",
                    "width": 1920,
                    "id": "camera0",
                    "fps": 30,
                    "insight_channel": 0,
                    "height": 1080,
                    "codec": "h264",
                }
            ]
        ),
        [main.StreamConfig("camera0", "rtsp://127.0.0.1/ordered", "h264", 0, 1920, 1080, 30)],
    ),
    (
        "video-disabled-skips-video-port-range",
        config_text(
            [stream(0, channel=60000)], {"metadata_port_base": 1}, {"video_enabled": False}
        ),
        [stream_config(0, channel=60000)],
    ),
    (
        "codec-spellings",
        config_text([stream(i, codec=c) for i, c in enumerate(("avc", "H.264", "HEVC", "h.265"))]),
        [stream_config(0), stream_config(1), stream_config(2, "h265"), stream_config(3, "h265")],
    ),
]


@pytest.mark.parametrize(
    ("text", "streams"),
    [case[1:] for case in ACCEPTED_CONFIGS],
    ids=[case[0] for case in ACCEPTED_CONFIGS],
)
def test_accepted_configs(tmp_path: Path, text: str, streams: list):
    assert load_config(tmp_path, text).streams == streams


def test_restored_settings_are_read_with_their_defaults(tmp_path: Path):
    def restored(cfg):
        return (
            cfg.max_detections,
            cfg.max_inflight_per_stream,
            cfg.max_pending_jobs,
            cfg.pose_temporal_filter_enabled,
            cfg.video_enabled,
        )

    assert restored(load_config(tmp_path, BASE)) == (100, 4, 64, True, True)
    assert restored(main.load_app_config(main.DEFAULT_CONFIG)) == (100, 4, 64, True, True)
    custom = config_text(
        detector={"max_detections": 7, "max_inflight_per_stream": -1},
        pose={"max_pending_jobs": 3, "temporal_filter_enabled": False},
        output={"video_enabled": False},
    )
    assert restored(load_config(tmp_path, custom)) == (7, -1, 3, False, False)


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


def test_roi_landmark_and_metadata_contract():
    box = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0, "score": 0.9, "class_id": 0}
    assert main.square_roi(box, 1.5) == (-10, 10, 60, 60)
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[0] = [4.0, 8.0, 0.0, 2.0, -2.0]
    raw_world = np.zeros((39, 3), dtype=np.float32)
    raw_world[0] = [0.1, -0.2, 0.3]
    pose = main.decode_pose(raw, raw_world, (2.0, 0.0, 10.0, 0.0, 3.0, 20.0), box, 0.93, 2)
    nose, world_nose = pose["keypoints"][0], pose["world_keypoints"][0]
    assert pose["presence"] == pytest.approx(0.93)
    assert (nose["x"], nose["y"]) == pytest.approx((18.0, 44.0))
    assert nose["confidence"] == world_nose["confidence"] == pytest.approx(main.sigmoid(-2.0))
    assert (world_nose["x"], world_nose["y"], world_nose["z"]) == pytest.approx((0.1, -0.2, 0.3))

    data = main.poses_data([pose], "camera0")
    published = data["poses"][0]
    assert data["stream_id"] == "camera0"
    assert (published["id"], published["presence"]) == ("pose_3", pytest.approx(0.93))
    assert len(published["keypoints"]) == len(published["world_keypoints"]) == 33
    assert published["keypoints"][0]["name"] == published["world_keypoints"][0]["name"] == "nose"
    assert published["world_keypoints"][0]["z"] == pytest.approx(0.3)

    auxiliary = main.world_pose_auxiliary_data([pose], "camera0")
    assert {key: auxiliary[key] for key in ("schema_version", "id", "renderer", "stream_id")} == {
        "schema_version": 1,
        "id": "world-pose",
        "renderer": "blazepose-3d",
        "stream_id": "camera0",
    }
    assert auxiliary["payload"]["poses"] == [
        {"id": "pose_3", "presence": 0.93, "keypoints": published["world_keypoints"]}
    ]
    point_cloud = {"points": [{"x": 0.1, "y": 0.2, "z": 0.3, "value": 7}], "axes": ["east"]}
    generic = main.auxiliary_visualization_data("depth-cloud", "point-cloud-3d", point_cloud)
    assert generic == {
        "schema_version": 1,
        "id": "depth-cloud",
        "renderer": "point-cloud-3d",
        "payload": point_cloud,
    }
    json.dumps([data, auxiliary, generic])


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def to_numpy(self, *, copy):
        return self.values.copy() if copy else self.values


def parse_pose(monkeypatch, presence_logit: float, poison=None):
    tensors = [FakeTensor(np.zeros(195)), FakeTensor([presence_logit]), FakeTensor(np.zeros(117))]
    if poison is not None:
        tensor_index, position, value = poison
        tensors[tensor_index].values[position] = value
    monkeypatch.setattr(main, "tensors_from_sample", lambda *_args: tensors)
    context = main.PoseInputContext(1, 0, 0, BOX, UNIT_AFFINE, IDENTITY)
    return main.parse_pose_output(object(), context, SimpleNamespace(pose_presence_threshold=0.5))


def test_pose_presence_logit_is_activated_before_thresholding(monkeypatch):
    assert parse_pose(monkeypatch, 0.0)["presence"] == pytest.approx(0.5)
    assert parse_pose(monkeypatch, -0.01) is None


@pytest.mark.parametrize(
    "poison",
    [(0, 0, np.nan), (0, 194, np.inf), (2, 0, np.nan), (2, 116, -np.inf)],
    ids=["screen-nan", "screen-inf", "world-nan", "world-negative-inf"],
)
def test_non_finite_landmarks_discard_only_that_pose(monkeypatch, poison):
    assert parse_pose(monkeypatch, 4.0) is not None
    assert parse_pose(monkeypatch, 4.0, poison) is None


def test_pose_preprocess_copies_readonly_tensor_exports():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image.flags.writeable = False
    exported = main.writable_rgb_view(SimpleNamespace(to_numpy=lambda *, copy: image))
    assert exported.flags.writeable
    assert not np.shares_memory(exported, image)


def test_frame_identity_falls_back_through_source_sequence_fields():
    assert main.select_frame_id(9, 8, 7, 6) == 9
    assert main.select_frame_id(-1, 8, 7, 6) == 8
    assert main.select_frame_id(-1, -1, 7, 6) == 7
    assert main.select_frame_id(-1, -1, -1, 6) == 6


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
        attempts = 0

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
    result = main.try_push_with_context(runtime, run, "input", object(), pending, context, "closed")
    assert result is main.NonblockingPushResult.ACCEPTED
    assert run.attempts == 3
    assert list(pending) == [context]


def test_pose_dispatch_aborts_expired_rejected_push_and_continues(monkeypatch):
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender, outstanding=2)
    now = main.time.monotonic()
    first_job = main.FrameJob(1, 1, 0, object(), [{}], IDENTITY, now + 1.0)
    jobs = iter([first_job, main.FrameJob(2, 2, 0, object(), [], IDENTITY, now + 1.0), None])

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
        state=state, streams=[stream_runtime], pose_run=Run(), pose_model=object()
    )
    monkeypatch.setattr(main, "take_next_job", lambda *_args: next(jobs))
    monkeypatch.setattr(main, "writable_rgb_view", lambda _tensor: object())
    monkeypatch.setattr(main, "square_roi", lambda *_args: (0, 0, 1, 1))
    monkeypatch.setattr(main, "affine_from_tensor", lambda _tensor: UNIT_AFFINE)
    monkeypatch.setattr(main, "pose_input_sample", lambda *_args: object())
    monkeypatch.setattr(
        main,
        "pyneat",
        SimpleNamespace(
            stages=SimpleNamespace(preproc=lambda *_args, **_kwargs: [FakeFrame()]),
            PreprocessRoi=lambda *_args: object(),
            PixelFormat=SimpleNamespace(RGB="rgb"),
        ),
    )

    main.dispatch_pose_jobs(runtime, SimpleNamespace(roi_scale=1.0, max_pending_jobs=1))

    assert state.error is None
    assert not state.pending_pose_outputs
    assert not state.aggregates
    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 2
    assert stream_runtime.outstanding_frames == 0
    assert len(sender.calls) == 4
    assert Run.attempts == 1


def test_rejected_detector_push_accepts_concurrent_expiry_tombstone(monkeypatch):
    stream_runtime = runtime_stream(RecordingSender())
    job = main.FrameJob(1, 1, 0, FakeFrame(), [], IDENTITY, main.time.monotonic() + 1.0)
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
        target=main.dispatch_detector_jobs, args=(runtime, SimpleNamespace(max_pending_jobs=1))
    )

    worker.start()
    wait_until(lambda: stream_runtime.outstanding_frames == 0)

    assert worker.is_alive()
    assert not state.pending_detector_outputs
    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 0
    assert state.error is None
    stop_and_join(state, worker)


def test_detector_dispatch_aborts_expired_rejected_push_and_continues(monkeypatch):
    stream_runtime = runtime_stream(RecordingSender(), outstanding=2)
    now = main.time.monotonic()
    first_job = main.FrameJob(1, 1, 0, FakeFrame(), [], IDENTITY, now + 1.0)
    jobs = iter([first_job, main.FrameJob(2, 2, 0, FakeFrame(), [], IDENTITY, now + 1.0), None])

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
    assert [job.job_id for job in state.pending_detector_outputs] == [2]
    assert Run.attempts == 2


def test_detector_timeout_completes_without_output_and_discards_late_result(monkeypatch):
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender)
    state = main.SharedState(1)
    state.pending_detector_outputs.append(
        main.FrameJob(1, 1, 0, object(), [], IDENTITY, main.time.monotonic() - 1.0)
    )
    runtime = SimpleNamespace(
        state=state, streams=[stream_runtime], detector_run=LateOutputRun(state)
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
    assert len(sender.calls) == 2
    assert not state.pending_detector_outputs
    assert state.error is None


def test_detector_dispatch_expires_while_tombstones_hold_capacity(monkeypatch):
    class Run:
        @staticmethod
        def try_push(*_args):
            raise AssertionError("an expired frame must not reach the detector")

    stream_runtime = runtime_stream(RecordingSender())
    state = main.SharedState(1)
    state.pending_detector_outputs.append(None)
    state.detector_mailboxes[0] = main.FrameJob(
        1, 1, 0, FakeFrame(), [], IDENTITY, main.time.monotonic() + 0.02
    )
    runtime = SimpleNamespace(state=state, streams=[stream_runtime], detector_run=Run())
    monkeypatch.setattr(main, "image_input_sample", lambda *_args: object())
    worker = threading.Thread(
        target=main.dispatch_detector_jobs, args=(runtime, SimpleNamespace(max_pending_jobs=1))
    )

    worker.start()
    wait_until(lambda: stream_runtime.outstanding_frames == 0)
    stop_and_join(state, worker)

    assert stream_runtime.timed_out_jobs == 1
    assert stream_runtime.metadata_frames == 1
    assert stream_runtime.outstanding_frames == 0
    assert list(state.pending_detector_outputs) == [None]
    assert state.error is None


def pose_job_state(deadline: float):
    state = main.SharedState(1)
    state.pending_pose_outputs.append(main.PoseInputContext(1, 0, 0, {}, UNIT_AFFINE, IDENTITY))
    state.aggregates[1] = main.PoseAggregate(0, 1, 1, IDENTITY, deadline)
    return state


def test_pose_timeout_retains_correlation_until_late_roi_output(monkeypatch):
    stream_runtime = runtime_stream(RecordingSender())
    state = pose_job_state(main.time.monotonic() - 1.0)
    runtime = SimpleNamespace(state=state, streams=[stream_runtime], pose_run=LateOutputRun(state))
    monkeypatch.setattr(
        main, "parse_pose_output", lambda *_args: pytest.fail("late ROI output must not be parsed")
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
    state = pose_job_state(main.time.monotonic() + 60.0)
    runtime = SimpleNamespace(state=state, streams=[], pose_run=LateOutputRun(state, timeouts=0))
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

    stream = RacingStream(**vars(runtime_stream(RecordingSender())))
    runtime.streams.append(stream)
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

    main.complete_frame(stream, 1, IDENTITY, [])

    assert sender.calls == ["pose-estimation", "auxiliary-visualization"]
    assert stream.metadata_frames == 0
    assert stream.metadata_send_failures == 1
    assert stream.outstanding_frames == 1

    sender.fail = False
    main.complete_frame(stream, 2, IDENTITY, [])

    assert stream.metadata_frames == 1
    assert stream.metadata_send_failures == 1
    assert stream.outstanding_frames == 0
    stream.closed = True
    runtime = SimpleNamespace(streams=[stream])
    assert main.all_streams_done(runtime, 2)
    with pytest.raises(RuntimeError, match="before reaching runtime.frames=2: camera0"):
        main.require_successful_completion(runtime, 2)


def test_all_closed_streams_fail_an_unbounded_run():
    stream = SimpleNamespace(config=SimpleNamespace(id="camera0"), closed=True, metadata_frames=2)
    with pytest.raises(RuntimeError, match="all source streams stopped"):
        main.require_successful_completion(SimpleNamespace(streams=[stream]), 0)


def test_each_source_run_starts_and_is_pulled_on_its_own_thread(monkeypatch):
    build_threads, pull_threads = set(), set()
    build_barrier, pull_barrier = threading.Barrier(2), threading.Barrier(2)

    class ClosedRun:
        pull_count = 0

        def pull(self, _output, timeout_ms):
            assert timeout_ms == -1
            pull_threads.add(threading.get_ident())
            self.pull_count += 1
            pull_barrier.wait(timeout=1)
            return None

    class SourceGraph:
        def __init__(self):
            self.run = ClosedRun()

        def build(self, _options):
            build_threads.add(threading.get_ident())
            build_barrier.wait(timeout=1)
            return self.run

    monkeypatch.setattr(
        main,
        "pyneat",
        SimpleNamespace(
            RunOptions=type("RunOptions", (), {}),
            RunPreset=SimpleNamespace(Realtime="realtime"),
            OutputMemory=SimpleNamespace(ZeroCopy="zero-copy"),
        ),
    )
    streams = [
        SimpleNamespace(
            index=index,
            config=SimpleNamespace(id=f"camera{index}"),
            source_graph=SourceGraph(),
            source_run=None,
            closed=False,
            metadata_lock=threading.Lock(),
            metadata_frames=0,
            outstanding_frames=0,
        )
        for index in range(2)
    ]
    runtime = SimpleNamespace(streams=streams, state=main.SharedState(2))
    cfg = SimpleNamespace(frame_limit=0)
    pullers = [
        threading.Thread(target=main.run_source_stream, args=(runtime, cfg, index))
        for index in range(2)
    ]

    for puller in pullers:
        puller.start()
    for puller in pullers:
        puller.join(timeout=2)

    assert all(not puller.is_alive() for puller in pullers)
    assert len(build_threads) == len(pull_threads) == 2
    assert [stream.source_graph.run.pull_count for stream in streams] == [1, 1]
    assert all(stream.closed for stream in streams)


def test_publish_metadata_sends_paired_overlay_and_auxiliary_messages():
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender, temporal_filter=True)
    identity = main.FrameIdentity("camera0", 7, 1_234_000_000, -1, -1, 7, 7)

    main.complete_frame(stream_runtime, 1, identity, [])

    calls = sender.calls
    assert [call[0] for call in calls] == ["pose-estimation", "auxiliary-visualization"]
    assert all(call[2:] == (1234, "7") for call in calls)
    assert json.loads(calls[0][1]) == {"stream_id": "camera0", "poses": []}
    assert json.loads(calls[1][1])["stream_id"] == "camera0"
    assert json.loads(calls[1][1])["payload"] == {"poses": []}
    assert stream_runtime.metadata_frames == 1


def test_frame_publication_waits_for_prior_sequence_and_skips_dropped_work():
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender, outstanding=3)

    def identity(frame_id: int):
        return main.FrameIdentity("camera0", frame_id, frame_id * 1_000_000, -1, -1, 0, 0)

    main.complete_frame(stream_runtime, 2, identity(11), [])
    assert sender.calls == []
    main.complete_frame(stream_runtime, 1, identity(10), [])
    main.complete_frame(stream_runtime, 3, identity(12), None)

    assert [call[3] for call in sender.calls] == ["10", "10", "11", "11"]
    assert stream_runtime.metadata_frames == 2
    assert stream_runtime.next_publication_sequence == 4


def test_shutdown_does_not_wait_for_an_uninterruptible_source_build():
    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()
    runtime = SimpleNamespace(streams=[SimpleNamespace(config=SimpleNamespace(id="offline"))])

    started = main.time.monotonic()
    unfinished = main.join_source_workers(runtime, [worker], timeout_s=0.01)

    assert unfinished == ["offline"]
    assert main.time.monotonic() - started < 0.25
    release.set()
    worker.join(timeout=1)


def pose_sample(x: float, confidence: float, world_x: float, roi_index: int = 0) -> dict:
    return {
        "roi_index": roi_index,
        "presence": confidence,
        "box": dict(BOX),
        "keypoints": [{"name": "nose", "x": x, "y": 50.0, "confidence": confidence}],
        "world_keypoints": [
            {"name": "nose", "x": world_x, "y": 0.0, "z": 0.0, "confidence": confidence}
        ],
    }


def test_pose_smoother_filters_2d_world_and_confidence_together_without_buffering():
    smoother = main.PoseSmoother()
    first = smoother.filter([pose_sample(50.0, 0.2, 0.0)], 1_000_000_000)[0]
    assert first["keypoints"][0]["x"] == 50.0

    second = smoother.filter([pose_sample(54.0, 0.4, 0.04, roi_index=1)], 1_040_000_000)[0]
    image_fraction = (second["keypoints"][0]["x"] - 50.0) / 4.0
    world_fraction = second["world_keypoints"][0]["x"] / 0.04
    assert 0.45 < image_fraction < 0.90
    assert world_fraction == pytest.approx(image_fraction)
    assert second["keypoints"][0]["confidence"] == pytest.approx(0.24)
    assert second["world_keypoints"][0]["confidence"] == pytest.approx(0.24)
    assert second["roi_index"] == 1  # A matched pose keeps its current detector rank as its id.

    fast = smoother.filter([pose_sample(154.0, 0.4, 1.04)], 1_080_000_000)[0]
    assert fast["keypoints"][0]["x"] > 140.0

    raw_after_gap = pose_sample(30.0, 0.9, -0.2)
    reset = smoother.filter([copy.deepcopy(raw_after_gap)], 1_400_000_000)[0]
    assert reset == raw_after_gap


def test_pose_smoother_weights_motion_by_elapsed_pts():
    def moved_fraction(elapsed_ns: int) -> float:
        smoother = main.PoseSmoother()
        smoother.filter([pose_sample(50.0, 0.9, 0.0)], 1_000_000_000)
        moved = smoother.filter([pose_sample(52.0, 0.9, 0.0)], 1_000_000_000 + elapsed_ns)
        return (moved[0]["keypoints"][0]["x"] - 50.0) / 2.0

    assert moved_fraction(40_000_000) < moved_fraction(120_000_000) < 1.0


def test_pose_smoother_state_is_independent_per_stream():
    left, right = main.PoseSmoother(), main.PoseSmoother()
    left.filter([pose_sample(10.0, 1.0, 0.0)], 1_000_000_000)
    right.filter([pose_sample(90.0, 1.0, 1.0)], 1_000_000_000)
    left_result = left.filter([pose_sample(12.0, 1.0, 0.02)], 1_040_000_000)[0]
    right_result = right.filter([pose_sample(88.0, 1.0, 0.98)], 1_040_000_000)[0]
    assert left_result["keypoints"][0]["x"] < 12.0 < 88.0 < right_result["keypoints"][0]["x"]


def test_pose_smoother_bridges_two_missing_results_without_buffering():
    smoother = main.PoseSmoother()
    smoother.filter([pose_sample(50.0, 0.9, 0.0)], 1_000_000_000)

    first_gap = smoother.filter([], 1_040_000_000)[0]["keypoints"][0]
    second_gap = smoother.filter([], 1_080_000_000)[0]["keypoints"][0]

    assert first_gap["x"] == second_gap["x"] == 50.0
    assert second_gap["confidence"] < first_gap["confidence"] < 0.9
    assert smoother.filter([], 1_120_000_000) == []


def test_pose_smoother_drops_stale_subject_after_coasting_without_pts():
    smoother = main.PoseSmoother()
    smoother.filter([pose_sample(50.0, 0.9, 0.0)], -1)
    assert smoother.filter([], -1) and smoother.filter([], -1)
    assert smoother.filter([], -1) == []
    assert smoother.filter([pose_sample(54.0, 0.4, 0.0)], -1)[0]["keypoints"][0]["x"] == 54.0
