"""Unit tests for the multi-stream YOLO26-to-BlazePose application."""

from __future__ import annotations

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

IDENTITY = main.FrameIdentity("camera0", 7, 1_234_000_000)
BOX = {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0, "score": 0.9, "class_id": 0}
UNIT_AFFINE = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def stream(index: int, *, channel: int | None = None, **fields) -> dict:
    return {
        "id": f"camera{index}",
        "url": f"rtsp://127.0.0.1/src{index}",
        "codec": "h265" if index == 1 else "h264",
        "insight_channel": index if channel is None else channel,
        "width": 640,
        "height": 480,
        "fps": 30,
        **fields,
    }


def config_text(streams=None, insight=None, **sections) -> str:
    config = {
        "models": {"detector_path": "detector.tar.gz", "pose_path": "pose.tar.gz"},
        "streams": [stream(0)] if streams is None else streams,
        **sections,
        "output": {"insight": {"host": "127.0.0.1", **(insight or {})}},
    }
    return yaml.safe_dump(config, sort_keys=False)


def load_config(tmp_path: Path, text: str) -> main.AppConfig:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return main.load_app_config(path)


class RecordingSender:
    def __init__(self, fail: str | None = None):
        self.calls = []
        self.fail = fail

    def send_metadata(self, *args):
        self.calls.append(args)
        if self.fail == "raises" and args[0] == "auxiliary-visualization":
            raise RuntimeError("metadata datagram could not be queued")
        return not (self.fail == "returns-false" and args[0] == "auxiliary-visualization")


def runtime_stream(sender=None, *, temporal_filter: bool = False, outstanding: int = 0):
    return SimpleNamespace(
        index=0,
        config=SimpleNamespace(id="camera0"),
        width=640,
        height=480,
        metadata_lock=threading.Lock(),
        metadata_sender=sender or RecordingSender(),
        pose_smoother=main.PoseSmoother(),
        last_published_frame_id=0,
        pose_temporal_filter_enabled=temporal_filter,
        frames_in=0,
        frames_out=0,
        outstanding_frames=outstanding,
        closed=False,
    )


class LateOutputRun:
    """Returns `outputs` samples, then reports a timeout and stops the app."""

    def __init__(self, state, outputs: int = 1):
        self.state = state
        self.outputs = outputs

    def pull(self, _name, _timeout):
        if self.outputs > 0:
            self.outputs -= 1
            return object()
        with self.state.condition:
            self.state.stopping = True
        return None

    @staticmethod
    def can_pull():
        return True


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


# test_config_validation() in test_unit.cpp runs the same cases against the C++ app.
REJECTED_CONFIGS = [
    ("no-host", config_text().replace("host: 127.0.0.1", "host: ''"), "host must be set"),
    ("five-streams", config_text([stream(i) for i in range(5)]), "between 1 and 4 entries"),
    ("duplicate-id", config_text([stream(0), stream(1, id="camera0")]), "ids must be unique"),
    ("numeric-id", config_text([stream(0, id=123)]), "id must be a string"),
    (
        "timestamp-id",
        config_text().replace("id: camera0", "id: 2026-10-04"),
        "id must be a string",
    ),
    ("boolean-url", config_text([stream(0, url=True)]), "url must be a string"),
    ("null-id", config_text([stream(0, id=None)]), "id must be set"),
    ("null-url", config_text([stream(0, url=None)]), "url must be set"),
    (
        "standalone-dash",
        config_text([stream(0), stream(1, id="camera0")]).replace(
            "- id: camera0", "-\n  id: camera0", 1
        ),
        "ids must be unique",
    ),
    ("duplicate-channel", config_text([stream(0), stream(1, channel=0)]), "channels must be"),
    (
        "port-overlap",
        config_text([stream(0), stream(1, channel=100)], {"video_port_base": 9000}),
        "video and metadata ports must not overlap",
    ),
    (
        "video-port-range",
        config_text([stream(0, channel=60000)], {"metadata_port_base": 1}),
        "stream video port must be <= 65535",
    ),
    (
        "metadata-port-range",
        config_text([stream(0, channel=60000)], {"video_port_base": 1}),
        "stream metadata port must be <= 65535",
    ),
    (
        "missing-caps",
        config_text([stream(0, width=0, height=0, fps=0)]),
        "width, height, and fps must all be > 0",
    ),
    ("quoted-width", config_text([stream(0, width="640")]), "width must be an integer"),
    ("unknown-codec", config_text([stream(0, codec="hevc")]), "codec must be h264 or h265"),
    ("eleven-people", config_text(pose={"max_people_per_frame": 11}), "between 1 and 10"),
    ("null-frames", config_text(runtime={"frames": None}), "frames must be an integer"),
    ("null-tcp", config_text(input={"tcp": None}), "tcp must be true or false"),
    ("quoted-frames", config_text(runtime={"frames": "1"}), "frames must be an integer"),
    ("quoted-tcp", config_text(input={"tcp": "true"}), "tcp must be true or false"),
    ("quoted-min-score", config_text(detector={"min_score": "0.5"}), "min_score must be numeric"),
] + [
    (f"roi-scale-{value}", config_text() + f"pose:\n  roi_scale: {value}\n", "finite and > 0")
    for value in (".nan", ".inf", "0")
]


@pytest.mark.parametrize(
    ("text", "message"),
    [case[1:] for case in REJECTED_CONFIGS],
    ids=[case[0] for case in REJECTED_CONFIGS],
)
def test_config_validation(tmp_path: Path, text: str, message: str):
    with pytest.raises((TypeError, ValueError), match=message):
        load_config(tmp_path, text)


def test_config_reads_streams_and_settings(tmp_path: Path):
    cfg = load_config(
        tmp_path,
        config_text(
            [stream(0, width=1920, height=1080, fps=30), stream(1)],
            pose={"temporal_filter_enabled": False},
            runtime={"frames": 8},
        ),
    )
    assert cfg.streams == [
        main.StreamConfig("camera0", "rtsp://127.0.0.1/src0", "h264", 0, 1920, 1080, 30),
        main.StreamConfig("camera1", "rtsp://127.0.0.1/src1", "h265", 1, 640, 480, 30),
    ]
    assert (cfg.pose_temporal_filter_enabled, cfg.frame_limit) == (False, 8)
    assert main.load_app_config(main.DEFAULT_CONFIG).pose_temporal_filter_enabled


def test_config_defaults_null_codec_and_decodes_quoted_strings(tmp_path: Path):
    text = config_text([stream(0, codec=None)])
    text = text.replace("host: 127.0.0.1", 'host: "127.0.0.\\x31"')
    cfg = load_config(tmp_path, text)
    assert cfg.streams[0].codec == "h264"
    assert cfg.insight_host == "127.0.0.1"


@pytest.mark.parametrize(
    "body",
    [
        "streams: [{id: camera0, url: rtsp://127.0.0.1/src0, insight_channel: 0}]\n",
        "streams:\n  - {id: camera0, url: rtsp://127.0.0.1/src0, insight_channel: 0}\n",
        "runtime: {frames: 1}\n",
    ],
)
def test_flow_style_collections_are_rejected_like_cpp(tmp_path: Path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(config_text() + body, encoding="utf-8")
    with pytest.raises(ValueError, match="flow-style YAML collections are not supported"):
        main.load_app_config(path)


def test_empty_flow_collections_stay_allowed(tmp_path: Path):
    cfg = load_config(tmp_path, config_text() + "pose: {}\n")
    assert len(cfg.streams) == 1


def test_legacy_yaml_booleans_match_cpp(tmp_path: Path):
    cfg = load_config(
        tmp_path,
        config_text() + "input:\n  tcp: yes\npose:\n  temporal_filter_enabled: OFF\n",
    )
    assert cfg.tcp is True
    assert cfg.pose_temporal_filter_enabled is False


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


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_source_frame_rate_is_a_decoder_hint_not_a_caps_pin(monkeypatch, codec: str):
    # A 29.97 fps camera probes as 30 but negotiates 30000/1001; pinning 30/1 into
    # the encoded or raw caps stops that stream with an incompatible-caps error.
    monkeypatch.setattr(
        main,
        "pyneat",
        SimpleNamespace(
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
        ),
    )
    stream_cfg = main.StreamConfig("camera0", "rtsp://127.0.0.1/src0", codec, 0)
    cfg = main.AppConfig("detector.tar.gz", "pose.tar.gz", [stream_cfg])
    options = main.build_source_options(cfg, stream_cfg, 1280, 720, 30)
    assert options.dec_fps == 30
    assert options.source_fps == -1
    assert options.fallback_h264_fps == (30 if codec == "h264" else -1)

    _, encoded = main.make_encoded_source(options).nodes[0]
    assert encoded.source_fps == -1
    assert encoded.fallback_h264_fps == options.fallback_h264_fps

    decoder = main.make_decoder(options).nodes
    _, decode = decoder[1]
    assert decode.dec_fps == 30
    assert [args for kind, args in decoder if kind == "caps"] == [("NV12", 1280, 720, -1, "any")]


def test_roi_landmark_and_metadata_contract():
    box = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0, "score": 0.9, "class_id": 0}
    assert main.square_roi(box, 1.5) == (-10, 10, 60, 60)
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[0] = [4.0, 8.0, 0.0, 2.0, -2.0]
    raw_world = np.zeros((39, 3), dtype=np.float32)
    raw_world[0] = [0.1, -0.2, 0.3]
    pose = main.decode_pose(raw, raw_world, (2.0, 0.0, 10.0, 0.0, 3.0, 20.0), box, 0.93, 2)
    nose, world_nose = pose["keypoints"][0], pose["world_keypoints"][0]
    assert (nose["x"], nose["y"]) == pytest.approx((18.0, 44.0))
    assert nose["confidence"] == world_nose["confidence"] == pytest.approx(main.sigmoid(-2.0))
    assert (world_nose["x"], world_nose["y"], world_nose["z"]) == pytest.approx((0.1, -0.2, 0.3))

    data = main.poses_data([pose], "camera0")
    published = data["poses"][0]
    assert data["stream_id"] == "camera0"
    assert (published["id"], published["presence"]) == ("pose_3", pytest.approx(0.93))
    assert len(published["keypoints"]) == len(published["world_keypoints"]) == 33
    assert published["keypoints"][0]["name"] == published["world_keypoints"][0]["name"] == "nose"

    auxiliary = main.world_pose_auxiliary_from_overlay(data)
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


def test_select_people_discards_non_finite_scores(monkeypatch):
    boxes = [
        {**BOX, "score": float("nan")},
        {**BOX, "score": float("inf")},
        {**BOX, "score": 0.9},
        {**BOX, "score": 1.0, "class_id": 1},
    ]
    monkeypatch.setattr(main, "extract_bbox_payload", lambda _sample: b"bbox")
    monkeypatch.setattr(main, "parse_boxes_strict", lambda *_args: boxes)
    stream = SimpleNamespace(width=640, height=480)
    cfg = SimpleNamespace(max_people_per_frame=4)

    assert main.select_people(object(), stream, cfg) == [boxes[2]]


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
    context = main.PoseInputContext(1, 0, BOX, UNIT_AFFINE)
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


def pose_sample(x: float, world_x: float, box_x: float = 0.0) -> dict:
    return {
        "roi_index": 0,
        "presence": 0.9,
        "box": {**BOX, "x1": box_x, "x2": box_x + 100.0},
        "keypoints": [{"name": "nose", "x": x, "y": 50.0, "confidence": 0.9}],
        "world_keypoints": [{"name": "nose", "x": world_x, "y": 0.0, "z": 0.0, "confidence": 0.9}],
    }


def test_pose_smoother_blends_matched_2d_and_world_landmarks_per_stream():
    smoother, other_stream = main.PoseSmoother(), main.PoseSmoother()
    smoother.filter([pose_sample(50.0, 0.0)])
    other_stream.filter([pose_sample(90.0, 1.0)])

    matched = smoother.filter([pose_sample(54.0, 0.04)])[0]
    assert matched["keypoints"][0]["x"] == pytest.approx(52.0)
    assert matched["world_keypoints"][0]["x"] == pytest.approx(0.02)
    unmatched = smoother.filter([pose_sample(400.0, 1.0, box_x=300.0)])[0]
    assert unmatched["keypoints"][0]["x"] == 400.0
    assert other_stream.filter([pose_sample(80.0, 1.0)])[0]["keypoints"][0]["x"] == 85.0


def test_keep_latest_replaces_queued_work():
    mailboxes = [None, None]
    assert not main.keep_latest(mailboxes, 1, "first")
    assert main.keep_latest(mailboxes, 1, "second")
    assert mailboxes == [None, "second"]


@pytest.mark.parametrize("failure", [None, "returns-false", "raises"])
def test_publish_frame_sends_a_correlated_pair_and_counts_only_full_pairs(failure):
    sender = RecordingSender(failure)
    stream_runtime = runtime_stream(sender, temporal_filter=True)

    main.publish_frame(stream_runtime, IDENTITY, [])

    assert [call[0] for call in sender.calls] == ["pose-estimation", "auxiliary-visualization"]
    assert all(call[2:] == (1234, "7") for call in sender.calls)
    assert json.loads(sender.calls[0][1]) == {"stream_id": "camera0", "poses": []}
    auxiliary = json.loads(sender.calls[1][1])
    assert (auxiliary["stream_id"], auxiliary["payload"]) == ("camera0", {"poses": []})
    assert stream_runtime.frames_out == (1 if failure is None else 0)


def test_publish_frame_discards_a_pose_result_older_than_an_empty_frame():
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender, temporal_filter=True)
    newer = main.FrameIdentity("camera0", 8, 1_267_000_000)

    main.publish_frame(stream_runtime, newer, [])
    main.publish_frame(stream_runtime, IDENTITY, [pose_sample(50.0, 0.0)])

    assert [call[0] for call in sender.calls] == [
        "pose-estimation",
        "auxiliary-visualization",
    ]
    assert stream_runtime.last_published_frame_id == 8
    assert stream_runtime.pose_smoother.previous == []


def test_pushed_samples_carry_stream_id_frame_id_and_pts(monkeypatch):
    monkeypatch.setattr(
        main,
        "pyneat",
        SimpleNamespace(
            make_tensor_sample=lambda *_args: SimpleNamespace(),
            PayloadType=SimpleNamespace(Image="image", Tensor="tensor"),
        ),
    )
    tensor = SimpleNamespace(semantic=SimpleNamespace(tess=SimpleNamespace(format="BF16")))
    for sample in (
        main.image_input_sample("detector_input", tensor, IDENTITY),
        main.pose_input_sample(tensor, IDENTITY),
    ):
        assert (sample.stream_id, sample.frame_id, sample.pts_ns) == ("camera0", 7, 1_234_000_000)


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
    run = Run()
    assert main.push_with_context(runtime, run, "input", object(), pending, "context", "closed")
    assert run.attempts == 3
    assert list(pending) == ["context"]


def test_detector_outputs_keep_only_the_latest_frame_with_people(monkeypatch):
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender, outstanding=3)
    state = main.SharedState(1)
    jobs = [main.FrameJob(job_id, 0, object(), IDENTITY) for job_id in (1, 2, 3)]
    state.pending_detector_outputs.extend(jobs)
    people = iter([[], [BOX], [BOX]])
    monkeypatch.setattr(main, "select_people", lambda *_args: next(people))
    runtime = SimpleNamespace(
        state=state, streams=[stream_runtime], detector_run=LateOutputRun(state, outputs=3)
    )

    main.pull_detector_outputs(runtime, SimpleNamespace())

    assert state.error is None
    # Job 1 had nobody and job 2 was replaced by job 3; both are finished.
    assert state.pose_mailboxes == [jobs[2]]
    # Job 1 still publishes an empty pair so Insight clears stale poses.
    assert [(call[0], json.loads(call[1])["stream_id"]) for call in sender.calls] == [
        ("pose-estimation", "camera0"),
        ("auxiliary-visualization", "camera0"),
    ]
    assert stream_runtime.outstanding_frames == 1


def test_accepted_input_without_output_stops_the_app(monkeypatch):
    monkeypatch.setattr(main, "INFERENCE_STALL_TIMEOUT_S", 0.05)
    run = SimpleNamespace(pull=lambda *_args: None, can_pull=lambda: True)
    runtime = SimpleNamespace(state=main.SharedState(1))
    with pytest.raises(RuntimeError, match="BlazePose inference stalled: no output for 0.05 s"):
        main.pull_model_output(runtime, run, "pose_output", "BlazePose", main.deque(["roi"]))


def test_pose_outputs_publish_a_frame_once_all_its_rois_return(monkeypatch):
    sender = RecordingSender()
    stream_runtime = runtime_stream(sender, outstanding=1)
    state = main.SharedState(1)
    state.aggregates[1] = main.PoseAggregate(0, 2, IDENTITY)
    state.pending_pose_outputs.extend(
        main.PoseInputContext(1, index, BOX, UNIT_AFFINE) for index in (0, 1)
    )
    poses = iter([pose_sample(50.0, 0.0), None])
    monkeypatch.setattr(main, "parse_pose_output", lambda *_args: next(poses))
    runtime = SimpleNamespace(
        state=state, streams=[stream_runtime], pose_run=LateOutputRun(state, outputs=2)
    )

    main.pull_pose_outputs(runtime, SimpleNamespace())

    assert state.error is None
    assert not state.aggregates
    assert [call[0] for call in sender.calls] == ["pose-estimation", "auxiliary-visualization"]
    assert len(json.loads(sender.calls[0][1])["poses"]) == 1
    assert (stream_runtime.frames_out, stream_runtime.outstanding_frames) == (1, 0)


def test_finite_run_completes_only_when_every_stream_reached_its_limit():
    closed = [
        SimpleNamespace(
            config=SimpleNamespace(id=f"camera{index}"),
            closed=True,
            outstanding_frames=0,
            frames_in=frames,
        )
        for index, frames in enumerate((8, 3))
    ]
    runtime = SimpleNamespace(streams=closed)
    assert main.all_streams_done(runtime)
    closed[1].outstanding_frames = 1
    assert not main.all_streams_done(runtime)
    with pytest.raises(RuntimeError, match="before reaching runtime.frames=8: camera1"):
        main.require_successful_completion(runtime, 8)
    with pytest.raises(RuntimeError, match="all source streams stopped"):
        main.require_successful_completion(runtime, 0)


def test_each_source_run_starts_and_is_pulled_on_its_own_thread(monkeypatch):
    build_threads, pull_threads = set(), set()
    build_barrier, pull_barrier = threading.Barrier(2), threading.Barrier(2)

    class ClosedRun:
        def pull(self, _output, timeout_ms):
            assert timeout_ms == -1
            pull_threads.add(threading.get_ident())
            pull_barrier.wait(timeout=1)
            return None

    class SourceGraph:
        def build(self, _options):
            build_threads.add(threading.get_ident())
            build_barrier.wait(timeout=1)
            return ClosedRun()

    monkeypatch.setattr(
        main,
        "pyneat",
        SimpleNamespace(
            RunOptions=type("RunOptions", (), {}),
            RunPreset=SimpleNamespace(Realtime="realtime"),
            OutputMemory=SimpleNamespace(ZeroCopy="zero-copy"),
        ),
    )
    streams = [runtime_stream() for _ in range(2)]
    for index, stream_runtime in enumerate(streams):
        stream_runtime.index, stream_runtime.source_graph = index, SourceGraph()
    runtime = SimpleNamespace(streams=streams, state=main.SharedState(2))
    cfg = SimpleNamespace(frame_limit=0)
    pullers = [
        threading.Thread(target=main.run_source_stream, args=(runtime, cfg, stream_runtime))
        for stream_runtime in streams
    ]

    for puller in pullers:
        puller.start()
    for puller in pullers:
        puller.join(timeout=2)

    assert all(not puller.is_alive() for puller in pullers)
    assert len(build_threads) == len(pull_threads) == 2
    assert all(stream_runtime.closed for stream_runtime in streams)


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


def test_shutdown_summary_reports_frames_in_and_out_per_stream(capsys):
    stream_runtime = runtime_stream()
    stream_runtime.frames_in, stream_runtime.frames_out = 8, 5
    main.print_summary(SimpleNamespace(streams=[stream_runtime]))
    assert capsys.readouterr().out == "[summary stream=camera0] frames_in=8 frames_out=5\n"
