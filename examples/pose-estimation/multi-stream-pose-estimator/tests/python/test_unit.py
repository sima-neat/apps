"""Unit tests for the multistream pose estimation Insight example."""

from __future__ import annotations

import copy
import json
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils.config_cases import config_writer, load_example_main
from tests.utils.fake_run import FakeRun

from tests.utils.metadata_json_listener import _MetadataReassembler

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"

main_module = load_example_main(EXAMPLE_DIR, "multi_stream_pose_estimator_main")

pytestmark = pytest.mark.unit


def metadata_chunk(message_id: int, index: int, count: int, payload: bytes) -> bytes:
    return bytes((0x4E, 0x01)) + struct.pack(">QBB", message_id, index, count) + payload


def write_config(
    tmp_path: Path,
    streams: list[str],
    codec: str | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_inflight_per_stream: int | None = None,
    max_inflight_total: int | None = None,
) -> Path:
    stream_lines = "\n".join(f"  - {stream}" for stream in streams)
    inference = []
    input_config = []
    if codec is not None or max_width is not None or max_height is not None:
        input_config.append("input:")
        if codec is not None:
            input_config.append(f"  codec: {codec}")
        if max_width is not None:
            input_config.append(f"  max_width: {max_width}")
        if max_height is not None:
            input_config.append(f"  max_height: {max_height}")
    if max_inflight_per_stream is not None or max_inflight_total is not None:
        inference.append("inference:")
        if max_inflight_per_stream is not None:
            inference.append(f"  max_inflight_per_stream: {max_inflight_per_stream}")
        if max_inflight_total is not None:
            inference.append(f"  max_inflight_total: {max_inflight_total}")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  path: models/yolo26m-pose-int8-b1.tar.gz",
                "streams:",
                stream_lines,
                *input_config,
                *inference,
                "output:",
                "  insight:",
                "    host: 127.0.0.1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


class TestMainEntrypoint:
    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--validate-config-only" in result.stdout

    def test_missing_config_file_fails_cleanly(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "does-not-exist.yaml"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 2
        assert "config file not found" in result.stderr


class TestConfigLoading:
    def test_load_app_config_accepts_four_streams(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
                "rtsp://127.0.0.1:8554/src3",
                "rtsp://127.0.0.1:8554/src4",
            ],
        )

        cfg = main_module.load_app_config(config_path)

        assert cfg.model_path == "models/yolo26m-pose-int8-b1.tar.gz"
        assert len(cfg.rtsp_urls) == 4
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30
        assert cfg.input_max_width == 1920
        assert cfg.input_max_height == 1080
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16

    def test_load_app_config_accepts_custom_input_capacity(self, tmp_path: Path):
        cfg = main_module.load_app_config(
            write_config(
                tmp_path,
                ["rtsp://127.0.0.1:8554/src1"],
                max_width=2560,
                max_height=1440,
            )
        )

        assert cfg.input_max_width == 2560
        assert cfg.input_max_height == 1440

    @pytest.mark.parametrize(("codec", "expected"), [("avc", "h264"), ("hevc", "h265")])
    def test_load_app_config_accepts_codec_alias(
        self, tmp_path: Path, codec: str, expected: str
    ):
        cfg = main_module.load_app_config(
            write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], codec=codec)
        )
        assert cfg.codec == expected

    def test_load_app_config_accepts_custom_inflight_limits(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            max_inflight_per_stream=3,
            max_inflight_total=12,
        )

        cfg = main_module.load_app_config(config_path)

        assert cfg.max_inflight_per_stream == 3
        assert cfg.max_inflight_total == 12

    def test_load_app_config_rejects_invalid_inflight_limit(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            max_inflight_per_stream=0,
        )

        with pytest.raises(
            ValueError, match="max_inflight_per_stream must be -1 or > 0"
        ):
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_too_many_streams(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
                "rtsp://127.0.0.1:8554/src3",
                "rtsp://127.0.0.1:8554/src4",
                "rtsp://127.0.0.1:8554/src5",
            ],
        )

        with pytest.raises(ValueError, match="up to four streams"):
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_empty_streams(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: models/yolo26m-pose-int8-b1.tar.gz
                streams: []
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="streams"):
            main_module.load_app_config(config_path)

    def test_validate_config_only_reports_stream_count(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
        )

        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--config",
                str(config_path),
                "--validate-config-only",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 0
        assert "streams=2" in result.stdout
        assert "max_inflight_per_stream=4" in result.stdout
        assert "max_inflight_total=16" in result.stdout


class TestRuntimeOptions:
    def test_probe_rtsp_forces_tcp_when_enabled(self, monkeypatch):
        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return {1: 2560, 2: 1440, 3: 20}[prop]

            def release(self):
                pass

        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
        monkeypatch.setattr(
            main_module,
            "cv2",
            SimpleNamespace(
                VideoCapture=lambda _url: FakeCapture(),
                CAP_PROP_FRAME_WIDTH=1,
                CAP_PROP_FRAME_HEIGHT=2,
                CAP_PROP_FPS=3,
            ),
        )

        assert main_module.probe_rtsp("rtsp://camera/stream", True) == (2560, 1440, 20)
        assert main_module.os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"

    def test_model_preprocess_uses_configured_capacity(self, monkeypatch):
        captured = SimpleNamespace(options=None)

        class FakeModelOptions:
            def __init__(self):
                self.preprocess = SimpleNamespace(color_convert=SimpleNamespace())

        def fake_model(_path, options):
            captured.options = options
            return object()

        monkeypatch.setattr(
            main_module,
            "pyneat",
            SimpleNamespace(
                ModelOptions=FakeModelOptions,
                InputKind=SimpleNamespace(Image="image"),
                AutoFlag=SimpleNamespace(On="on"),
                PreprocessColorFormat=SimpleNamespace(NV12="nv12"),
                NormalizePreset=SimpleNamespace(COCO_YOLO="coco-yolo"),
                BoxDecodeType=SimpleNamespace(YoloV26Pose="yolo26-pose"),
                Model=fake_model,
            ),
        )

        cfg = SimpleNamespace(
            model_path="pose.tar.gz",
            input_max_width=2560,
            input_max_height=1440,
            min_score=0.3,
            nms_iou=0.6,
            max_poses=50,
        )
        main_module.build_model(cfg)

        assert captured.options.preprocess.input_max_width == 2560
        assert captured.options.preprocess.input_max_height == 1440

    def test_encoded_input_options_carry_codec_format(self, monkeypatch):
        class FakeInputOptions:
            format = ""

        fake_pyneat = SimpleNamespace(
            InputOptions=FakeInputOptions,
            PayloadType=SimpleNamespace(Encoded="encoded"),
            Format=SimpleNamespace(H264="h264", H265="h265"),
            RtspCodec=SimpleNamespace(H264="codec-h264", H265="codec-h265"),
            InputMemoryPolicy=SimpleNamespace(Ev74="ev74", SystemMemory="system"),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        decode = main_module.encoded_decode_input_options(fake_pyneat.RtspCodec.H265)
        video = main_module.encoded_video_input_options(fake_pyneat.RtspCodec.H265)
        h264_decode = main_module.encoded_decode_input_options(fake_pyneat.RtspCodec.H264)
        h264_video = main_module.encoded_video_input_options(fake_pyneat.RtspCodec.H264)

        assert decode.format == "h265"
        assert video.format == "h265"
        assert h264_decode.format == "h264"
        assert h264_video.format == "h264"

    def test_realtime_link_sets_inflight_limits(self, monkeypatch):
        fake_pyneat = SimpleNamespace(
            GraphLinkOptions=type("GraphLinkOptions", (), {}),
            GraphLinkPolicy=SimpleNamespace(RealtimeLatestByStream="latest-by-stream"),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        link = main_module.realtime_link(2, 3, 12)

        assert link.policy == "latest-by-stream"
        assert link.stream_id == "stream2"
        assert link.max_inflight_per_stream == 3
        assert link.max_inflight_total == 12

class TestPullOutcomes:
    """The pull loop, driven by a run that yields no sample.

    A timeout keeps the loop going, and a runtime error raised by the pull ends it.
    pyneat's pull returns None for a closed output as for a timeout, so Python cannot
    yet tell a stream that ended from a slow one.
    """

    @staticmethod
    def _app(run):
        return main_module.AppRuntime(graph=None, run=run, model=None, streams=[])

    def test_timeout_is_not_a_sample(self):
        run = FakeRun("timeout")
        cfg = SimpleNamespace(save_dir="", save_every=0)

        assert main_module.process_run_once(self._app(run), cfg, "poses") is False
        assert run.pulls == [("poses", 50)]

    def test_runtime_error_ends_the_run(self):
        run = FakeRun(("error", "queue torn down"))
        cfg = SimpleNamespace(save_dir="", save_every=0)

        with pytest.raises(RuntimeError, match="queue torn down"):
            main_module.process_run_once(self._app(run), cfg, "poses")


class TestMetadataReassembly:
    def test_passes_legacy_json_unchanged(self):
        payload = b'{"type":"pose-estimation"}'

        assert _MetadataReassembler().accept(payload) == (payload, "")

    def test_joins_out_of_order_chunks(self):
        reassembler = _MetadataReassembler()

        assert reassembler.accept(metadata_chunk(7, 1, 2, b"42}")) == (None, "")
        assert reassembler.accept(metadata_chunk(7, 0, 2, b'{"value":')) == (
            b'{"value":42}',
            "",
        )

    def test_ignores_identical_duplicate_chunks(self):
        reassembler = _MetadataReassembler()
        first = metadata_chunk(8, 0, 2, b'{"value":')

        assert reassembler.accept(first) == (None, "")
        assert reassembler.accept(first) == (None, "")
        assert reassembler.accept(metadata_chunk(8, 1, 2, b"42}")) == (
            b'{"value":42}',
            "",
        )

    def test_rejects_malformed_chunks(self):
        payload, error = _MetadataReassembler().accept(bytes((0x4E, 0x01)))

        assert payload is None
        assert error == "invalid metadata chunk header"


class FakeMetadataSender:
    def __init__(self):
        self.calls = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
        self.calls.append((metadata_type, data_json, timestamp_ms, frame_id))
        return True


class FakeSample:
    frame_id = 42
    pts_ns = 1_234_000_000


class TestMetadata:
    def test_send_metadata_uses_pose_estimation_contract(self):
        sender = FakeMetadataSender()
        runtime = main_module.StreamRuntime(
            index=0,
            source_options=None,
            metadata_sender=sender,
            profile=main_module.ProfileWindow(False, 0),
            latest_debug_frame=None,
            frame_w=100,
            frame_h=100,
        )
        poses = [
            {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 40.0,
                "y2": 60.0,
                "score": 0.75,
                "keypoints": [
                    {"x": 11.0, "y": 21.0, "visibility": 0.9 if index == 0 else 0.1}
                    for index in range(17)
                ],
            }
        ]

        main_module.send_metadata(runtime, FakeSample(), poses)

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "pose-estimation"
        assert timestamp_ms == 1234
        assert frame_id == "42"
        payload = json.loads(data_json)
        assert len(payload["poses"]) == 1
        pose = payload["poses"][0]
        assert pose["id"] == "pose_1"
        assert pose["label"] == "person"
        assert pose["confidence"] == 0.75
        assert pose["bbox"] == [10, 20, 30, 40]
        assert len(pose["keypoints"]) == 17
        assert pose["keypoints"][0] == {
            "name": "nose",
            "x": 11,
            "y": 21,
            "confidence": 0.9,
        }
        assert pose["keypoints"][-1]["name"] == "right_ankle"

    def test_max_pose_metadata_fits_core_message_limit(self):
        points = [
            {"x": 1234.56789, "y": 1234.56789, "visibility": 0.987654321}
            for _ in range(17)
        ]
        poses = [
            {
                "x1": 1234.56789,
                "y1": 1234.56789,
                "x2": 2469.13578,
                "y2": 2469.13578,
                "score": 0.987654321,
                "keypoints": points,
            }
            for _ in range(50)
        ]

        payload = json.dumps(main_module.pose_metadata_data(poses), separators=(",", ":"))

        assert len(payload.encode("utf-8")) <= 65507
        assert all(len(pose["keypoints"]) == 17 for pose in json.loads(payload)["poses"])


# ---------------------------------------------------------------------------
# Configuration handling and option validation (Refs #526).
# Each test starts from VALID_CONFIG and breaks exactly one thing, so a failure
# names the rule that fired rather than "config rejected". The stream cap, the
# empty stream list and the invalid inflight limit are owned by
# TestConfigLoading above and are not repeated here.
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "model": {"path": "model.tar.gz"},
    "streams": ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
    "input": {"codec": "h264", "latency_ms": 100, "tcp": True, "max_width": 1920, "max_height": 1080},
    "inference": {
        "frames": 0,
        "max_inflight_per_stream": 4,
        "max_inflight_total": 16,
        "min_score": 0.55,
        "nms_iou": 0.6,
        "max_poses": 50,
    },
    "runtime": {"profile": False, "warmup_frames": 30},
    "output": {
        "video_enabled": True,
        "save_every": 0,
        "min_keypoint_visibility": 0.3,
        "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100},
    },
}


# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_full_config = config_writer(VALID_CONFIG)


class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main_module.load_app_config(write_full_config(tmp_path))

        assert cfg.model_path == "model.tar.gz"
        assert len(cfg.rtsp_urls) == 2
        assert cfg.codec == "h264"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": "model.tar.gz"},
            "streams": ["rtsp://127.0.0.1:8554/src1"],
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main_module.load_app_config(write_full_config(tmp_path, root=raw))

        assert cfg.codec == "h264"
        assert cfg.latency_ms == 100
        assert cfg.input_max_width == 1920
        assert cfg.input_max_height == 1080
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16
        assert cfg.min_score == pytest.approx(0.55)
        assert cfg.nms_iou == pytest.approx(0.60)
        assert cfg.max_poses == 50
        assert cfg.min_keypoint_visibility == pytest.approx(0.30)
        assert cfg.warmup_frames == 30
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100
        assert cfg.video_enabled == True
        assert cfg.save_every == 0


REJECTED = [
    pytest.param(('model', 'path'), '', 'model.path must be set', id='model-path-empty'),
    pytest.param(('output', 'insight', 'host'), '', 'output.insight.host must be set', id='insight-host-empty'),
    pytest.param(('input', 'codec'), 'vp9', 'input.codec must be h264/avc or h265/hevc', id='codec-unsupported'),
    pytest.param(('input', 'latency_ms'), -1, 'input.latency_ms must be >= 0', id='latency-negative'),
    pytest.param(('input', 'max_width'), 0, 'input.max_width must be > 0', id='max-width-zero'),
    pytest.param(('input', 'max_height'), 0, 'input.max_height must be > 0', id='max-height-zero'),
    pytest.param(('inference', 'frames'), -1, 'inference.frames must be >= 0', id='frames-negative'),
    pytest.param(('inference', 'min_score'), -0.01, 'inference.min_score must be between 0 and 1', id='min-score-below'),
    pytest.param(('inference', 'min_score'), 1.01, 'inference.min_score must be between 0 and 1', id='min-score-above'),
    pytest.param(('inference', 'nms_iou'), -0.01, 'inference.nms_iou must be between 0 and 1', id='nms-below'),
    pytest.param(('inference', 'nms_iou'), 1.01, 'inference.nms_iou must be between 0 and 1', id='nms-above'),
    pytest.param(('inference', 'max_poses'), 0, 'inference.max_poses must be > 0', id='max-poses-zero'),
    pytest.param(('output', 'min_keypoint_visibility'), 1.01, 'output.min_keypoint_visibility must be between 0 and 1', id='keypoint-visibility-above'),
    pytest.param(('runtime', 'warmup_frames'), -1, 'runtime.warmup_frames must be >= 0', id='warmup-negative'),
    pytest.param(('output', 'insight', 'video_port_base'), 0, 'output.insight.video_port_base must be > 0', id='video-port-base-zero'),
    pytest.param(('output', 'insight', 'metadata_port_base'), 0, 'output.insight.metadata_port_base must be > 0', id='metadata-port-base-zero'),
    pytest.param(('output', 'save_every'), -1, 'output.save_every must be >= 0', id='save-every-negative'),
]


class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        with pytest.raises(ValueError) as excinfo:
            main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))

        assert message in str(excinfo.value)


class TestBoundariesAreAccepted:
    """An off-by-one that rejects a legal value is the failure nobody writes a test for."""

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (('inference', 'min_score'), 0.0),
            (('inference', 'min_score'), 1.0),
            (('inference', 'nms_iou'), 0.0),
            (('inference', 'nms_iou'), 1.0),
            (('output', 'min_keypoint_visibility'), 0.0),
            (('output', 'min_keypoint_visibility'), 1.0),
            (('input', 'latency_ms'), 0),
            (('input', 'max_width'), 1),
            (('input', 'max_height'), 1),
            (('inference', 'frames'), 0),
            (('runtime', 'warmup_frames'), 0),
            (('output', 'save_every'), 0),
            (('inference', 'max_poses'), 1),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))


class TestStreamList:
    def test_a_missing_stream_list_is_rejected(self, tmp_path):
        raw = copy.deepcopy(VALID_CONFIG)
        del raw["streams"]

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(write_full_config(tmp_path, root=raw))

    def test_a_scalar_instead_of_a_list_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(
                write_full_config(tmp_path, [(("streams",), "rtsp://127.0.0.1:8554/src1")])
            )

    @pytest.mark.parametrize(
        ("streams", "bad_index"),
        [
            ([""], 0),
            (["   "], 0),
            (["rtsp://127.0.0.1:8554/src1", ""], 1),
            (["rtsp://127.0.0.1:8554/src1", 42], 1),
        ],
    )
    def test_a_blank_or_non_string_entry_names_its_index(self, tmp_path, streams, bad_index):
        """The index matters: with four streams, "one of them is wrong" is not actionable."""
        with pytest.raises(ValueError, match=rf"streams\[{bad_index}\] must be a non-empty string"):
            main_module.load_app_config(write_full_config(tmp_path, [(("streams",), streams)]))

    def test_stream_order_is_preserved(self, tmp_path):
        """Stream order decides which Insight port each stream publishes on."""
        streams = ["rtsp://host/c", "rtsp://host/a", "rtsp://host/b"]

        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("streams",), streams)]))

        assert cfg.rtsp_urls == streams


class TestInflightSentinels:
    """-1 means unbounded; 0 and negatives other than -1 are mistakes."""

    @pytest.mark.parametrize("key", ["max_inflight_per_stream", "max_inflight_total"])
    @pytest.mark.parametrize("value", [-1, 1, 64])
    def test_unbounded_and_positive_values_are_accepted(self, tmp_path, key, value):
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("inference", key), value)]))

        assert getattr(cfg, key) == value

    @pytest.mark.parametrize("key", ["max_inflight_per_stream", "max_inflight_total"])
    @pytest.mark.parametrize("value", [0, -2])
    def test_zero_and_other_negatives_are_rejected(self, tmp_path, key, value):
        with pytest.raises(ValueError, match=f"inference.{key} must be -1 or > 0"):
            main_module.load_app_config(write_full_config(tmp_path, [(("inference", key), value)]))


class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main_module.load_app_config(config_path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(config_path)
