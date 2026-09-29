"""Unit tests for the multistream people tracker Insight example."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

# The configuration tests bind `main` under an example-specific module name:
# every application has a module called `main`, and a plain `import main`
# binds whichever was imported first when pytest runs across examples.
_CONFIG_SPEC = importlib.util.spec_from_file_location("multi_stream_people_tracker_main", MAIN_PY)
assert _CONFIG_SPEC is not None and _CONFIG_SPEC.loader is not None
main_module = importlib.util.module_from_spec(_CONFIG_SPEC)
sys.modules[_CONFIG_SPEC.name] = main_module
_CONFIG_SPEC.loader.exec_module(main_module)

pytestmark = pytest.mark.unit


def write_config(
    tmp_path: Path,
    streams: list[str],
    codec: str | None = None,
    max_inflight_per_stream: int | None = None,
    max_inflight_total: int | None = None,
) -> Path:
    stream_lines = "\n".join(f"  - {stream}" for stream in streams)
    inference = []
    input_config = ["input:", f"  codec: {codec}"] if codec else []
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
                "  path: models/yolo26m-det-int8-b1.tar.gz",
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
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 2
        assert "config file not found" in result.stderr


class TestConfigLoading:
    def test_load_app_config_accepts_four_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
                "rtsp://127.0.0.1:8554/src3",
                "rtsp://127.0.0.1:8554/src4",
            ],
        )

        cfg = load_app_config(config_path)

        assert cfg.model_path == "models/yolo26m-det-int8-b1.tar.gz"
        assert len(cfg.rtsp_urls) == 4
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30
        assert cfg.tracker_max_missing == 15
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16

    def test_load_app_config_accepts_hevc(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(
            write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], codec="hevc")
        )
        assert cfg.codec == "h265"

    def test_load_app_config_accepts_custom_inflight_limits(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            max_inflight_per_stream=3,
            max_inflight_total=12,
        )

        cfg = load_app_config(config_path)

        assert cfg.max_inflight_per_stream == 3
        assert cfg.max_inflight_total == 12

    def test_load_app_config_rejects_invalid_inflight_limit(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            max_inflight_per_stream=0,
        )

        with pytest.raises(ValueError, match="max_inflight_per_stream must be -1 or > 0"):
            load_app_config(config_path)

    def test_default_config_uses_tracking_threshold(self):
        from main import load_app_config

        cfg = load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.min_score == 0.30

    def test_load_app_config_rejects_too_many_streams(self, tmp_path: Path):
        from main import load_app_config

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
            load_app_config(config_path)

    def test_load_app_config_rejects_empty_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: models/yolo26m-det-int8-b1.tar.gz
                streams: []
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="streams"):
            load_app_config(config_path)

    def test_validate_config_only_reports_stream_count(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path), "--validate-config-only"],
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
    def test_encoded_input_options_carry_codec_format(self, monkeypatch):
        import main

        class FakeInputOptions:
            format = ""

        fake_pyneat = SimpleNamespace(
            InputOptions=FakeInputOptions,
            PayloadType=SimpleNamespace(Encoded="encoded"),
            Format=SimpleNamespace(H264="h264", H265="h265"),
            RtspCodec=SimpleNamespace(H264="codec-h264", H265="codec-h265"),
            InputMemoryPolicy=SimpleNamespace(Ev74="ev74", SystemMemory="system"),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        decode = main.encoded_decode_input_options(fake_pyneat.RtspCodec.H265)
        video = main.encoded_video_input_options(fake_pyneat.RtspCodec.H265)
        h264_decode = main.encoded_decode_input_options(fake_pyneat.RtspCodec.H264)
        h264_video = main.encoded_video_input_options(fake_pyneat.RtspCodec.H264)

        assert decode.format == "h265"
        assert video.format == "h265"
        assert h264_decode.format == "h264"
        assert h264_video.format == "h264"

    def test_realtime_link_sets_inflight_limits(self, monkeypatch):
        import main

        fake_pyneat = SimpleNamespace(
            GraphLinkOptions=type("GraphLinkOptions", (), {}),
            GraphLinkPolicy=SimpleNamespace(RealtimeLatestByStream="latest-by-stream"),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        link = main.realtime_link(2, 4, 3, 12)

        assert link.policy == "latest-by-stream"
        assert link.queue_depth == 4
        assert link.stream_id == "stream2"
        assert link.max_inflight_per_stream == 3
        assert link.max_inflight_total == 12


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
    def test_send_metadata_uses_tracking_contract(self):
        from main import ProfileWindow, StreamRuntime, send_metadata
        from utils.tracker import PeopleTracker, TrackedDetection

        sender = FakeMetadataSender()
        runtime = StreamRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            source_options=None,
            metadata_sender=sender,
            tracker=PeopleTracker(),
            profile=ProfileWindow(False, 0),
            latest_debug_frame=None,
            frame_w=100,
            frame_h=100,
            output_fps=30,
            video_port=9000,
        )
        tracks = [TrackedDetection(7, 10.0, 20.0, 40.0, 60.0, 0.75, 0)]

        send_metadata(runtime, FakeSample(), tracks)

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "tracking"
        assert timestamp_ms == 1234
        assert frame_id == "42"
        assert json.loads(data_json) == {
            "tracks": [
                {
                    "id": "7",
                    "label": "person",
                    "confidence": 0.75,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                }
            ]
        }


class TestTracker:
    def test_tracker_reuses_track_id_for_nearby_detection(self):
        from utils.tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=2)
        first = tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 12.0, "y1": 11.0, "x2": 52.0, "y2": 81.0, "score": 0.8, "class_id": 0}],
            frame_index=1,
        )

        assert len(first) == 1
        assert len(second) == 1
        assert first[0].track_id == second[0].track_id

    def test_tracker_drops_track_after_missing_budget(self):
        from utils.tracker import PeopleTracker

        tracker = PeopleTracker(iou_threshold=0.3, max_missing_frames=1)
        tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        tracker.update([], frame_index=1)
        tracker.update([], frame_index=2)

        assert tracker.active_track_count() == 0



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
    "input": {"codec": "h264", "latency_ms": 100, "tcp": True},
    "inference": {
        "frames": 0,
        "fps": 0,
        "max_inflight_per_stream": 4,
        "max_inflight_total": 16,
        "person_class_id": 0,
        "min_score": 0.55,
        "nms_iou": 0.6,
        "max_detections": 50,
    },
    "runtime": {"profile": False, "warmup_frames": 30},
    "tracking": {"iou_threshold": 0.3, "max_missing_frames": 15},
    "output": {
        "video_enabled": True,
        "save_every": 0,
        "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100},
    },
}


def write_full_config(tmp_path: Path, overrides=None, *, root=None) -> Path:
    """Write VALID_CONFIG with `overrides` applied as ((section, ..., key), value)."""
    raw = copy.deepcopy(VALID_CONFIG) if root is None else root
    for path, value in (overrides or []):
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


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
        assert cfg.tcp == True
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16
        assert cfg.person_class_id == 0
        assert cfg.min_score == pytest.approx(0.55)
        assert cfg.nms_iou == pytest.approx(0.60)
        assert cfg.max_detections == 50
        assert cfg.warmup_frames == 30
        assert cfg.tracker_iou_threshold == pytest.approx(0.3)
        assert cfg.tracker_max_missing == 15
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100
        assert cfg.video_enabled == True
        assert cfg.save_every == 0


REJECTED = [
    pytest.param(('model', 'path'), '', 'model.path must be set', id='model-path-empty'),
    pytest.param(('output', 'insight', 'host'), '', 'output.insight.host must be set', id='insight-host-empty'),
    pytest.param(('input', 'codec'), 'vp9', 'input.codec must be h264/avc or h265/hevc', id='codec-unsupported'),
    pytest.param(('input', 'latency_ms'), -1, 'input.latency_ms must be >= 0', id='latency-negative'),
    pytest.param(('inference', 'frames'), -1, 'inference.frames must be >= 0', id='frames-negative'),
    pytest.param(('inference', 'fps'), -1, 'inference.fps must be >= 0', id='fps-negative'),
    pytest.param(('inference', 'person_class_id'), -1, 'inference.person_class_id must be >= 0', id='person-class-negative'),
    pytest.param(('inference', 'min_score'), -0.01, 'inference.min_score must be between 0 and 1', id='min-score-below'),
    pytest.param(('inference', 'min_score'), 1.01, 'inference.min_score must be between 0 and 1', id='min-score-above'),
    pytest.param(('inference', 'nms_iou'), -0.01, 'inference.nms_iou must be between 0 and 1', id='nms-below'),
    pytest.param(('inference', 'nms_iou'), 1.01, 'inference.nms_iou must be between 0 and 1', id='nms-above'),
    pytest.param(('inference', 'max_detections'), 0, 'inference.max_detections must be > 0', id='max-detections-zero'),
    pytest.param(('runtime', 'warmup_frames'), -1, 'runtime.warmup_frames must be >= 0', id='warmup-negative'),
    pytest.param(('tracking', 'iou_threshold'), 1.01, 'tracking.iou_threshold must be between 0 and 1', id='tracking-iou-above'),
    pytest.param(('tracking', 'max_missing_frames'), -1, 'tracking.max_missing_frames must be >= 0', id='max-missing-negative'),
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
            (('tracking', 'iou_threshold'), 0.0),
            (('tracking', 'iou_threshold'), 1.0),
            (('input', 'latency_ms'), 0),
            (('inference', 'frames'), 0),
            (('inference', 'fps'), 0),
            (('inference', 'person_class_id'), 0),
            (('runtime', 'warmup_frames'), 0),
            (('tracking', 'max_missing_frames'), 0),
            (('output', 'save_every'), 0),
            (('inference', 'max_detections'), 1),
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


class TestCodecAliases:
    @pytest.mark.parametrize(
        ("alias", "codec"), [("h264", "h264"), ("avc", "h264"), ("h265", "h265"), ("hevc", "h265")]
    )
    def test_every_documented_alias_is_accepted(self, tmp_path, alias, codec):
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("input", "codec"), alias)]))

        assert cfg.codec == codec


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
