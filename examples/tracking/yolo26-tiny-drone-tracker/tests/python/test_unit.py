"""Unit tests for the YOLO26 tiny-drone tracker Insight example."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import textwrap
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils.config_cases import config_writer, load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"

main_module = load_example_main(EXAMPLE_DIR, "yolo26_tiny_drone_tracker_main")

pytestmark = pytest.mark.unit


def write_config(
    tmp_path: Path,
    streams: list[str],
    codec: str | None = None,
    max_inflight_per_stream: int | None = None,
    max_inflight_total: int | None = None,
    video_port_base: int | None = None,
    metadata_port_base: int | None = None,
    video_enabled: bool | None = None,
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
                "  path: models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz",
                "streams:",
                stream_lines,
                *input_config,
                *inference,
                "output:",
                *(
                    [f"  video_enabled: {str(video_enabled).lower()}"]
                    if video_enabled is not None
                    else []
                ),
                "  insight:",
                "    host: 127.0.0.1",
                *(
                    [f"    video_port_base: {video_port_base}"]
                    if video_port_base is not None
                    else []
                ),
                *(
                    [f"    metadata_port_base: {metadata_port_base}"]
                    if metadata_port_base is not None
                    else []
                ),
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

        assert cfg.model_path == "models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz"
        assert len(cfg.rtsp_urls) == 4
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30
        assert cfg.tracker_max_missing == 30
        assert cfg.target_label == "drone"
        assert cfg.num_classes == 1
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 4

    def test_load_app_config_accepts_hevc(self, tmp_path: Path):
        cfg = main_module.load_app_config(
            write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], codec="hevc")
        )
        assert cfg.codec == "h265"

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

    @pytest.mark.parametrize(
        ("port_name", "other_port_name"),
        [
            ("video_port_base", "metadata_port_base"),
            ("metadata_port_base", "video_port_base"),
        ],
    )
    def test_load_app_config_accepts_last_port_at_udp_limit(
        self, tmp_path: Path, port_name: str, other_port_name: str
    ):
        port_values = {port_name: 65532, other_port_name: 9000}
        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 5)],
            **port_values,
        )

        cfg = main_module.load_app_config(config_path)

        assert getattr(cfg, port_name) + len(cfg.rtsp_urls) - 1 == 65535

    @pytest.mark.parametrize("port_name", ["video_port_base", "metadata_port_base"])
    def test_load_app_config_rejects_port_range_overflow(
        self, tmp_path: Path, port_name: str
    ):
        port_values = {port_name: 65533}
        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 5)],
            **port_values,
        )

        with pytest.raises(
            ValueError,
            match=rf"output\.insight\.{port_name} must be between 1 and 65532",
        ):
            main_module.load_app_config(config_path)

    @pytest.mark.parametrize(
        ("video_port_base", "metadata_port_base"),
        [(9000, 9001), (9001, 9000)],
    )
    def test_load_app_config_rejects_overlapping_insight_port_ranges(
        self, tmp_path: Path, video_port_base: int, metadata_port_base: int
    ):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
            video_port_base=video_port_base,
            metadata_port_base=metadata_port_base,
        )

        with pytest.raises(ValueError, match="port ranges must not overlap"):
            main_module.load_app_config(config_path)

    def test_load_app_config_allows_overlap_when_video_is_disabled(
        self, tmp_path: Path
    ):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
            video_port_base=9000,
            metadata_port_base=9000,
            video_enabled=False,
        )

        cfg = main_module.load_app_config(config_path)

        assert cfg.video_enabled is False

    def test_default_config_uses_one_class_motion_tracking(self):
        cfg = main_module.load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.model_path.endswith("yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz")
        assert cfg.num_classes == 1
        assert cfg.target_class_id == 0
        assert cfg.target_label == "drone"
        assert cfg.min_score == 0.05
        assert cfg.tracker_center_distance_enabled is True
        assert cfg.tracker_min_confirmed_hits == 2

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
                  path: models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz
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
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 0
        assert "streams=2" in result.stdout
        assert "max_inflight_per_stream=4" in result.stdout
        assert "max_inflight_total=4" in result.stdout


class TestRuntimeOptions:
    @pytest.mark.parametrize(
        ("tcp", "inherited_options", "expected_transport"),
        [
            (True, None, "rtsp_transport;tcp"),
            (False, "rtsp_transport;tcp", "rtsp_transport;udp"),
        ],
    )
    def test_probe_rtsp_uses_configured_transport_without_leaking_environment(
        self, monkeypatch, tcp, inherited_options, expected_transport
    ):
        observed_options = []

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return {1: 640, 2: 512, 3: 30}[prop]

            def release(self):
                pass

        if inherited_options is None:
            monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
        else:
            monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", inherited_options)

        def open_capture(_url):
            observed_options.append(
                main_module.os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            )
            return FakeCapture()

        monkeypatch.setattr(
            main_module,
            "cv2",
            SimpleNamespace(
                VideoCapture=open_capture,
                CAP_PROP_FRAME_WIDTH=1,
                CAP_PROP_FRAME_HEIGHT=2,
                CAP_PROP_FPS=3,
            ),
        )

        assert main_module.probe_rtsp("rtsp://camera/stream", tcp) == (640, 512, 30)
        assert observed_options == [expected_transport]
        assert main_module.os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == inherited_options

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

        link = main_module.realtime_link(2, 4, 3, 12)

        assert link.policy == "latest-by-stream"
        assert link.queue_depth == 4
        assert link.stream_id == "stream2"
        assert link.max_inflight_per_stream == 3
        assert link.max_inflight_total == 12

    def test_configured_fps_enables_videorate_without_changing_source(self):
        options = SimpleNamespace(
            source_fps=30,
            use_videorate=False,
            video_rate_fps=-1,
            output_caps=SimpleNamespace(fps=30),
        )

        output_fps = main_module.configure_output_fps(options, options.source_fps, 10)

        assert output_fps == 10
        assert options.source_fps == 30
        assert options.use_videorate is True
        assert options.video_rate_fps == 10
        assert options.output_caps.fps == 10

    def test_source_fps_default_does_not_insert_videorate(self):
        options = SimpleNamespace(
            source_fps=30,
            use_videorate=True,
            video_rate_fps=10,
            output_caps=SimpleNamespace(fps=10),
        )

        output_fps = main_module.configure_output_fps(options, options.source_fps, 0)

        assert output_fps == 30
        assert options.source_fps == 30
        assert options.use_videorate is False
        assert options.video_rate_fps == -1
        assert options.output_caps.fps == 30

    def test_none_pull_distinguishes_timeout_from_closed_output(self):
        timeout_run = SimpleNamespace(
            last_error=lambda: "", running=lambda: True, can_pull=lambda: False
        )
        closed_run = SimpleNamespace(
            last_error=lambda: "source reached EOS", running=lambda: False
        )

        assert main_module.pull_result_has_sample(timeout_run, None, "detections") is False
        with pytest.raises(
            RuntimeError,
            match="detections output closed unexpectedly: source reached EOS",
        ):
            main_module.pull_result_has_sample(closed_run, None, "detections")

    def test_debug_frame_matching_rejects_newer_unrelated_frame(self):
        matching_image = object()
        newer_image = object()
        stream = SimpleNamespace(
            debug_frames=deque(
                [
                    main_module.DebugFrame(frame_id=43, pts_ns=2_000_000, frame=newer_image),
                    main_module.DebugFrame(frame_id=42, pts_ns=1_000_000, frame=matching_image),
                ],
                maxlen=32,
            )
        )
        detection = SimpleNamespace(frame_id=42, pts_ns=1_000_000)

        assert main_module.take_debug_frame(stream, detection) is matching_image
        assert len(stream.debug_frames) == 1
        assert stream.debug_frames[0].frame is newer_image
        assert not main_module.samples_correlate(stream.debug_frames[0], detection)

    def test_debug_frame_matching_falls_back_to_pts(self):
        detection = SimpleNamespace(frame_id=-1, pts_ns=3_000_000)
        frame = SimpleNamespace(frame_id=-1, pts_ns=3_000_000)
        partially_identified_frame = SimpleNamespace(frame_id=42, pts_ns=3_000_000)

        assert main_module.samples_correlate(frame, detection)
        assert not main_module.samples_correlate(partially_identified_frame, detection)


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
        sender = FakeMetadataSender()
        runtime = main_module.StreamRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            source_options=None,
            metadata_sender=sender,
            tracker=main_module.ObjectTracker(),
            profile=main_module.ProfileWindow(False, 0),
            debug_frames=deque(maxlen=32),
            frame_w=100,
            frame_h=100,
            output_fps=30,
            video_port=9000,
        )
        tracks = [main_module.TrackedDetection(7, 10.0, 20.0, 40.0, 60.0, 0.75, 0)]

        cfg = main_module.AppConfig(model_path="model.tar.gz", rtsp_urls=[runtime.url])
        main_module.send_metadata(runtime, cfg, FakeSample(), tracks)

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "tracking"
        assert timestamp_ms == 1234
        assert frame_id == "42"
        assert json.loads(data_json) == {
            "tracks": [
                {
                    "id": "7",
                    "label": "drone",
                    "confidence": 0.75,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                }
            ]
        }


class TestTracker:
    def test_tracker_reuses_track_id_for_nearby_detection(self):
        tracker = main_module.ObjectTracker(
            main_module.TrackerConfig(match_iou_threshold=0.3, max_missing_frames=2)
        )
        first = tracker.update(
            [
                {
                    "x1": 10.0,
                    "y1": 10.0,
                    "x2": 50.0,
                    "y2": 80.0,
                    "score": 0.9,
                    "class_id": 0,
                }
            ],
            frame_index=0,
        )
        second = tracker.update(
            [
                {
                    "x1": 12.0,
                    "y1": 11.0,
                    "x2": 52.0,
                    "y2": 81.0,
                    "score": 0.8,
                    "class_id": 0,
                }
            ],
            frame_index=1,
        )

        assert len(first) == 1
        assert len(second) == 1
        assert first[0].track_id == second[0].track_id

    def test_tracker_drops_track_after_missing_budget(self):
        tracker = main_module.ObjectTracker(
            main_module.TrackerConfig(match_iou_threshold=0.3, max_missing_frames=1)
        )
        tracker.update(
            [
                {
                    "x1": 10.0,
                    "y1": 10.0,
                    "x2": 50.0,
                    "y2": 80.0,
                    "score": 0.9,
                    "class_id": 0,
                }
            ],
            frame_index=0,
        )
        tracker.update([], frame_index=1)
        tracker.update([], frame_index=2)

        assert tracker.active_track_count() == 0

    def test_motion_matching_reuses_id_when_boxes_have_zero_iou(self):
        tracker = main_module.ObjectTracker(
            main_module.TrackerConfig(
                match_iou_threshold=0.3,
                max_center_distance=3.0,
                velocity_momentum=0.0,
                center_distance_enabled=True,
            )
        )
        first = tracker.update(
            [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 20, "y1": 0, "x2": 30, "y2": 10, "score": 0.8, "class_id": 0}],
            frame_index=1,
        )
        third = tracker.update(
            [{"x1": 40, "y1": 0, "x2": 50, "y2": 10, "score": 0.8, "class_id": 0}],
            frame_index=2,
        )

        assert [item.track_id for item in (first[0], second[0], third[0])] == [1, 1, 1]

    def test_low_score_detection_only_recovers_confirmed_track(self):
        tracker = main_module.ObjectTracker(
            main_module.TrackerConfig(
                high_score_threshold=0.5,
                new_track_threshold=0.5,
                match_iou_threshold=0.1,
                min_confirmed_hits=2,
            )
        )
        high = {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9, "class_id": 0}
        low = {**high, "score": 0.2}

        assert tracker.update([high], 0) == []
        assert tracker.update([low], 1) == []
        confirmed = tracker.update([high], 2)
        recovered = tracker.update([low], 3)

        assert confirmed[0].track_id == 1
        assert recovered[0].track_id == 1

    def test_tracker_bounds_active_state(self):
        tracker = main_module.ObjectTracker(main_module.TrackerConfig(max_active_tracks=2))
        detections = [
            {
                "x1": index * 20,
                "y1": 0,
                "x2": index * 20 + 10,
                "y2": 10,
                "score": 0.9,
                "class_id": 0,
            }
            for index in range(5)
        ]

        assert len(tracker.update(detections, 0)) == 2
        assert tracker.active_track_count() == 2

    def test_tracker_expires_stale_state_before_creating_replacement(self):
        tracker = main_module.ObjectTracker(
            main_module.TrackerConfig(
                max_active_tracks=1,
                max_missing_frames=0,
                center_distance_enabled=False,
            )
        )
        first = tracker.update(
            [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        replacement = tracker.update(
            [{"x1": 100, "y1": 0, "x2": 110, "y2": 10, "score": 0.9, "class_id": 0}],
            frame_index=1,
        )

        assert len(replacement) == 1
        assert replacement[0].track_id != first[0].track_id
        assert tracker.active_track_count() == 1


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
        "max_inflight_total": 4,
        "num_classes": 1,
        "target_class_id": 0,
        "target_label": "drone",
        "min_score": 0.05,
        "nms_iou": 0.6,
        "max_detections": 100,
    },
    "runtime": {"profile": False, "warmup_frames": 30},
    # high_score_threshold and new_track_threshold are left to their defaults,
    # which follow min_score, so the min_score boundary cases stay valid.
    "tracking": {
        "match_iou_threshold": 0.05,
        "max_center_distance": 3.0,
        "velocity_momentum": 0.75,
        "max_missing_frames": 30,
        "min_confirmed_hits": 2,
        "max_active_tracks": 256,
    },
    "output": {
        "video_enabled": True,
        "save_every": 0,
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
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 4
        assert cfg.num_classes == 1
        assert cfg.target_class_id == 0
        assert cfg.target_label == "drone"
        assert cfg.min_score == pytest.approx(0.05)
        assert cfg.nms_iou == pytest.approx(0.60)
        assert cfg.max_detections == 100
        assert cfg.warmup_frames == 30
        assert cfg.tracker_iou_threshold == pytest.approx(0.05)
        assert cfg.tracker_high_score == pytest.approx(0.05)
        assert cfg.tracker_new_track_score == pytest.approx(0.05)
        assert cfg.tracker_max_center_distance == pytest.approx(3.0)
        assert cfg.tracker_velocity_momentum == pytest.approx(0.75)
        assert cfg.tracker_max_missing == 30
        assert cfg.tracker_min_confirmed_hits == 2
        assert cfg.tracker_max_active == 256
        assert cfg.tracker_center_distance_enabled == True
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100
        assert cfg.save_every == 0


REJECTED = [
    pytest.param(('model', 'path'), '', 'model.path must be set', id='model-path-empty'),
    pytest.param(('output', 'insight', 'host'), '', 'output.insight.host must be set', id='insight-host-empty'),
    pytest.param(('input', 'codec'), 'vp9', 'input.codec must be h264/avc or h265/hevc', id='codec-unsupported'),
    pytest.param(('input', 'latency_ms'), -1, 'input.latency_ms must be >= 0', id='latency-negative'),
    pytest.param(('inference', 'frames'), -1, 'inference.frames must be >= 0', id='frames-negative'),
    pytest.param(('inference', 'fps'), -1, 'inference.fps must be >= 0', id='fps-negative'),
    pytest.param(('inference', 'num_classes'), 0, 'inference.num_classes must be > 0', id='num-classes-zero'),
    pytest.param(('inference', 'target_class_id'), -1, 'inference.target_class_id must be >= 0', id='target-class-negative'),
    pytest.param(('inference', 'target_label'), '', 'inference.target_label must be set', id='target-label-empty'),
    pytest.param(('inference', 'min_score'), -0.01, 'inference.min_score must be between 0 and 1', id='min-score-below'),
    pytest.param(('inference', 'min_score'), 1.01, 'inference.min_score must be between 0 and 1', id='min-score-above'),
    pytest.param(('inference', 'nms_iou'), -0.01, 'inference.nms_iou must be between 0 and 1', id='nms-below'),
    pytest.param(('inference', 'nms_iou'), 1.01, 'inference.nms_iou must be between 0 and 1', id='nms-above'),
    pytest.param(('inference', 'max_detections'), 0, 'inference.max_detections must be > 0', id='max-detections-zero'),
    pytest.param(('runtime', 'warmup_frames'), -1, 'runtime.warmup_frames must be >= 0', id='warmup-negative'),
    pytest.param(('tracking', 'match_iou_threshold'), 1.01, 'tracking.match_iou_threshold must be between 0 and 1', id='match-iou-above'),
    pytest.param(('tracking', 'high_score_threshold'), 1.01, 'tracking.high_score_threshold must be in [inference.min_score, 1]', id='high-score-above-one'),
    pytest.param(('tracking', 'new_track_threshold'), 1.01, 'tracking.new_track_threshold must be in [high_score_threshold, 1]', id='new-track-above-one'),
    pytest.param(('tracking', 'max_center_distance'), -1, 'tracking.max_center_distance must be >= 0', id='center-distance-negative'),
    pytest.param(('tracking', 'velocity_momentum'), 1.0, 'tracking.velocity_momentum must be in [0, 1)', id='momentum-at-one'),
    pytest.param(('tracking', 'max_missing_frames'), -1, 'tracking.max_missing_frames must be >= 0', id='max-missing-negative'),
    pytest.param(('tracking', 'min_confirmed_hits'), 0, 'tracking.min_confirmed_hits must be >= 1', id='min-hits-zero'),
    pytest.param(('tracking', 'max_active_tracks'), 0, 'tracking.max_active_tracks must be >= 1', id='max-active-zero'),
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
            (('tracking', 'match_iou_threshold'), 0.0),
            (('tracking', 'match_iou_threshold'), 1.0),
            (('tracking', 'velocity_momentum'), 0.0),
            (('tracking', 'velocity_momentum'), 0.999),
            (('tracking', 'max_center_distance'), 0.0),
            (('tracking', 'max_missing_frames'), 0),
            (('tracking', 'min_confirmed_hits'), 1),
            (('tracking', 'max_active_tracks'), 1),
            (('input', 'latency_ms'), 0),
            (('inference', 'frames'), 0),
            (('inference', 'fps'), 0),
            (('runtime', 'warmup_frames'), 0),
            (('output', 'save_every'), 0),
            (('inference', 'max_detections'), 1),
            (('inference', 'num_classes'), 1),
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


class TestTrackerThresholdsRelateToEachOther:
    """Three rules compare two keys, so a single-key table cannot reach them."""

    def test_a_high_score_below_min_score_is_rejected(self, tmp_path):
        overrides = [(("inference", "min_score"), 0.5), (("tracking", "high_score_threshold"), 0.4)]

        with pytest.raises(ValueError, match=r"high_score_threshold must be in \[inference.min_score, 1\]"):
            main_module.load_app_config(write_full_config(tmp_path, overrides))

    def test_a_new_track_threshold_below_high_score_is_rejected(self, tmp_path):
        overrides = [
            (("tracking", "high_score_threshold"), 0.6),
            (("tracking", "new_track_threshold"), 0.5),
        ]

        with pytest.raises(ValueError, match=r"new_track_threshold must be in \[high_score_threshold, 1\]"):
            main_module.load_app_config(write_full_config(tmp_path, overrides))

    def test_a_target_class_outside_num_classes_is_rejected(self, tmp_path):
        overrides = [(("inference", "num_classes"), 2), (("inference", "target_class_id"), 2)]

        with pytest.raises(ValueError, match="must be less than inference.num_classes"):
            main_module.load_app_config(write_full_config(tmp_path, overrides))

    def test_the_last_class_index_is_accepted(self, tmp_path):
        overrides = [(("inference", "num_classes"), 3), (("inference", "target_class_id"), 2)]

        cfg = main_module.load_app_config(write_full_config(tmp_path, overrides))

        assert cfg.target_class_id == 2

    def test_thresholds_default_to_min_score_in_order(self, tmp_path):
        """high_score follows min_score and new_track follows high_score, so a
        config that sets only min_score stays consistent by construction."""
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("inference", "min_score"), 0.4)]))

        assert cfg.tracker_high_score == pytest.approx(0.4)
        assert cfg.tracker_new_track_score == pytest.approx(0.4)
