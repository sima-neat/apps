"""Unit tests for the YOLO26 tiny-drone tracker Insight example."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

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

        with pytest.raises(
            ValueError, match="max_inflight_per_stream must be -1 or > 0"
        ):
            load_app_config(config_path)

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
        from main import load_app_config

        port_values = {port_name: 65532, other_port_name: 9000}
        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 5)],
            **port_values,
        )

        cfg = load_app_config(config_path)

        assert getattr(cfg, port_name) + len(cfg.rtsp_urls) - 1 == 65535

    @pytest.mark.parametrize("port_name", ["video_port_base", "metadata_port_base"])
    def test_load_app_config_rejects_port_range_overflow(
        self, tmp_path: Path, port_name: str
    ):
        from main import load_app_config

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
            load_app_config(config_path)

    @pytest.mark.parametrize(
        ("video_port_base", "metadata_port_base"),
        [(9000, 9001), (9001, 9000)],
    )
    def test_load_app_config_rejects_overlapping_insight_port_ranges(
        self, tmp_path: Path, video_port_base: int, metadata_port_base: int
    ):
        from main import load_app_config

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
            load_app_config(config_path)

    def test_load_app_config_allows_overlap_when_video_is_disabled(
        self, tmp_path: Path
    ):
        from main import load_app_config

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

        cfg = load_app_config(config_path)

        assert cfg.video_enabled is False

    def test_default_config_uses_one_class_motion_tracking(self):
        from main import load_app_config

        cfg = load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.model_path.endswith("yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz")
        assert cfg.num_classes == 1
        assert cfg.target_class_id == 0
        assert cfg.target_label == "drone"
        assert cfg.min_score == 0.05
        assert cfg.tracker_center_distance_enabled is True
        assert cfg.tracker_min_confirmed_hits == 2

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
        import main

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
                main.os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            )
            return FakeCapture()

        monkeypatch.setattr(
            main,
            "cv2",
            SimpleNamespace(
                VideoCapture=open_capture,
                CAP_PROP_FRAME_WIDTH=1,
                CAP_PROP_FRAME_HEIGHT=2,
                CAP_PROP_FPS=3,
            ),
        )

        assert main.probe_rtsp("rtsp://camera/stream", tcp) == (640, 512, 30)
        assert observed_options == [expected_transport]
        assert main.os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") == inherited_options

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

    def test_configured_fps_enables_videorate_without_changing_source(self):
        from main import configure_output_fps

        options = SimpleNamespace(
            source_fps=30,
            use_videorate=False,
            video_rate_fps=-1,
            output_caps=SimpleNamespace(fps=30),
        )

        output_fps = configure_output_fps(options, options.source_fps, 10)

        assert output_fps == 10
        assert options.source_fps == 30
        assert options.use_videorate is True
        assert options.video_rate_fps == 10
        assert options.output_caps.fps == 10

    def test_source_fps_default_does_not_insert_videorate(self):
        from main import configure_output_fps

        options = SimpleNamespace(
            source_fps=30,
            use_videorate=True,
            video_rate_fps=10,
            output_caps=SimpleNamespace(fps=10),
        )

        output_fps = configure_output_fps(options, options.source_fps, 0)

        assert output_fps == 30
        assert options.source_fps == 30
        assert options.use_videorate is False
        assert options.video_rate_fps == -1
        assert options.output_caps.fps == 30

    def test_none_pull_distinguishes_timeout_from_closed_output(self):
        from main import pull_result_has_sample

        timeout_run = SimpleNamespace(
            last_error=lambda: "", running=lambda: True, can_pull=lambda: False
        )
        closed_run = SimpleNamespace(
            last_error=lambda: "source reached EOS", running=lambda: False
        )

        assert pull_result_has_sample(timeout_run, None, "detections") is False
        with pytest.raises(
            RuntimeError,
            match="detections output closed unexpectedly: source reached EOS",
        ):
            pull_result_has_sample(closed_run, None, "detections")

    def test_debug_frame_matching_rejects_newer_unrelated_frame(self):
        from main import DebugFrame, samples_correlate, take_debug_frame

        matching_image = object()
        newer_image = object()
        stream = SimpleNamespace(
            debug_frames=deque(
                [
                    DebugFrame(frame_id=43, pts_ns=2_000_000, frame=newer_image),
                    DebugFrame(frame_id=42, pts_ns=1_000_000, frame=matching_image),
                ],
                maxlen=32,
            )
        )
        detection = SimpleNamespace(frame_id=42, pts_ns=1_000_000)

        assert take_debug_frame(stream, detection) is matching_image
        assert len(stream.debug_frames) == 1
        assert stream.debug_frames[0].frame is newer_image
        assert not samples_correlate(stream.debug_frames[0], detection)

    def test_debug_frame_matching_falls_back_to_pts(self):
        from main import samples_correlate

        detection = SimpleNamespace(frame_id=-1, pts_ns=3_000_000)
        frame = SimpleNamespace(frame_id=-1, pts_ns=3_000_000)
        partially_identified_frame = SimpleNamespace(frame_id=42, pts_ns=3_000_000)

        assert samples_correlate(frame, detection)
        assert not samples_correlate(partially_identified_frame, detection)


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
        from main import AppConfig, ProfileWindow, StreamRuntime, send_metadata
        from utils.tracker import ObjectTracker, TrackedDetection

        sender = FakeMetadataSender()
        runtime = StreamRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            source_options=None,
            metadata_sender=sender,
            tracker=ObjectTracker(),
            profile=ProfileWindow(False, 0),
            debug_frames=deque(maxlen=32),
            frame_w=100,
            frame_h=100,
            output_fps=30,
            video_port=9000,
        )
        tracks = [TrackedDetection(7, 10.0, 20.0, 40.0, 60.0, 0.75, 0)]

        cfg = AppConfig(model_path="model.tar.gz", rtsp_urls=[runtime.url])
        send_metadata(runtime, cfg, FakeSample(), tracks)

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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(match_iou_threshold=0.3, max_missing_frames=2)
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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(match_iou_threshold=0.3, max_missing_frames=1)
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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(max_active_tracks=2))
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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
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
