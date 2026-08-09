"""Unit tests for the multistream people tracker Insight example."""

from __future__ import annotations

import importlib.util
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
ACCURACY_PATH = PYTHON_DIR / "evaluate_tracking.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit

ACCURACY_SPEC = importlib.util.spec_from_file_location(
    "evaluate_tracking", ACCURACY_PATH
)
assert ACCURACY_SPEC and ACCURACY_SPEC.loader
evaluate_tracking = importlib.util.module_from_spec(ACCURACY_SPEC)
ACCURACY_SPEC.loader.exec_module(evaluate_tracking)


def write_config(
    tmp_path: Path,
    streams: list[str],
    codec: str | None = None,
    max_inflight_per_stream: int | None = None,
    max_inflight_total: int | None = None,
    num_classes: int | None = None,
    target_class_id: int | None = None,
    overflow_policy: str | None = None,
) -> Path:
    stream_lines = "\n".join(f"  - {stream}" for stream in streams)
    inference = []
    input_config = ["input:", f"  codec: {codec}"] if codec else []
    runtime_config = (
        ["runtime:", f"  overflow_policy: {overflow_policy}"] if overflow_policy else []
    )
    if any(
        value is not None
        for value in (
            max_inflight_per_stream,
            max_inflight_total,
            num_classes,
            target_class_id,
        )
    ):
        inference.append("inference:")
        if max_inflight_per_stream is not None:
            inference.append(f"  max_inflight_per_stream: {max_inflight_per_stream}")
        if max_inflight_total is not None:
            inference.append(f"  max_inflight_total: {max_inflight_total}")
        if num_classes is not None:
            inference.append(f"  num_classes: {num_classes}")
        if target_class_id is not None:
            inference.append(f"  target_class_id: {target_class_id}")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  path: models/yolo26m-det-int8-b1.tar.gz",
                "streams:",
                stream_lines,
                *input_config,
                *runtime_config,
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
        assert cfg.tracker_box_smoothing_alpha == 1.0
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16
        assert cfg.overflow_policy == "keep_latest"
        assert cfg.num_classes == 80
        assert cfg.min_score == 0.55
        assert cfg.tracker_high_score == 0.55
        assert cfg.tracker_new_track_score == 0.55
        assert cfg.tracker_iou_threshold == 0.30
        assert cfg.tracker_center_distance_enabled is False
        assert cfg.tracker_camera_motion_compensation is False

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

    def test_load_app_config_accepts_block_overflow_policy(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(
            write_config(
                tmp_path,
                ["rtsp://127.0.0.1:8554/src1"],
                overflow_policy="block",
            )
        )

        assert cfg.overflow_policy == "block"

    def test_load_app_config_rejects_invalid_overflow_policy(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            overflow_policy="drop_oldest",
        )

        with pytest.raises(
            ValueError,
            match="runtime.overflow_policy must be keep_latest or block",
        ):
            load_app_config(config_path)

    def test_load_app_config_rejects_block_with_shared_detector_fan_in(
        self, tmp_path: Path
    ):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
            overflow_policy="block",
        )

        with pytest.raises(
            ValueError,
            match="runtime.overflow_policy=block requires exactly one stream",
        ):
            load_app_config(config_path)

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

    def test_load_app_config_rejects_non_positive_class_count(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            num_classes=0,
        )

        with pytest.raises(ValueError, match="inference.num_classes must be > 0"):
            load_app_config(config_path)

    def test_load_app_config_rejects_target_outside_class_count(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            num_classes=1,
            target_class_id=1,
        )

        with pytest.raises(
            ValueError,
            match=r"inference.target_class_id \(1\) must be less than inference.num_classes \(1\)",
        ):
            load_app_config(config_path)

    def test_default_config_uses_tracking_threshold(self):
        from main import load_app_config

        cfg = load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.min_score == 0.30
        assert cfg.target_class_id == 0
        assert cfg.num_classes == 80
        assert cfg.target_label == "person"
        assert cfg.tracker_iou_threshold == 0.10
        assert cfg.tracker_center_distance_enabled is True

    def test_tiny_drone_config_uses_one_class_contract(self):
        from main import load_app_config

        cfg = load_app_config(EXAMPLE_DIR / "src" / "common" / "tiny-drone.yaml")

        assert cfg.num_classes == 1
        assert cfg.target_class_id == 0
        assert cfg.tracker_max_center_distance == 0.50
        assert cfg.tracker_velocity_momentum == 0.90
        assert cfg.tracker_box_smoothing_alpha == 0.50
        assert cfg.tracker_camera_motion_compensation is True
        assert cfg.tracker_max_prediction_frames == 1

    def test_legacy_iou_config_keeps_iou_only_matching(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: models/yolo26m-det-int8-b1.tar.gz
                streams:
                  - rtsp://127.0.0.1:8554/src1
                tracking:
                  iou_threshold: 0.50
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.tracker_iou_threshold == 0.50
        assert cfg.tracker_center_distance_enabled is False

    def test_omitted_tracking_thresholds_follow_decoder_floor(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: models/yolo26m-det-int8-b1.tar.gz
                streams:
                  - rtsp://127.0.0.1:8554/src1
                inference:
                  min_score: 0.70
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.tracker_high_score == 0.70
        assert cfg.tracker_new_track_score == 0.70

    def test_omitted_new_track_threshold_follows_high_threshold(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: models/yolo26m-det-int8-b1.tar.gz
                streams:
                  - rtsp://127.0.0.1:8554/src1
                tracking:
                  high_score_threshold: 0.75
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.tracker_high_score == 0.75
        assert cfg.tracker_new_track_score == 0.75

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
        assert "max_inflight_total=16" in result.stdout
        assert "overflow_policy=keep_latest" in result.stdout


class TestRuntimeOptions:
    @pytest.mark.parametrize(
        ("configured", "expected_link"),
        [("keep_latest", "keep-latest"), ("block", "default")],
    )
    def test_realtime_run_and_configured_link_options(
        self, monkeypatch, configured, expected_link
    ):
        import main

        class FakeRunOptions:
            def __init__(self):
                self.overflow_policy = "auto"

        fake_pyneat = SimpleNamespace(
            RunOptions=FakeRunOptions,
            RunPreset=SimpleNamespace(Realtime="realtime"),
            OverflowPolicy=SimpleNamespace(KeepLatest="keep-latest", Block="block"),
            OutputMemory=SimpleNamespace(ZeroCopy="zero-copy"),
            GraphLinkOptions=type("GraphLinkOptions", (), {}),
            GraphLinkPolicy=SimpleNamespace(
                Default="default", RealtimeLatestByStream="keep-latest"
            ),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)
        cfg = main.AppConfig(
            model_path="model.tar.gz",
            rtsp_urls=["rtsp://source"],
            overflow_policy=configured,
        )

        run_options = main.build_run_options()
        link_options = main.stream_link(cfg, 0, 4, 3, 12)

        assert run_options.preset == "realtime"
        assert run_options.overflow_policy == "auto"
        assert link_options.policy == expected_link
        assert link_options.stream_id == "stream0"

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

        cfg = main.AppConfig(model_path="model.tar.gz", rtsp_urls=["rtsp://source"])
        link = main.stream_link(cfg, 2, 4, 3, 12)

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
            debug_frames=deque(),
            frame_w=100,
            frame_h=100,
            output_fps=30,
            video_port=9000,
        )
        tracks = [TrackedDetection(7, 10.0, 20.0, 40.0, 60.0, 0.75, 0)]

        cfg = AppConfig(
            model_path="models/yolo26n-p2-tiny-drone-int8-qat-b1.tar.gz",
            rtsp_urls=[runtime.url],
            target_label="drone",
        )
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


class TestDebugFrameSynchronization:
    def test_pts_matches_across_different_segment_frame_ids(self):
        from main import DebugFrame, samples_identify_same_frame

        sample = SimpleNamespace(
            pts_ns=123_000,
            orig_input_seq=-1,
            input_seq=-1,
            frame_id=17,
        )
        frame = DebugFrame(3, -1, -1, 123_000, object())

        assert samples_identify_same_frame(sample, frame)

    def test_mismatched_pts_never_falls_back_to_equal_frame_id(self):
        from main import DebugFrame, samples_identify_same_frame

        sample = SimpleNamespace(
            pts_ns=123_000,
            orig_input_seq=-1,
            input_seq=-1,
            frame_id=17,
        )
        frame = DebugFrame(17, -1, -1, 124_000, object())

        assert not samples_identify_same_frame(sample, frame)

    def test_waits_for_a_lagging_debug_branch(self, monkeypatch):
        import main

        sample = SimpleNamespace(
            pts_ns=123_000,
            orig_input_seq=-1,
            input_seq=-1,
            frame_id=17,
        )
        stream = SimpleNamespace(index=0, debug_frames=deque())
        app = SimpleNamespace(streams=[stream])
        cfg = main.AppConfig(
            model_path="model.tar.gz",
            rtsp_urls=["rtsp://127.0.0.1/src"],
            save_dir="debug",
            save_every=1,
        )
        timeouts = []

        def pull_lagging_frame(_app, target_stream, timeout_ms):
            timeouts.append(timeout_ms)
            target_stream.debug_frames.append(
                main.DebugFrame(3, -1, -1, 123_000, object())
            )
            return True

        monkeypatch.setattr(main, "pull_debug_frame", pull_lagging_frame)
        main.await_matching_debug_frame(app, cfg, 0, sample)

        assert timeouts and timeouts[0] > 0
        assert main.samples_identify_same_frame(sample, stream.debug_frames[0])


class TestTracker:
    def test_tracker_reuses_track_id_for_nearby_detection(self):
        from utils.tracker import ObjectTracker

        tracker = ObjectTracker()
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

    def test_motion_compensated_box_smoothing_reduces_jitter(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(box_smoothing_alpha=0.5, velocity_momentum=0.9)
        )
        tracker.update(
            [{"x1": 0, "y1": 0, "x2": 4, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        smoothed = tracker.update(
            [{"x1": 2, "y1": 0, "x2": 6, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=1,
        )

        assert smoothed[0].x1 == pytest.approx(1.0)
        assert smoothed[0].x2 == pytest.approx(5.0)

    def test_camera_motion_compensation_preserves_ids_across_fast_pan(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
                max_center_distance=0.5,
                box_smoothing_alpha=0.5,
                camera_motion_compensation=True,
            )
        )
        first_detections = [
            {
                "x1": 40 * index,
                "y1": 20,
                "x2": 40 * index + 4,
                "y2": 24,
                "score": 0.9,
                "class_id": 0,
            }
            for index in range(6)
        ]
        panned_detections = [
            {**detection, "x1": detection["x1"] + 20, "x2": detection["x2"] + 20}
            for detection in first_detections
        ]

        first = tracker.update(first_detections, frame_index=0)
        panned = tracker.update(panned_detections, frame_index=1)

        assert [track.track_id for track in panned] == [
            track.track_id for track in first
        ]
        assert [track.x1 for track in panned] == pytest.approx(
            [detection["x1"] for detection in panned_detections]
        )

    def test_camera_motion_compensation_does_not_bridge_scene_cut(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(max_center_distance=0.5, camera_motion_compensation=True)
        )
        first_detections = [
            {
                "x1": 20 * index,
                "y1": 20,
                "x2": 20 * index + 4,
                "y2": 24,
                "score": 0.9,
                "class_id": 0,
            }
            for index in range(8)
        ]
        cut_detections = [
            {
                **detection,
                "x1": detection["x1"] + 100,
                "x2": detection["x2"] + 100,
                "y1": 100,
                "y2": 104,
            }
            for detection in first_detections
        ]

        first = tracker.update(first_detections, frame_index=0)
        after_cut = tracker.update(cut_detections, frame_index=1)

        assert after_cut[0].track_id != first[0].track_id
        assert tracker.active_track_count() == 16

    def test_external_camera_transform_is_not_learned_as_object_velocity(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
                max_center_distance=0.5,
                velocity_momentum=0.9,
                camera_motion_compensation=True,
            )
        )
        first = tracker.update(
            [{"x1": 0, "y1": 20, "x2": 4, "y2": 24, "score": 0.9, "class_id": 0}],
            0,
        )
        track_id = first[0].track_id
        pan = (1.0, 0.0, 20.0, 0.0, 1.0, 0.0)
        for frame in range(1, 13):
            x = 20 * frame
            tracked = tracker.update(
                [{"x1": x, "y1": 20, "x2": x + 4, "y2": 24, "score": 0.9, "class_id": 0}],
                frame,
                pan,
            )
            assert tracked[0].track_id == track_id
            assert tracked[0].x1 == pytest.approx(x)

    def test_orb_camera_motion_estimator_recovers_translation(self):
        from utils.tracker import FrameCameraMotionEstimator

        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        first = np.random.default_rng(12345).integers(
            0, 256, size=(256, 320), dtype=np.uint8
        )
        first = cv2.GaussianBlur(first, (3, 3), 0.8)
        second = cv2.warpAffine(
            first,
            np.float32([[1, 0, 18], [0, 1, 7]]),
            (first.shape[1], first.shape[0]),
        )
        estimator = FrameCameraMotionEstimator()

        assert estimator.update(first) is None
        motion = estimator.update(second)
        assert motion is not None
        assert motion[2] == pytest.approx(18, abs=2)
        assert motion[5] == pytest.approx(7, abs=2)

    def test_recent_track_wins_before_stale_track(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(max_center_distance=2.0, velocity_momentum=0.9)
        )
        first = tracker.update(
            [
                {"x1": 0, "y1": 0, "x2": 4, "y2": 4, "score": 0.9, "class_id": 0},
                {"x1": 10, "y1": 0, "x2": 14, "y2": 4, "score": 0.9, "class_id": 0},
            ],
            frame_index=0,
        )
        tracker.update(
            [{"x1": 6, "y1": 0, "x2": 10, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=1,
        )
        recent = tracker.update(
            [{"x1": 2, "y1": 0, "x2": 6, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=2,
        )

        assert recent[0].track_id == first[1].track_id

    def test_tracker_drops_track_after_missing_budget(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(max_missing_frames=1))
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

    def test_zero_missing_budget_keeps_continuous_track(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(max_missing_frames=0))
        first = tracker.update(
            [{"x1": 10, "y1": 10, "x2": 14, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        continuous = tracker.update(
            [{"x1": 11, "y1": 10, "x2": 15, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=1,
        )

        assert first[0].track_id == continuous[0].track_id

    def test_tracker_recovers_after_exact_missing_budget(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(max_missing_frames=1))
        first = tracker.update(
            [{"x1": 10, "y1": 10, "x2": 14, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        tracker.update([], frame_index=1)
        recovered = tracker.update(
            [{"x1": 11, "y1": 10, "x2": 15, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=2,
        )

        assert first[0].track_id == recovered[0].track_id

    def test_tracker_matches_tiny_boxes_after_zero_iou_shift(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(match_iou_threshold=0.5, max_center_distance=2.0)
        )
        first = tracker.update(
            [{"x1": 10, "y1": 10, "x2": 12, "y2": 12, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 13, "y1": 10, "x2": 15, "y2": 12, "score": 0.8, "class_id": 0}],
            frame_index=1,
        )

        assert len(first) == len(second) == 1
        assert first[0].track_id == second[0].track_id

    def test_tracker_can_disable_center_distance_matching(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
                match_iou_threshold=0.5,
                max_center_distance=2.5,
                center_distance_enabled=False,
            )
        )
        first = tracker.update(
            [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        below_iou = tracker.update(
            [{"x1": 5, "y1": 0, "x2": 15, "y2": 10, "score": 0.9, "class_id": 0}],
            frame_index=1,
        )

        assert first[0].track_id != below_iou[0].track_id

    def test_iou_only_tracker_does_not_apply_motion_prediction(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
                match_iou_threshold=0.3,
                velocity_momentum=0.0,
                center_distance_enabled=False,
            )
        )
        track_ids = []
        for frame_index, x1 in enumerate((0, 5, 10, 15, 20, 15)):
            tracked = tracker.update(
                [
                    {
                        "x1": x1,
                        "y1": 0,
                        "x2": x1 + 10,
                        "y2": 10,
                        "score": 0.9,
                        "class_id": 0,
                    }
                ],
                frame_index=frame_index,
            )
            track_ids.append(tracked[0].track_id)

        assert track_ids == [track_ids[0]] * len(track_ids)

    def test_low_score_detection_recovers_but_cannot_create_track(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        config = TrackerConfig(high_score_threshold=0.5, new_track_threshold=0.7)
        established = ObjectTracker(config)
        first = established.update(
            [{"x1": 10, "y1": 10, "x2": 14, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        recovered = established.update(
            [{"x1": 11, "y1": 10, "x2": 15, "y2": 14, "score": 0.2, "class_id": 0}],
            frame_index=1,
        )

        fresh = ObjectTracker(config)
        low_only = fresh.update(
            [{"x1": 10, "y1": 10, "x2": 14, "y2": 14, "score": 0.2, "class_id": 0}],
            frame_index=0,
        )

        assert len(first) == len(recovered) == 1
        assert first[0].track_id == recovered[0].track_id
        assert low_only == []
        assert fresh.active_track_count() == 0

    def test_confirmation_suppresses_single_frame_noise(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(min_confirmed_hits=2))
        first = tracker.update(
            [{"x1": 10, "y1": 10, "x2": 14, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        second = tracker.update(
            [{"x1": 10.5, "y1": 10, "x2": 14.5, "y2": 14, "score": 0.8, "class_id": 0}],
            frame_index=1,
        )

        assert first == []
        assert len(second) == 1

    def test_global_assignment_avoids_greedy_identity_loss(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(match_iou_threshold=0.10, center_distance_enabled=False)
        )
        first = tracker.update(
            [
                {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9, "class_id": 0},
                {"x1": 8, "y1": 0, "x2": 18, "y2": 10, "score": 0.9, "class_id": 0},
            ],
            frame_index=0,
        )
        second = tracker.update(
            [
                {"x1": 1, "y1": 0, "x2": 11, "y2": 10, "score": 0.9, "class_id": 0},
                {"x1": -3, "y1": 0, "x2": 7, "y2": 10, "score": 0.9, "class_id": 0},
            ],
            frame_index=1,
        )

        assert [track.track_id for track in second] == [
            first[1].track_id,
            first[0].track_id,
        ]

    def test_prediction_bridges_one_high_confidence_gap(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(
                high_score_threshold=0.5,
                new_track_threshold=0.5,
                velocity_momentum=0.0,
                max_missing_frames=3,
                max_prediction_frames=1,
            )
        )
        tracker.update(
            [{"x1": 0, "y1": 0, "x2": 4, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        observed = tracker.update(
            [{"x1": 1, "y1": 0, "x2": 5, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=1,
        )
        bridged = tracker.update([], frame_index=2)
        beyond_horizon = tracker.update([], frame_index=3)

        assert bridged[0].track_id == observed[0].track_id
        assert bridged[0].predicted is True
        assert bridged[0].x1 == pytest.approx(2.0)
        assert beyond_horizon == []

    def test_unconfirmed_track_expires_on_first_miss(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(
            TrackerConfig(min_confirmed_hits=2, max_missing_frames=30)
        )
        tracker.update(
            [{"x1": 0, "y1": 0, "x2": 4, "y2": 4, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        tracker.update([], frame_index=1)

        assert tracker.active_track_count() == 0

    def test_tracker_does_not_revive_after_missing_budget(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(max_missing_frames=1))
        first = tracker.update(
            [{"x1": 10, "y1": 10, "x2": 14, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=0,
        )
        replacement = tracker.update(
            [{"x1": 11, "y1": 10, "x2": 15, "y2": 14, "score": 0.9, "class_id": 0}],
            frame_index=3,
        )

        assert first[0].track_id != replacement[0].track_id

    def test_tracker_enforces_monotonic_frames_without_active_tracks(self):
        from utils.tracker import ObjectTracker

        tracker = ObjectTracker()
        tracker.update([], frame_index=5)
        with pytest.raises(ValueError, match="monotonic"):
            tracker.update([], frame_index=4)

    def test_tracker_rejects_non_finite_motion_gate(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        with pytest.raises(ValueError, match="max_center_distance"):
            ObjectTracker(TrackerConfig(max_center_distance=float("nan")))

    def test_tracker_rejects_excessive_prediction_horizon(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        with pytest.raises(ValueError, match="max_prediction_frames"):
            ObjectTracker(TrackerConfig(max_missing_frames=1, max_prediction_frames=2))

    def test_tracker_rejects_invalid_box_smoothing(self):
        from utils.tracker import ObjectTracker, TrackerConfig

        with pytest.raises(ValueError, match="box_smoothing_alpha"):
            ObjectTracker(TrackerConfig(box_smoothing_alpha=0.0))


class TestAccuracyEvaluation:
    @pytest.mark.parametrize("fps", [float("nan"), float("inf")])
    def test_nonfinite_fps_is_rejected(self, fps: float):
        truth = {
            0: {
                "frame_index": 0,
                "width": 640,
                "height": 640,
                "objects": [],
            }
        }

        with pytest.raises(ValueError, match="fps must be finite and positive"):
            evaluate_tracking.evaluate(
                truth, {}, iou_threshold=0.3, fps=fps, model_size=640
            )

    def test_cli_rejects_nan_false_positive_limit(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(ACCURACY_PATH),
                "--ground-truth",
                "missing-truth.jsonl",
                "--predictions",
                "missing-predictions.jsonl",
                "--output",
                "unused.json",
                "--fps",
                "30",
                "--maximum-false-positives-per-minute",
                "nan",
            ],
        )

        with pytest.raises(SystemExit) as error:
            evaluate_tracking.main()

        assert error.value.code == 2
        assert (
            "maximum-false-positives-per-minute must be >= 0" in capsys.readouterr().err
        )

    def test_metrics_count_recall_false_positives_and_id_switches(self):
        truth = {
            0: {
                "frame_index": 0,
                "width": 1920,
                "height": 1080,
                "objects": [{"track_id": "d1", "bbox": [10, 10, 12, 8]}],
            },
            1: {
                "frame_index": 1,
                "width": 1920,
                "height": 1080,
                "objects": [{"track_id": "d1", "bbox": [12, 10, 12, 8]}],
            },
            2: {
                "frame_index": 2,
                "width": 1920,
                "height": 1080,
                "objects": [{"track_id": "d1", "bbox": [14, 10, 12, 8]}],
            },
        }
        predictions = {
            0: {"frame_index": 0, "tracks": [{"id": "1", "bbox": [10, 10, 12, 8]}]},
            1: {
                "frame_index": 1,
                "tracks": [{"id": "noise", "bbox": [100, 100, 10, 10]}],
            },
            2: {"frame_index": 2, "tracks": [{"id": "2", "bbox": [14, 10, 12, 8]}]},
        }

        report = evaluate_tracking.evaluate(
            truth, predictions, iou_threshold=0.3, fps=30.0, model_size=640
        )

        assert report["detection"]["true_positives"] == 2
        assert report["detection"]["false_positives"] == 1
        assert report["detection"]["false_negatives"] == 1
        assert report["detection"]["recall"] == pytest.approx(2 / 3)
        assert report["tracking"]["id_switches"] == 1
        assert report["tracking"]["fragmentations"] == 1
        assert report["tracking"]["available"] is True
        assert (
            report["detection"]["recall_by_model_input_size"]["tiny"]["ground_truth"]
            == 3
        )

    def test_missing_ground_truth_ids_make_tracking_metrics_unavailable(self):
        truth = {
            0: {
                "frame_index": 0,
                "width": 640,
                "height": 640,
                "objects": [{"bbox": [10, 10, 8, 8]}],
            }
        }
        predictions = {
            0: {"frame_index": 0, "tracks": [{"id": "1", "bbox": [10, 10, 8, 8]}]}
        }

        report = evaluate_tracking.evaluate(
            truth, predictions, iou_threshold=0.3, fps=30.0, model_size=640
        )

        assert report["detection"]["recall"] == 1.0
        assert report["tracking"] == {
            "available": False,
            "unavailable_reason": "ground-truth objects require non-empty track_id values",
            "ground_truth_track_count": None,
            "id_switches": None,
            "fragmentations": None,
        }

    def test_tracking_gate_fails_when_ground_truth_ids_are_missing(self):
        report = {
            "frames": 1,
            "detection": {
                "recall": 1.0,
                "false_positives_per_minute": 0.0,
                "recall_by_model_input_size": {"tiny": {"recall": 1.0}},
            },
            "tracking": {
                "available": False,
                "unavailable_reason": "ground-truth objects require non-empty track_id values",
                "ground_truth_track_count": None,
                "id_switches": None,
                "fragmentations": None,
            },
        }
        args = SimpleNamespace(
            minimum_frames=1,
            minimum_recall=0.0,
            minimum_tiny_recall=0.0,
            maximum_false_positives_per_minute=float("inf"),
            maximum_id_switches=None,
            maximum_fragmentations=None,
        )

        assert evaluate_tracking.enforce_gates(report, args) == []

        args.maximum_id_switches = 0
        failures = evaluate_tracking.enforce_gates(report, args)

        assert failures == [
            "tracking metrics unavailable: "
            "ground-truth objects require non-empty track_id values"
        ]

    @pytest.mark.parametrize(
        ("truth_ids", "prediction_ids", "reason"),
        [
            (
                ["same", "same"],
                ["1", "2"],
                "ground-truth objects require unique track_id values within each frame",
            ),
            (
                ["a", "b"],
                ["same", "same"],
                "predicted tracks require unique id values within each frame",
            ),
        ],
    )
    def test_duplicate_per_frame_ids_make_tracking_metrics_unavailable(
        self, truth_ids: list[str], prediction_ids: list[str], reason: str
    ):
        truth = {
            0: {
                "frame_index": 0,
                "width": 640,
                "height": 640,
                "objects": [
                    {"track_id": truth_ids[0], "bbox": [0, 0, 10, 10]},
                    {"track_id": truth_ids[1], "bbox": [100, 100, 10, 10]},
                ],
            }
        }
        predictions = {
            0: {
                "frame_index": 0,
                "tracks": [
                    {"id": prediction_ids[0], "bbox": [0, 0, 10, 10]},
                    {"id": prediction_ids[1], "bbox": [100, 100, 10, 10]},
                ],
            }
        }

        report = evaluate_tracking.evaluate(
            truth, predictions, iou_threshold=0.3, fps=30.0, model_size=640
        )

        assert report["detection"]["true_positives"] == 2
        assert report["tracking"]["available"] is False
        assert report["tracking"]["unavailable_reason"] == reason
        assert report["tracking"]["id_switches"] is None
        assert report["tracking"]["fragmentations"] is None

    def test_matching_maximizes_cardinality_before_iou(self):
        truth = [{"bbox": [0, 0, 10, 10]}, {"bbox": [4, 0, 10, 10]}]
        predictions = [{"bbox": [1, 0, 10, 10]}, {"bbox": [-3, 0, 10, 10]}]

        matches = evaluate_tracking.optimal_matches(truth, predictions, 0.3)

        assert [
            (truth_index, prediction_index)
            for truth_index, prediction_index, _ in matches
        ] == [(0, 1), (1, 0)]

    def test_matching_maximizes_total_iou_at_equal_cardinality(self):
        truth = [{"bbox": [0, 0, 10, 10]}, {"bbox": [4, 0, 10, 10]}]
        predictions = [{"bbox": [1, 0, 10, 10]}, {"bbox": [3, 0, 10, 10]}]

        matches = evaluate_tracking.optimal_matches(
            truth, predictions, 0.3, {(0, 1), (1, 0)}
        )

        assert [
            (truth_index, prediction_index)
            for truth_index, prediction_index, _ in matches
        ] == [(0, 0), (1, 1)]

    def test_matching_preserves_prior_identity_only_after_iou_ties(self):
        truth = {
            0: {
                "frame_index": 0,
                "width": 640,
                "height": 640,
                "objects": [
                    {"track_id": "a", "bbox": [10, 10, 8, 8]},
                    {"track_id": "b", "bbox": [10, 10, 8, 8]},
                ],
            },
            1: {
                "frame_index": 1,
                "width": 640,
                "height": 640,
                "objects": [
                    {"track_id": "b", "bbox": [10, 10, 8, 8]},
                    {"track_id": "a", "bbox": [10, 10, 8, 8]},
                ],
            },
        }
        predictions = {
            frame_index: {
                "frame_index": frame_index,
                "tracks": [
                    {"id": "1", "bbox": [10, 10, 8, 8]},
                    {"id": "2", "bbox": [10, 10, 8, 8]},
                ],
            }
            for frame_index in truth
        }

        report = evaluate_tracking.evaluate(
            truth, predictions, iou_threshold=0.3, fps=30.0, model_size=640
        )

        assert report["detection"]["true_positives"] == 4
        assert report["tracking"]["id_switches"] == 0

    def test_prediction_frames_outside_annotation_are_rejected(self):
        truth = {
            10: {
                "frame_index": 10,
                "width": 640,
                "height": 640,
                "objects": [],
            }
        }
        predictions = {
            10: {"frame_index": 10, "tracks": []},
            11: {"frame_index": 11, "tracks": []},
        }

        with pytest.raises(
            ValueError, match="predictions contain frames without ground truth: 11"
        ):
            evaluate_tracking.evaluate(
                truth, predictions, iou_threshold=0.3, fps=30.0, model_size=640
            )

    def test_reader_accepts_insight_metadata_envelope(self, tmp_path: Path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "type": "tracking",
                    "frame_id": "42",
                    "timestamp": 1000,
                    "data": {"tracks": [{"id": "7", "bbox": [1, 2, 3, 4]}]},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        frames = evaluate_tracking.read_jsonl(path, "tracks")

        assert frames[42]["tracks"][0]["id"] == "7"

    def test_reader_accepts_json_encoded_insight_data(self, tmp_path: Path):
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            json.dumps(
                {
                    "type": "tracking",
                    "frame_id": 7,
                    "data": json.dumps({"tracks": [{"id": "2", "bbox": [1, 2, 3, 4]}]}),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        frames = evaluate_tracking.read_jsonl(path, "tracks")

        assert frames[7]["tracks"][0]["id"] == "2"

    def test_empty_prediction_capture_is_scored_as_no_detections(self, tmp_path: Path):
        path = tmp_path / "predictions.jsonl"
        path.write_text("", encoding="utf-8")
        truth = {
            0: {
                "frame_index": 0,
                "width": 640,
                "height": 640,
                "objects": [{"track_id": "drone", "bbox": [10, 10, 8, 8]}],
            }
        }

        predictions = evaluate_tracking.read_jsonl(path, "tracks", allow_empty=True)
        report = evaluate_tracking.evaluate(
            truth, predictions, iou_threshold=0.3, fps=30.0, model_size=640
        )

        assert predictions == {}
        assert report["detection"]["recall"] == 0.0
        assert report["detection"]["false_negatives"] == 1
        assert report["frames"] == 1
