"""Unit tests for the multistream people tracker Insight example."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
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

ACCURACY_SPEC = importlib.util.spec_from_file_location("evaluate_tracking", ACCURACY_PATH)
assert ACCURACY_SPEC and ACCURACY_SPEC.loader
evaluate_tracking = importlib.util.module_from_spec(ACCURACY_SPEC)
ACCURACY_SPEC.loader.exec_module(evaluate_tracking)


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
        assert cfg.target_class_id == 0
        assert cfg.target_label == "person"

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
            latest_debug_frame=None,
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


class TestTracker:
    def test_tracker_reuses_track_id_for_nearby_detection(self):
        from utils.tracker import ObjectTracker

        tracker = ObjectTracker()
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
        from utils.tracker import ObjectTracker, TrackerConfig

        tracker = ObjectTracker(TrackerConfig(max_missing_frames=1))
        tracker.update(
            [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 80.0, "score": 0.9, "class_id": 0}],
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


class TestAccuracyEvaluation:
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
        assert (
            report["detection"]["recall_by_model_input_size"]["tiny"]["ground_truth"]
            == 3
        )

    def test_greedy_matching_is_one_to_one(self):
        truth = [{"bbox": [0, 0, 10, 10]}, {"bbox": [1, 1, 10, 10]}]
        predictions = [{"bbox": [0, 0, 10, 10]}]

        matches = evaluate_tracking.greedy_matches(truth, predictions, 0.3)

        assert len(matches) == 1

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
