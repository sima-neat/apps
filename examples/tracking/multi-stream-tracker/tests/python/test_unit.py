"""Unit tests for the multi-stream, multi-class tracker example."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"
GOLDEN = EXAMPLE_DIR / "tests" / "common" / "tracker_golden.json"

DEFAULT_CLASSES = """\
tracking:
  classes:
    - class: person
    - class: car
      max_missing_frames: 45
"""

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


def write_config(
    tmp_path: Path,
    streams: list[str],
    codec: str | None = None,
    max_inflight_per_stream: int | None = None,
    max_inflight_total: int | None = None,
    tracking: str = DEFAULT_CLASSES,
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
                tracking.rstrip("\n"),
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
        assert [c.label for c in cfg.tracker_classes] == ["person", "car"]
        assert [c.max_missing_frames for c in cfg.tracker_classes] == [30, 45]
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

    def test_default_config_loads_classes(self):
        from main import load_app_config

        cfg = load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.min_score == 0.10
        assert [c.label for c in cfg.tracker_classes] == ["person", "car"]

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
        assert "classes=person,car" in result.stdout
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
        from utils.tracker import MultiClassTracker, TrackedDetection, parse_class_configs

        sender = FakeMetadataSender()
        runtime = StreamRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            source_options=None,
            metadata_sender=sender,
            tracker=MultiClassTracker(parse_class_configs([{"class": "person"}])),
            profile=ProfileWindow(False, 0),
            latest_debug_frame=None,
            frame_w=100,
            frame_h=100,
            output_fps=30,
            video_port=9000,
        )
        tracks = [
            TrackedDetection(7, 10.0, 20.0, 40.0, 60.0, 0.75, 0, "person"),
            TrackedDetection(8, 50.0, 50.0, 90.0, 80.0, 0.5, 2, "car"),
        ]

        send_metadata(runtime, FakeSample(), tracks)

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "tracking"
        assert timestamp_ms == 1234
        assert frame_id == "42"
        assert json.loads(data_json) == {
            "tracks": [
                {"id": "7", "label": "person", "confidence": 0.75, "bbox": [10.0, 20.0, 30.0, 40.0]},
                {"id": "8", "label": "car", "confidence": 0.5, "bbox": [50.0, 50.0, 40.0, 30.0]},
            ]
        }


def det(x1, y1, x2, y2, score=0.9, class_id=0):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score, "class_id": class_id}


def moving(frame, class_id=0, x=10.0, speed=4.0, score=0.9):
    return det(x + speed * frame, 10.0, x + speed * frame + 40.0, 90.0, score, class_id)


class TestClassConfig:
    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
    def test_accepts_one_to_five_classes(self, count):
        from utils.tracker import parse_class_configs

        names = ["person", "car", "bicycle", "dog", "truck"][:count]
        configs = parse_class_configs([{"class": name} for name in names])

        assert [c.label for c in configs] == names

    @pytest.mark.parametrize(
        "classes, message",
        [
            (None, "1 to 5 classes"),
            ([], "1 to 5 classes"),
            ([{"class": n} for n in ["person", "car", "bus", "dog", "cat", "truck"]], "1 to 5 classes"),
            ([{"class": "person"}, {"class": "person"}], "duplicate class 'person'"),
            ([{"class": "person"}, {"class": 0}], "duplicate class 'person'"),
            ([{"class": "spaceship"}], "unsupported class"),
            ([{"class": 80}], "unsupported class id 80"),
            ([{"class": -1}], "unsupported class id -1"),
            ([{"class": True}], "unsupported class"),
            (["person"], "must be a mapping with a 'class' key"),
            ([{"max_missing_frames": 3}], "must be a mapping with a 'class' key"),
            ([{"class": "person", "speed": 1}], "unknown key 'speed'"),
            ([{"class": "person", "max_missing_frames": -1}], "max_missing_frames must be >= 0"),
            ([{"class": "person", "max_missing_frames": 2.5}], "must be an integer"),
            ([{"class": "person", "match_iou_threshold": 1.5}], "between 0 and 1"),
            ([{"class": "person", "low_score_threshold": 0.6}], "low_score_threshold must be <="),
            ([{"class": "person", "new_track_threshold": 0.3}], "new_track_threshold must be >="),
            ([{"class": "person", "velocity_noise": 0}], "velocity_noise must be > 0"),
            ([{"class": "person", "min_confirmed_hits": 0}], "min_confirmed_hits must be >= 1"),
        ],
    )
    def test_rejects_invalid_classes(self, classes, message):
        from utils.tracker import parse_class_configs

        with pytest.raises(ValueError, match=message):
            parse_class_configs(classes)

    def test_accepts_numeric_ids_and_names(self):
        from utils.tracker import parse_class_configs

        configs = parse_class_configs([{"class": 2}, {"class": "Traffic Light"}, {"class": "7"}])

        assert [(c.class_id, c.label) for c in configs] == [(2, "car"), (9, "traffic light"), (7, "truck")]

    def test_config_file_rejects_missing_classes(self, tmp_path: Path):
        from main import load_app_config

        path = write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], tracking="")
        with pytest.raises(ValueError, match="1 to 5 classes"):
            load_app_config(path)

    def test_config_file_rejects_min_score_above_class_low_threshold(self, tmp_path: Path):
        from main import load_app_config

        tracking = "tracking:\n  classes:\n    - class: person\n      low_score_threshold: 0.05\n"
        path = write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], tracking=tracking)
        with pytest.raises(ValueError, match="min_score must be <= low_score_threshold of class 'person'"):
            load_app_config(path)

    def test_validate_config_only_rejects_duplicate_class(self, tmp_path: Path):
        tracking = "tracking:\n  classes:\n    - class: car\n    - class: 2\n"
        path = write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], tracking=tracking)
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(path), "--validate-config-only"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 1
        assert "duplicate class 'car'" in result.stderr


class TestTracker:
    def make(self, classes):
        from utils.tracker import MultiClassTracker, parse_class_configs

        return MultiClassTracker(parse_class_configs(classes))

    def test_reuses_track_id_for_moving_object(self):
        tracker = self.make([{"class": "person"}])
        ids = [t.track_id for f in range(10) for t in tracker.update([moving(f)], f)]

        assert len(ids) == 9  # Published after the second hit.
        assert set(ids) == {1}

    def test_publishes_class_label_and_track_id(self):
        tracker = self.make([{"class": "person"}, {"class": "car"}])
        for f in range(3):
            out = tracker.update([moving(f, 0), moving(f, 2, x=400.0)], f)

        assert [(t.track_id, t.class_id, t.label) for t in out] == [(1, 0, "person"), (2, 2, "car")]

    def test_multiple_tracks_from_multiple_classes(self):
        tracker = self.make([{"class": "person"}, {"class": "car"}, {"class": "dog"}])
        for f in range(5):
            dets = [moving(f, cls, x=x) for cls in (0, 2, 16) for x in (10.0, 300.0, 600.0)]
            out = tracker.update(dets, f)

        assert len(out) == 9
        assert len({t.track_id for t in out}) == 9
        assert sorted({t.label for t in out}) == ["car", "dog", "person"]

    def test_ignores_unconfigured_classes(self):
        tracker = self.make([{"class": "person"}])
        for f in range(4):
            out = tracker.update([moving(f, 0), moving(f, 2, x=300.0), moving(f, 9, x=10.0)], f)

        assert [t.label for t in out] == ["person"]

    def test_class_isolation_same_box_other_class(self):
        tracker = self.make([{"class": "person"}, {"class": "car"}])
        for f in range(4):
            tracker.update([moving(f, 0)], f)
        # The person disappears; a car appears exactly where the person would be.
        car_ids = set()
        person_track_classes = set()
        for f in range(4, 12):
            for t in tracker.update([moving(f, 2)], f):
                car_ids.add(t.track_id)
                assert t.class_id == 2 and t.label == "car"
        for f in range(12, 16):
            for t in tracker.update([moving(f, 0)], f):
                person_track_classes.add((t.track_id, t.class_id))

        assert 1 not in car_ids
        assert person_track_classes == {(1, 0)}

    def test_tracks_never_switch_class(self):
        tracker = self.make([{"class": "person"}, {"class": "car"}, {"class": "truck"}])
        seen = {}
        for f in range(60):
            cls = (0, 2, 7)[(f // 3) % 3]  # Detector flickers between classes for one object.
            for t in tracker.update([moving(f, cls), moving(f, 0, x=500.0)], f):
                assert seen.setdefault(t.track_id, t.class_id) == t.class_id

    def test_short_missed_detections_keep_id(self):
        tracker = self.make([{"class": "person", "max_missing_frames": 5}])
        ids = set()
        for f in range(20):
            dets = [] if f in (8, 9, 10) else [moving(f, speed=8.0)]
            ids |= {t.track_id for t in tracker.update(dets, f)}

        assert ids == {1}

    def test_low_score_detection_recovers_active_track(self):
        tracker = self.make([{"class": "person"}])
        outs = []
        for f in range(10):
            score = 0.2 if f in (5, 6) else 0.9
            outs.append(tracker.update([moving(f, score=score)], f))

        assert [len(o) for o in outs[1:]] == [1] * 9
        assert {o[0].track_id for o in outs[1:]} == {1}

    def test_low_score_detection_never_starts_track(self):
        tracker = self.make([{"class": "person"}])
        outs = [tracker.update([moving(f, score=0.3)], f) for f in range(10)]

        assert all(not o for o in outs)
        assert tracker.active_track_count() == 0

    def test_track_expiry_uses_class_budget(self):
        tracker = self.make([{"class": "person", "max_missing_frames": 3}, {"class": "car", "max_missing_frames": 12}])
        for f in range(5):
            tracker.update([moving(f, 0), moving(f, 2, x=400.0)], f)
        for f in range(5, 10):
            tracker.update([], f)
        assert tracker.active_track_count() == 1  # Person expired, car still waiting.

        out = []
        for f in range(10, 13):
            out = tracker.update([moving(f, 0), moving(f, 2, x=400.0)], f)
        ids = {t.label: t.track_id for t in out}
        assert ids["car"] == 2
        assert ids["person"] > 2

    def test_class_specific_match_threshold(self):
        # Fast object, no motion history yet: overlap between frames is ~0.18 IoU.
        def run(threshold):
            tracker = self.make([{"class": "car", "match_iou_threshold": threshold, "min_confirmed_hits": 1}])
            return {t.track_id for f in range(3) for t in tracker.update([moving(f, 2, speed=28.0)], f)}

        assert run(0.1) == {1}
        assert len(run(0.3)) > 1

    def test_class_specific_motion_noise_changes_prediction(self):
        def predicted_x(velocity_noise):
            tracker = self.make([{"class": "car", "velocity_noise": velocity_noise}])
            for f in range(6):
                out = tracker.update([moving(f, 2, speed=20.0)], f)
            return out[0].x1

        assert predicted_x(0.05) != pytest.approx(predicted_x(0.00625))

    def test_streams_are_independent(self):
        stream_a = self.make([{"class": "person"}, {"class": "car"}])
        stream_b = self.make([{"class": "person"}, {"class": "car"}])
        for f in range(4):
            a = stream_a.update([moving(f, 0), moving(f, 2, x=300.0)], f)
            b = stream_b.update([moving(f, 2, x=300.0)], f)

        assert [(t.track_id, t.label) for t in a] == [(1, "person"), (2, "car")]
        assert [(t.track_id, t.label) for t in b] == [(1, "car")]

    def test_linear_assignment_is_optimal(self):
        from utils.tracker import linear_assignment

        assert linear_assignment([[1, 2, 3], [2, 4, 6], [3, 6, 9]], 10) == [(0, 2), (1, 1), (2, 0)]
        assert linear_assignment([[0.1, 0.2], [0.05, 0.9], [0.4, 0.3]], 1) == [(0, 1), (1, 0)]
        assert linear_assignment([[0.9]], 0.5) == []
        assert linear_assignment([], 1) == []


class TestGoldenParity:
    """Replays the shared fixture that the C++ unit test also replays."""

    @pytest.mark.parametrize(
        "case", json.loads(GOLDEN.read_text())["cases"], ids=lambda case: case["name"]
    )
    def test_matches_golden(self, case):
        from utils.tracker import MultiClassTracker, parse_class_configs

        tracker = MultiClassTracker(parse_class_configs(case["classes"]))
        for f, (dets, expected) in enumerate(zip(case["frames"], case["expected"])):
            out = tracker.update(
                [{"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3], "score": d[4], "class_id": d[5]} for d in dets],
                f,
            )
            assert [[t.track_id, t.class_id, t.label] for t in out] == [e[:3] for e in expected], f"frame {f}"
            for t, e in zip(out, expected):
                assert (t.x1, t.y1, t.x2, t.y2) == pytest.approx(tuple(e[3:]), abs=1e-2), f"frame {f}"
