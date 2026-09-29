"""Unit tests for multi-stream-multi-model (Python)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit

DEFAULT_STREAMS = [
    ("rtsp://127.0.0.1:8554/src1", "detection", "yolov8", "models/yolo_11s_mpk.tar.gz"),
    ("rtsp://127.0.0.1:8554/src2", "segmentation", "yolov8", "models/yolo_11s_seg_mpk.tar.gz"),
    ("rtsp://127.0.0.1:8554/src3", "pose", "yolo26", "models/yolo26m-pose-int8-b1.tar.gz"),
    ("rtsp://127.0.0.1:8554/src4", "detection", "yolo26", "models/yolo26m-det-int8-b1.tar.gz"),
]


def write_config(
    tmp_path: Path,
    streams: list[tuple[str, str, str, str]],
    codec: str | None = None,
    max_inflight_per_stream: int | None = None,
    extra_output: list[str] | None = None,
) -> Path:
    stream_lines: list[str] = []
    for url, task, decode, model in streams:
        stream_lines.append(f"  - url: {url}")
        stream_lines.append(f"    task: {task}")
        if decode:
            stream_lines.append(f"    decode: {decode}")
        stream_lines.append(f"    model: {model}")

    input_config = ["input:", f"  codec: {codec}"] if codec else []
    inference: list[str] = []
    if max_inflight_per_stream is not None:
        inference.append("inference:")
        inference.append(f"  max_inflight_per_stream: {max_inflight_per_stream}")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                f"  labels: {EXAMPLE_DIR / 'src' / 'common' / 'coco_label.txt'}",
                "streams:",
                *stream_lines,
                *input_config,
                *inference,
                "output:",
                "  insight:",
                "    host: 127.0.0.1",
                *(extra_output or []),
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

    def test_validate_config_only_reports_tasks(self, tmp_path: Path):
        config_path = write_config(tmp_path, DEFAULT_STREAMS)

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path), "--validate-config-only"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 0
        assert "streams=4" in result.stdout
        assert "tasks=detection,segmentation,pose,detection" in result.stdout

    def test_packaged_config_validates(self):
        result = subprocess.run(
            [
                sys.executable,
                str(MAIN_PY),
                "--config",
                str(COMMON_CONFIG),
                "--validate-config-only",
            ],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )

        assert result.returncode == 0, result.stderr
        assert "tasks=detection,segmentation,pose,detection" in result.stdout


class TestConfigLoading:
    def test_each_stream_carries_its_own_task_decode_and_model(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(write_config(tmp_path, DEFAULT_STREAMS))

        assert [stream.task for stream in cfg.streams] == [
            "detection",
            "segmentation",
            "pose",
            "detection",
        ]
        assert [stream.decode for stream in cfg.streams] == [
            "yolov8",
            "yolov8",
            "yolo26",
            "yolo26",
        ]
        assert [stream.index for stream in cfg.streams] == [0, 1, 2, 3]
        assert cfg.streams[2].model_path == "models/yolo26m-pose-int8-b1.tar.gz"
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30
        assert cfg.max_inflight_per_stream == 4

    def test_decode_defaults_to_yolo26(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(
            write_config(tmp_path, [("rtsp://127.0.0.1:8554/src1", "pose", "", "models/m.tar.gz")])
        )

        assert cfg.streams[0].decode == "yolo26"

    @pytest.mark.parametrize(("codec", "expected"), [("avc", "h264"), ("hevc", "h265")])
    def test_codec_alias_is_accepted(self, tmp_path: Path, codec: str, expected: str):
        from main import load_app_config

        cfg = load_app_config(write_config(tmp_path, DEFAULT_STREAMS[:1], codec=codec))

        assert cfg.codec == expected

    def test_custom_inflight_limits_are_accepted(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(
            write_config(
                tmp_path,
                DEFAULT_STREAMS[:1],
                max_inflight_per_stream=3,
            )
        )

        assert cfg.max_inflight_per_stream == 3

    def test_invalid_inflight_limit_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(tmp_path, DEFAULT_STREAMS[:1], max_inflight_per_stream=0)

        with pytest.raises(ValueError, match="max_inflight_per_stream must be -1 or > 0"):
            load_app_config(config_path)

    def test_unknown_task_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path, [("rtsp://127.0.0.1:8554/src1", "tracking", "yolo26", "models/m.tar.gz")]
        )

        with pytest.raises(ValueError, match=r"streams\[0\].task must be one of"):
            load_app_config(config_path)

    def test_unknown_decode_family_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path, [("rtsp://127.0.0.1:8554/src1", "detection", "yolov5", "models/m.tar.gz")]
        )

        with pytest.raises(ValueError, match=r"streams\[0\].decode must be one of"):
            load_app_config(config_path)

    def test_stream_without_model_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                streams:
                  - url: rtsp://127.0.0.1:8554/src1
                    task: detection
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"streams\[0\].model"):
            load_app_config(config_path)

    def test_more_than_four_streams_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        streams = DEFAULT_STREAMS + [
            ("rtsp://127.0.0.1:8554/src5", "detection", "yolo26", "models/m.tar.gz")
        ]

        with pytest.raises(ValueError, match="up to four streams"):
            load_app_config(write_config(tmp_path, streams))

    def test_empty_streams_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
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

    def test_invalid_mask_threshold_is_rejected(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path, DEFAULT_STREAMS[:1], extra_output=["  mask_threshold: 1.5"]
        )

        with pytest.raises(ValueError, match="mask_threshold"):
            load_app_config(config_path)


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

        assert main.encoded_decode_input_options(fake_pyneat.RtspCodec.H265).format == "h265"
        assert main.encoded_video_input_options(fake_pyneat.RtspCodec.H265).format == "h265"
        assert main.encoded_decode_input_options(fake_pyneat.RtspCodec.H264).format == "h264"
        assert main.encoded_video_input_options(fake_pyneat.RtspCodec.H264).format == "h264"

    def test_realtime_link_tags_its_stream(self, monkeypatch):
        import main

        fake_pyneat = SimpleNamespace(
            GraphLinkOptions=type("GraphLinkOptions", (), {}),
            GraphLinkPolicy=SimpleNamespace(RealtimeLatestByStream="latest-by-stream"),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        link = main.realtime_link(2, 4)

        assert link.policy == "latest-by-stream"
        assert link.queue_depth == 4
        assert link.stream_id == "stream2"

    def test_source_frame_endpoints_are_distinct_per_stream(self):
        """Streams share one source Run, so their frame endpoints must not collide."""
        import main

        names = [main.source_frame_name(index) for index in range(4)]

        assert len(set(names)) == len(names)
        assert names[2] == "frame_2"

    def test_model_endpoints_are_shared_because_each_model_has_its_own_run(self):
        """Model endpoint names are constants: nothing shares a Run with another model."""
        import main

        assert main.MODEL_INPUT == "image"
        assert main.MODEL_OUTPUT == "results"

    @pytest.mark.parametrize(
        ("task", "decode", "expected"),
        [
            ("detection", "yolov8", "YoloV8"),
            ("detection", "yolo26", "YoloV26"),
            ("segmentation", "yolov8", "YoloV8Seg"),
            ("segmentation", "yolo26", "YoloV26Seg"),
            ("pose", "yolov8", "YoloV8Pose"),
            ("pose", "yolo26", "YoloV26Pose"),
        ],
    )
    def test_decode_family_follows_task_and_head_layout(
        self, monkeypatch, task: str, decode: str, expected: str
    ):
        import main

        fake_pyneat = SimpleNamespace(
            BoxDecodeType=SimpleNamespace(
                YoloV8="YoloV8",
                YoloV8Seg="YoloV8Seg",
                YoloV8Pose="YoloV8Pose",
                YoloV26="YoloV26",
                YoloV26Seg="YoloV26Seg",
                YoloV26Pose="YoloV26Pose",
            )
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        assert main.decode_type_for(task, decode) == expected


class FakeMetadataSender:
    def __init__(self):
        self.calls = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
        self.calls.append((metadata_type, data_json, timestamp_ms, frame_id))
        return True


class FakeSample:
    frame_id = 42
    pts_ns = 1_234_000_000


# The stamp the feeder captured from the source frame. The model lives in its own Run and its
# results come back with no PTS, so the metadata timestamp has to come from here or Insight
# cannot correlate the overlay to the frame it describes.
SOURCE_STAMP = (1_234_000_000, 42)


def make_stream(task: str, sender: FakeMetadataSender):
    import main

    return main.StreamRuntime(
        index=0,
        task=task,
        url="rtsp://127.0.0.1:8554/src1",
        model_path="models/model.tar.gz",
        source_options=None,
        model=None,
        metadata_sender=sender,
        labels=["person"],
        profile=main.ProfileWindow(False, 0, task),
        latest_debug_frame=None,
        frame_w=100,
        frame_h=100,
        output_fps=30,
        video_port=9000,
    )


class TestMetadata:
    """Every task publishes on its own Insight contract, from one shared send path."""

    def setup_method(self):
        import main

        main.load_runtime_dependencies()

    def test_detection_stream_publishes_object_detection(self):
        import main

        sender = FakeMetadataSender()
        stream = make_stream("detection", sender)
        boxes = [
            {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": 0.75, "class_id": 0}
        ]

        dropped = main.send_metadata(_config(), stream, SOURCE_STAMP, boxes)

        assert dropped == 0
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "object-detection"
        # Insight correlates overlay to frame on this timestamp; -1 or a missing value makes it
        # paint the newest metadata onto whatever frame is on screen.
        assert timestamp_ms == 1234
        assert frame_id == "42"
        assert json.loads(data_json) == {
            "objects": [
                {
                    "id": "obj_1",
                    "label": "person",
                    "confidence": 0.75,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                }
            ]
        }

    def test_pose_stream_publishes_named_keypoints(self):
        import main

        sender = FakeMetadataSender()
        stream = make_stream("pose", sender)
        poses = [
            {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 40.0,
                "y2": 60.0,
                "score": 0.9,
                "keypoints": [
                    {"x": float(i), "y": float(i * 2), "visibility": 0.5}
                    for i in range(len(main.COCO_KEYPOINT_NAMES))
                ],
            }
        ]

        main.send_metadata(_config(), stream, SOURCE_STAMP, poses)

        metadata_type, data_json, _, _ = sender.calls[0]
        assert metadata_type == "pose-estimation"
        data = json.loads(data_json)
        assert data["poses"][0]["label"] == "person"
        assert data["poses"][0]["bbox"] == [10, 20, 30, 40]
        assert [point["name"] for point in data["poses"][0]["keypoints"]] == list(
            main.COCO_KEYPOINT_NAMES
        )

    def test_segmentation_stream_publishes_frame_absolute_polygons(self):
        import main

        sender = FakeMetadataSender()
        stream = make_stream("segmentation", sender)
        detections = [
            {
                "x1": 8.0,
                "y1": 8.0,
                "x2": 56.0,
                "y2": 56.0,
                "score": 0.9,
                "class_id": 0,
                "mask": np.full((160, 160), 255, dtype=np.uint8),
            }
        ]

        main.send_metadata(_config(), stream, SOURCE_STAMP, detections)

        metadata_type, data_json, _, _ = sender.calls[0]
        assert metadata_type == "segmentation"
        segment = json.loads(data_json)["segments"][0]
        assert segment["mask_format"] == "polygon"
        assert len(segment["mask"]) >= 3
        assert all(0 <= x <= 100 and 0 <= y <= 100 for x, y in segment["mask"])


class TestSegmentationBudget:
    def setup_method(self):
        import main

        main.load_runtime_dependencies()

    def test_budget_drops_lowest_confidence_first(self):
        """A full frame of detailed silhouettes overruns one datagram, so the tail is dropped."""
        import main

        detections = [
            {
                "x1": 0.0,
                "y1": 0.0,
                "x2": 1920.0,
                "y2": 1080.0,
                "score": 0.5 + 0.005 * index,
                "class_id": 0,
                "mask": _comb_mask(),
            }
            for index in range(50)
        ]

        data, dropped = main.segmentation_metadata_data(
            detections, ["person"], (1080, 1920, 3), 0.5
        )
        payload = json.dumps(data, separators=(",", ":"))

        assert len(payload) <= main.METADATA_BYTE_BUDGET
        assert dropped > 0
        assert len(data["segments"]) + dropped == len(detections)
        lowest_kept = min(segment["confidence"] for segment in data["segments"])
        assert lowest_kept == pytest.approx(0.5 + 0.005 * dropped, abs=1e-3)


def _config():
    """Config with defaults only, for the metadata paths that read thresholds off it."""
    import main

    return main.AppConfig(streams=[], labels_path=Path("coco_label.txt"))


def _comb_mask():
    """A mask whose contour keeps many vertices, so the byte budget is actually exercised."""
    import cv2

    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.rectangle(mask, (10, 60), (150, 150), 255, -1)
    for x in range(10, 150, 4):
        cv2.rectangle(mask, (x, 20), (x + 2, 60), 255, -1)
    return mask
