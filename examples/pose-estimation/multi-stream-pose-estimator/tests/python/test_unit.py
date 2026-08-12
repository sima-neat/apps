"""Unit tests for the multistream pose estimation Insight example."""

from __future__ import annotations

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

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


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

        assert cfg.model_path == "models/yolo26m-pose-int8-b1.tar.gz"
        assert len(cfg.rtsp_urls) == 4
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30
        assert cfg.input_max_width == 1920
        assert cfg.input_max_height == 1080
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16

    def test_load_app_config_accepts_custom_input_capacity(self, tmp_path: Path):
        from main import load_app_config

        cfg = load_app_config(
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
        from main import load_app_config

        cfg = load_app_config(
            write_config(tmp_path, ["rtsp://127.0.0.1:8554/src1"], codec=codec)
        )
        assert cfg.codec == expected

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
        import main

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                return {1: 2560, 2: 1440, 3: 20}[prop]

            def release(self):
                pass

        monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
        monkeypatch.setattr(
            main,
            "cv2",
            SimpleNamespace(
                VideoCapture=lambda _url: FakeCapture(),
                CAP_PROP_FRAME_WIDTH=1,
                CAP_PROP_FRAME_HEIGHT=2,
                CAP_PROP_FPS=3,
            ),
        )

        assert main.probe_rtsp("rtsp://camera/stream", True) == (2560, 1440, 20)
        assert main.os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"

    def test_model_preprocess_uses_configured_capacity(self, monkeypatch):
        import main

        captured = SimpleNamespace(options=None)

        class FakeModelOptions:
            def __init__(self):
                self.preprocess = SimpleNamespace(color_convert=SimpleNamespace())

        def fake_model(_path, options):
            captured.options = options
            return object()

        monkeypatch.setattr(
            main,
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
        main.build_model(cfg)

        assert captured.options.preprocess.input_max_width == 2560
        assert captured.options.preprocess.input_max_height == 1440

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
    def test_send_metadata_uses_pose_estimation_contract(self):
        from main import ProfileWindow, StreamRuntime, send_metadata

        sender = FakeMetadataSender()
        runtime = StreamRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            source_options=None,
            metadata_sender=sender,
            profile=ProfileWindow(False, 0),
            latest_debug_frame=None,
            frame_w=100,
            frame_h=100,
            output_fps=30,
            video_port=9000,
        )
        # Only the nose clears the visibility floor, so the published payload must
        # carry that one joint and drop the other sixteen.
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

        send_metadata(
            runtime, SimpleNamespace(min_keypoint_visibility=0.30), FakeSample(), poses
        )

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "pose-estimation"
        assert timestamp_ms == 1234
        assert frame_id == "42"
        assert json.loads(data_json) == {
            "poses": [
                {
                    "id": "pose_1",
                    "label": "person",
                    "confidence": 0.75,
                    "bbox": [10.0, 20.0, 30.0, 40.0],
                    "keypoints": [
                        {"name": "nose", "x": 11.0, "y": 21.0, "confidence": 0.9}
                    ],
                }
            ]
        }
