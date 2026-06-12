"""Unit tests for the 16-stream object detection Insight example."""

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
MODEL_PATH = "assets/models/yolo26m-det-int8-b1.tar.gz"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


def write_config(
    tmp_path: Path, streams: list[str], workers: int = 4, decode_type: str | None = None
) -> Path:
    stream_lines = "\n".join(f"  - {stream}" for stream in streams)
    model_lines = ["model:", f"  path: {MODEL_PATH}"]
    if decode_type is not None:
        model_lines.append(f"  decode_type: {decode_type}")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            model_lines
            + [
                "streams:",
                stream_lines,
                "input:",
                "  tcp: true",
                "  latency_ms: 100",
                "inference:",
                f"  workers: {workers}",
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
    def test_load_app_config_accepts_sixteen_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 17)],
            workers=4,
        )

        cfg = load_app_config(config_path)

        assert cfg.model_path == MODEL_PATH
        assert cfg.decode_type == "yolo26"
        assert len(cfg.rtsp_urls) == 16
        assert cfg.workers == 4
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30

    def test_load_app_config_accepts_yolov8_decode_type(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            decode_type="yolov8",
        )

        cfg = load_app_config(config_path)

        assert cfg.decode_type == "yolov8"

    def test_load_app_config_rejects_too_many_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 18)],
            workers=4,
        )

        with pytest.raises(ValueError, match="up to 16 streams"):
            load_app_config(config_path)

    def test_load_app_config_rejects_empty_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""
                model:
                  path: {MODEL_PATH}
                streams: []
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            load_app_config(config_path)

    def test_load_app_config_rejects_invalid_worker_count(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
            workers=3,
        )

        with pytest.raises(ValueError, match="workers must be <= stream count"):
            load_app_config(config_path)

    def test_load_app_config_rejects_fps_scheduler_knobs(self, tmp_path: Path):
        from main import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""
                model:
                  path: {MODEL_PATH}
                streams:
                  - rtsp://127.0.0.1:8554/src1
                inference:
                  workers: 1
                  target_fps: 15
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="target_fps is not supported"):
            load_app_config(config_path)

    def test_validate_config_only_reports_stream_and_worker_count(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
            workers=2,
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
        assert "workers=2" in result.stdout


class TestScheduler:
    def test_assign_streams_to_workers_uses_fixed_modulo_assignment(self):
        from main import assign_streams_to_workers

        assert assign_streams_to_workers(16, 4) == [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
        ]

    def test_assign_streams_to_workers_covers_each_stream_once(self):
        from main import assign_streams_to_workers

        assignments = assign_streams_to_workers(10, 3)
        assigned = sorted(stream for worker in assignments for stream in worker)

        assert assigned == list(range(10))
        assert (
            max(len(worker) for worker in assignments)
            - min(len(worker) for worker in assignments)
            <= 1
        )

    def test_latest_frame_mailbox_keeps_newest_frame(self):
        from main import LatestFrameMailbox

        mailbox = LatestFrameMailbox()
        mailbox.store("old")
        mailbox.store("new")

        frame, overwrites = mailbox.take()

        assert frame == "new"
        assert overwrites == 1
        assert mailbox.done() is False

        mailbox.close()

        assert mailbox.take() == (None, 0)
        assert mailbox.done() is True


class TestRuntimeOptions:
    def test_source_options_keep_decoded_handoff_device_visible(self, monkeypatch):
        import main

        class FakeRtspDecodedInputOptions:
            def __init__(self):
                self.output_caps = SimpleNamespace()

        fake_pyneat = SimpleNamespace(
            RtspDecodedInputOptions=FakeRtspDecodedInputOptions,
            Format=SimpleNamespace(NV12="NV12"),
            CapsMemory=SimpleNamespace(Any="Any"),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        cfg = main.AppConfig(
            model_path=MODEL_PATH,
            labels_path=Path("labels.txt"),
            rtsp_urls=["rtsp://127.0.0.1:8554/src1"],
        )

        opt = main.make_source_options(cfg, cfg.rtsp_urls[0], fps=30, width=640, height=480)

        assert opt.out_format == "NV12"
        assert opt.decoder_raw_output is True
        assert opt.decoder_next_element == "CVU"
        assert opt.output_caps.enable is True
        assert opt.output_caps.format == "NV12"
        assert opt.output_caps.width == 640
        assert opt.output_caps.height == 480
        assert opt.output_caps.fps == 30
        assert opt.output_caps.memory == "Any"

    def test_detector_input_tensors_materializes_frame_on_cvu(self, monkeypatch):
        import main

        class FakeTensor:
            def __init__(self):
                self.cvu_called = False

            def cvu(self):
                self.cvu_called = True
                return "ev74_tensor"

        tensor = FakeTensor()
        monkeypatch.setattr(main, "first_tensor_from_sample", lambda frame: tensor)

        assert main.detector_input_tensors_from_frame(object()) == ["ev74_tensor"]
        assert tensor.cvu_called is True


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
    def test_send_metadata_uses_object_detection_contract(self):
        from main import SourceRuntime, StreamProfile, send_metadata

        sender = FakeMetadataSender()
        runtime = SourceRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            graph=None,
            run=None,
            metadata_sender=sender,
            labels=["person"],
            profile=StreamProfile(False, 0),
            frame_w=100,
            frame_h=100,
            source_fps=30,
            video_port=9000,
        )
        boxes = [
            {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 40.0,
                "y2": 60.0,
                "score": 0.75,
                "class_id": 0,
            }
        ]

        send_metadata(runtime, FakeSample(), boxes)

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "object-detection"
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
