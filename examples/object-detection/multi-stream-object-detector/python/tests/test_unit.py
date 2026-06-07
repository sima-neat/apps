"""Unit tests for the multistream object detection Insight example (Python)."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import textwrap

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"
MAIN_PY = PYTHON_DIR / "main.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


class TestUnitSuiteEntrypoint:
    def test_support_modules_import(self):
        import utils.config  # noqa: F401
        import utils.image_utils  # noqa: F401
        import utils.model_family  # noqa: F401
        import utils.pipeline  # noqa: F401
        import utils.sample_utils  # noqa: F401
        import utils.workers  # noqa: F401


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
    def test_load_app_config_parses_runtime_worker_count(self, tmp_path: Path):
        from utils.config import VideoMode, load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: assets/models/yolo_v8m_mpk.tar.gz
                streams:
                  - rtsp://127.0.0.1:8554/src1
                  - rtsp://127.0.0.1:8554/src2
                input:
                  tcp: true
                  latency_ms: 120
                runtime:
                  worker_count: 4
                  mailbox_depth: 1
                  profile: true
                inference:
                  frames: 0
                  fps: 0
                  min_score: 0.25
                  nms_iou: 0.45
                  max_detections: 100
                output:
                  insight:
                    host: 127.0.0.1
                    video_port_base: 9000
                    metadata_port_base: 9100
                    metadata_offset_ms: 12.5
                  video_enabled: false
                  video_mode: clean
                  debug_dir: null
                  save_every: 0
                """
            ).strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.model.path == "assets/models/yolo_v8m_mpk.tar.gz"
        assert cfg.worker_count == 4
        assert cfg.mailbox_depth == 1
        assert cfg.profile is True
        assert cfg.insight_metadata_offset_ms == pytest.approx(12.5)
        assert cfg.video_enabled is False
        assert cfg.video_mode == VideoMode.CLEAN
        assert cfg.rtsp_urls == [
            "rtsp://127.0.0.1:8554/src1",
            "rtsp://127.0.0.1:8554/src2",
        ]

    def test_load_app_config_rejects_removed_model_family_field(self, tmp_path: Path):
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: assets/models/unsupported_mpk.tar.gz
                  family: yolov8
                streams:
                  - rtsp://127.0.0.1:8554/src1
                runtime:
                  worker_count: 2
                output:
                  insight:
                    host: 127.0.0.1
                """
            ).strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"model\.family is no longer supported"):
            load_app_config(config_path)

    def test_metadata_output_enabled_follows_video_mode_contract(self, tmp_path: Path):
        from utils.config import metadata_output_enabled, load_app_config

        for mode, expected in (("clean", True), ("annotated", False)):
            config_path = tmp_path / f"config_{mode}.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    model:
                      path: assets/models/yolo_v8m_mpk.tar.gz
                    streams:
                      - rtsp://127.0.0.1:8554/src1
                    runtime:
                      worker_count: 2
                    output:
                      insight:
                        host: 127.0.0.1
                      video_enabled: true
                      video_mode: {mode}
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_app_config(config_path)
            assert metadata_output_enabled(cfg) is expected

    def test_metadata_output_enabled_stays_enabled_when_video_is_disabled(self, tmp_path: Path):
        from utils.config import metadata_output_enabled, load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                """
                model:
                  path: assets/models/yolo_v8m_mpk.tar.gz
                streams:
                  - rtsp://127.0.0.1:8554/src1
                runtime:
                  worker_count: 2
                output:
                  insight:
                    host: 127.0.0.1
                  video_enabled: false
                  video_mode: annotated
                """
            ).strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)
        assert metadata_output_enabled(cfg) is True


class TestModelFamily:
    def test_resolve_model_family_auto_for_yolov8(self):
        from utils.model_family import ModelFamily, resolve_model_family

        assert (
            resolve_model_family("assets/models/yolo_v8m_mpk.tar.gz", ModelFamily.AUTO)
            == ModelFamily.YOLOV8
        )


class TestPipelineHelpers:
    def test_port_helpers_and_writer_fps(self):
        from utils.config import AppConfig, ModelConfig
        from utils.model_family import ModelFamily
        from utils.pipeline import (
            RtspProbe,
            effective_writer_fps,
            insight_metadata_port_for_stream,
            insight_video_port_for_stream,
        )

        cfg = AppConfig(
            model=ModelConfig("assets/models/yolo_v8m_mpk.tar.gz"),
            rtsp_urls=["rtsp://127.0.0.1:8554/src1"],
            insight_host="127.0.0.1",
            fps=0,
        )
        probe = RtspProbe(width=1280, height=720, fps=30)

        assert insight_video_port_for_stream(9000, 3) == 9003
        assert insight_metadata_port_for_stream(9100, 3) == 9103
        assert effective_writer_fps(cfg, probe) == 30

    def test_build_detection_run_uses_model_backed_preprocess_group(self):
        from utils.config import AppConfig, ModelConfig
        from utils.model_family import ModelFamily
        from utils.pipeline import RtspProbe, RuntimeModules, build_detection_run

        class FakeGraph:
            def __init__(self, name):
                self.name = name
                self.added = []

            def add(self, node):
                self.added.append(node)

            def build(self, *args, **kwargs):
                return "fake-run"

        class FakeModelOptions:
            def __init__(self):
                self.preprocess = type(
                    "FakePreprocess",
                    (),
                    {
                        "kind": None,
                        "color_convert": type("FakeColorConvert", (), {"input_format": None})(),
                        "input_max_width": 0,
                        "input_max_height": 0,
                        "input_max_depth": 0,
                    },
                )()
                self.decode_type = None
                self.score_threshold = 0.0
                self.nms_iou_threshold = 0.0
                self.top_k = 0
                self.boxdecode_original_width = 0
                self.boxdecode_original_height = 0

        class FakeRunOptions:
            def __init__(self):
                self.queue_depth = 0
                self.overflow_policy = None
                self.output_memory = None

        class FakeTensor:
            @staticmethod
            def from_numpy(*args, **kwargs):
                return ("seed", args, kwargs)

        class FakeModel:
            def __init__(self, path, opt):
                self.path = path
                self.opt = opt

            def input_appsrc_options(self, tensor_mode):
                class InputOpt:
                    payload_type = ""
                    format = ""
                    width = 0
                    height = 0
                    depth = 0

                return InputOpt()

            def preprocess(self):
                return ("model_preprocess", self)

        class FakeNodes:
            @staticmethod
            def input(opt):
                return ("input", opt)

            @staticmethod
            def preproc(_opt):
                raise AssertionError("build_detection_run should use model.preprocess()")

            @staticmethod
            def sima_box_decode(model, **kwargs):
                return ("boxdecode", model, kwargs)

            @staticmethod
            def output():
                return ("output",)

        class FakeGroups:
            @staticmethod
            def mla(model):
                return ("mla", model)

        class FakePyneat:
            ModelOptions = FakeModelOptions
            RunOptions = FakeRunOptions
            Tensor = FakeTensor

            class OverflowPolicy:
                KeepLatest = "keep-latest"

            class OutputMemory:
                Owned = "owned"

            class RunMode:
                Async = "async"

            class InputKind:
                Image = "image"

            class PreprocessColorFormat:
                RGB = "rgb"

            class BoxDecodeType:
                YoloV8 = "yolov8"

            class PixelFormat:
                RGB = "rgb"

            class PayloadType:
                Image = "image"

            class TensorMemory:
                EV74 = "ev74"

            nodes = FakeNodes()
            groups = FakeGroups()

            def __init__(self):
                self.last_graph = None

            def Graph(self, name):
                self.last_graph = FakeGraph(name)
                return self.last_graph

            def Model(self, path, opt):
                return FakeModel(path, opt)

        class FakeNp:
            uint8 = "uint8"

            @staticmethod
            def zeros(shape, dtype=None):
                return {"shape": shape, "dtype": dtype}

        cfg = AppConfig(
            model=ModelConfig("assets/models/yolo_v8m_mpk.tar.gz"),
            rtsp_urls=["rtsp://127.0.0.1:8554/src1"],
            insight_host="127.0.0.1",
        )
        probe = RtspProbe(width=1280, height=720, fps=30)
        runtime = RuntimeModules(cv2=None, np=FakeNp(), pyneat=FakePyneat())

        built = build_detection_run(runtime, cfg, ModelFamily.YOLOV8, probe)

        added_kinds = [node[0] for node in runtime.pyneat.last_graph.added]
        assert built.run == "fake-run"
        assert added_kinds == ["input", "model_preprocess", "mla", "boxdecode", "output"]

    def test_startup_trace_enabled_from_env(self, monkeypatch: pytest.MonkeyPatch):
        from utils.workers import startup_trace_enabled_from_env

        monkeypatch.delenv("SIMA_INSIGHT_STARTUP_TRACE", raising=False)
        assert startup_trace_enabled_from_env() is False

        monkeypatch.setenv("SIMA_INSIGHT_STARTUP_TRACE", "TRUE")
        assert startup_trace_enabled_from_env() is True

        monkeypatch.setenv("SIMA_INSIGHT_STARTUP_TRACE", "banana")
        assert startup_trace_enabled_from_env() is False

    def test_source_runtime_contract_matches_cpp(self):
        from utils.workers import (
            SOURCE_PULL_TIMEOUT_MS,
            SOURCE_STARTUP_PULL_TIMEOUT_MS,
            SOURCE_STARTUP_STAGGER_S,
        )
        from utils.pipeline import SOURCE_OUTPUT_EVERY_N, SOURCE_RUN_QUEUE_DEPTH

        assert SOURCE_STARTUP_PULL_TIMEOUT_MS == 50000
        assert SOURCE_PULL_TIMEOUT_MS == 10000
        assert SOURCE_STARTUP_STAGGER_S == pytest.approx(0.5)
        assert SOURCE_RUN_QUEUE_DEPTH == 4
        assert SOURCE_OUTPUT_EVERY_N == 1


class TestSampleUtils:
    def test_parse_bbox_payload_normalizes_yolov8_boxes(self):
        from utils.sample_utils import parse_bbox_payload

        payload = bytes(
            [
                1, 0, 0, 0,
                246, 255, 255, 255,
                20, 0, 0, 0,
                140, 0, 0, 0,
                50, 0, 0, 0,
            ]
        ) + struct.pack("<f", 0.9) + bytes([3, 0, 0, 0])

        boxes = parse_bbox_payload(payload, 100, 80)

        assert len(boxes) == 1
        box = boxes[0]
        assert box["x1"] == pytest.approx(0.0)
        assert box["y1"] == pytest.approx(20.0)
        assert box["x2"] == pytest.approx(100.0)
        assert box["y2"] == pytest.approx(70.0)
        assert box["class_id"] == 3

    def test_build_insight_detection_payload_builds_objects(self):
        from utils.sample_utils import Detection, build_insight_detection_payload

        detections = [Detection(-5.0, 10.0, 35.0, 50.0, 0.8, 2)]
        payload = build_insight_detection_payload(
            detections,
            30,
            40,
            ["person", "bicycle", "car"],
        )

        assert len(payload.objects) == 1
        obj = payload.objects[0]
        assert obj["id"] == "obj_1"
        assert obj["label"] == "car"
        assert obj["confidence"] == pytest.approx(0.8)
        assert obj["bbox"] == [0.0, 10.0, 30.0, 30.0]


class TestWorkers:
    def test_latest_frame_mailbox_deduplicates_ready_notifications(self):
        from utils.workers import LatestFrameMailbox, ReadyStreamQueue

        ready = ReadyStreamQueue()
        mailbox = LatestFrameMailbox(stream_index=7, capacity=1)

        assert mailbox.push("frame-0", ready) == 0
        assert mailbox.push("frame-1", ready) == 1
        assert ready.pop_wait(0.0) == 7
        assert ready.pop_wait(0.0) is None

        assert mailbox.take_for_processing() == "frame-1"
        assert mailbox.push("frame-2", ready) == 0
        assert ready.pop_wait(0.0) is None

        mailbox.complete(ready)
        assert ready.pop_wait(0.0) == 7
