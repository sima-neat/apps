"""Unit tests for the graph-native 16-stream object detection Insight example."""

from __future__ import annotations

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
STRESS_DIR = EXAMPLE_DIR / "stress"
MODEL_PATH = "assets/models/yolo26m-det-int8-b1.tar.gz"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


def load_script_module(name: str):
    path = STRESS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extra_lines(block: str) -> list[str]:
    lines = []
    for line in block.strip().splitlines():
        if not line.strip():
            continue
        lines.append(line if line.startswith("  ") else f"  {line}")
    return lines


def write_config(
    tmp_path: Path,
    streams: list[str],
    workers: int = 1,
    decode_type: str | None = None,
    input_extra: str = "",
    inference_extra: str = "",
    output_extra: str = "",
) -> Path:
    stream_lines = "\n".join(f"  - {stream}" for stream in streams)
    model_lines = ["model:", f"  path: {MODEL_PATH}"]
    if decode_type is not None:
        model_lines.append(f"  decode_type: {decode_type}")
    lines = (
        model_lines
        + [
            "streams:",
            stream_lines,
            "input:",
            "  tcp: true",
            "  latency_ms: 100",
        ]
        + _extra_lines(input_extra)
        + [
            "inference:",
            f"  workers: {workers}",
        ]
        + _extra_lines(inference_extra)
        + [
            "output:",
            "  insight:",
            "    host: 127.0.0.1",
        ]
        + _extra_lines(output_extra)
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(lines), encoding="utf-8")
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
    @pytest.mark.parametrize(
        (
            "filename",
            "streams",
            "fps",
            "decoder_buffers",
            "decoder_input_buffers",
            "decoder_tuning",
            "queue_depth",
            "internal_queue_depth",
            "max_inflight",
            "fan_in_policy",
        ),
        [
            ("config-16x720p25.yaml", 16, 25, 8, 2, "auto", 16, 1, 1, "latest"),
            ("config-24x720p20.yaml", 24, 20, 16, 2, "auto", 4, 1, 4, "every_frame"),
            (
                "config-48x720p10.yaml",
                48,
                10,
                4,
                2,
                "throughput-low-latency",
                1,
                2,
                1,
                "latest",
            ),
        ],
    )
    def test_named_profiles_validate_portably(
        self,
        filename: str,
        streams: int,
        fps: int,
        decoder_buffers: int,
        decoder_input_buffers: int,
        decoder_tuning: str,
        queue_depth: int,
        internal_queue_depth: int,
        max_inflight: int,
        fan_in_policy: str,
    ):
        from main import effective_insight_visible_streams, load_app_config

        path = COMMON_DIR / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = load_app_config(path)

        assert len(cfg.rtsp_urls) == streams
        assert (cfg.input_width, cfg.input_height, cfg.input_fps) == (1280, 720, fps)
        assert cfg.decoder_buffers == decoder_buffers
        assert cfg.decoder_input_buffers == decoder_input_buffers
        assert cfg.decoder_tuning == decoder_tuning
        assert cfg.queue_depth == queue_depth
        assert cfg.internal_queue_depth == internal_queue_depth
        assert cfg.max_inflight_per_stream == max_inflight
        assert cfg.fan_in_policy == fan_in_policy
        assert effective_insight_visible_streams(cfg) == streams
        assert (cfg.video_port_base, cfg.video_port_base + streams - 1) == (
            9000,
            9000 + streams - 1,
        )
        assert (cfg.metadata_port_base, cfg.metadata_port_base + streams - 1) == (
            9100,
            9100 + streams - 1,
        )
        assert cfg.video_port_base + streams - 1 < cfg.metadata_port_base
        assert not Path(raw["model"]["path"]).is_absolute()
        assert not Path(raw["model"]["labels"]).is_absolute()

    def test_default_config_is_the_named_24x20_profile(self):
        default = yaml.safe_load((COMMON_DIR / "config.yaml").read_text(encoding="utf-8"))
        named = yaml.safe_load(
            (COMMON_DIR / "config-24x720p20.yaml").read_text(encoding="utf-8")
        )
        assert default == named

    def test_config_rejects_invalid_fan_in_policy(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="fan_in_policy: lossy_magic",
        )
        with pytest.raises(ValueError, match="fan_in_policy must be one of"):
            load_app_config(config_path)

    def test_config_rejects_overlapping_insight_port_ranges(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{i}" for i in range(4)],
        )
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "    host: 127.0.0.1",
                "    host: 127.0.0.1\n"
                "    video_port_base: 9000\n"
                "    metadata_port_base: 9002\n"
                "    max_visible_streams: 4",
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="port ranges overlap"):
            load_app_config(config_path)

    def test_load_app_config_accepts_twenty_four_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 25)],
            workers=1,
            input_extra=textwrap.dedent(
                """
                  skip_rtsp_probe: true
                  width: 1280
                  height: 720
                  fps: 20
                """
            ).strip(),
        )

        cfg = load_app_config(config_path)

        assert cfg.model_path == str(tmp_path / MODEL_PATH)
        assert cfg.decode_type == "yolo26"
        assert len(cfg.rtsp_urls) == 24
        assert cfg.workers == 1
        assert cfg.queue_depth == 4
        assert cfg.max_inflight_per_stream == 4
        assert cfg.skip_rtsp_probe is True
        assert cfg.input_width == 1280
        assert cfg.input_height == 720
        assert cfg.input_fps == 20
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30

    def test_load_app_config_accepts_insight_visible_limit(self, tmp_path: Path):
        from main import (
            effective_insight_visible_streams,
            is_insight_visible_stream,
            load_app_config,
            should_send_metadata,
        )

        stream_lines = "\n".join(
            f"  - rtsp://127.0.0.1:8554/src{index}" for index in range(1, 25)
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""model:
  path: {MODEL_PATH}
streams:
{stream_lines}
input:
  tcp: true
  latency_ms: 100
  skip_rtsp_probe: true
  width: 1280
  height: 720
  fps: 20
inference:
  workers: 1
output:
  insight:
    host: 127.0.0.1
    max_visible_streams: 16
""",
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.insight_visible_streams == 16
        assert effective_insight_visible_streams(cfg) == 16
        assert is_insight_visible_stream(cfg, 15) is True
        assert is_insight_visible_stream(cfg, 16) is False
        assert should_send_metadata(cfg, 15) is True
        assert should_send_metadata(cfg, 16) is False

    def test_load_app_config_resolves_model_and_labels_from_config_directory(
        self, tmp_path: Path
    ):
        from main import load_app_config

        config_dir = tmp_path / "portable-bundle"
        config_dir.mkdir()
        config_path = config_dir / "app.yaml"
        config_path.write_text(
            """model:
  path: models/detector.mpk
  labels: labels/coco.txt
streams:
  - rtsp://127.0.0.1:8554/src1
output:
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.model_path == str(config_dir / "models" / "detector.mpk")
        assert cfg.labels_path == config_dir / "labels" / "coco.txt"

    def test_load_app_config_accepts_forty_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 41)],
            workers=1,
        )

        cfg = load_app_config(config_path)

        assert len(cfg.rtsp_urls) == 40

    def test_load_app_config_rejects_removed_output_paths(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            output_extra=textwrap.dedent(
                """
                  hidden_streams:
                    video_sink: dummy
                """
            ).strip(),
        )

        with pytest.raises(ValueError, match="output.hidden_streams was removed"):
            load_app_config(config_path)

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

    def test_load_app_config_accepts_decoder_tuning_and_aliases(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            input_extra=textwrap.dedent(
                """
                  decoder_buffers: 7
                  decoder_input_buffers: 2
                  decoder_tuning: throughput_low_latency
                  skip_rtsp_probe: true
                  width: 3840
                  height: 2160
                  fps: 30
                """
            ).strip(),
        )

        cfg = load_app_config(config_path)

        assert cfg.decoder_buffers == 7
        assert cfg.decoder_input_buffers == 2
        assert cfg.decoder_tuning == "throughput-low-latency"

    @pytest.mark.parametrize("depth", [-1, 33])
    def test_load_app_config_rejects_invalid_internal_queue_depth(
        self, tmp_path: Path, depth: int
    ):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra=f"internal_queue_depth: {depth}",
        )

        with pytest.raises(ValueError, match="inference.internal_queue_depth"):
            load_app_config(config_path)

    def test_load_app_config_rejects_too_many_streams(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 82)],
            workers=1,
        )

        with pytest.raises(ValueError, match="up to 80 streams"):
            load_app_config(config_path)

    def test_load_app_config_rejects_insight_visible_limit_above_stream_count(
        self, tmp_path: Path
    ):
        from main import load_app_config

        stream_lines = "\n".join(
            f"  - rtsp://127.0.0.1:8554/src{index}" for index in range(1, 5)
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"""model:
  path: {MODEL_PATH}
streams:
{stream_lines}
input:
  tcp: true
  latency_ms: 100
inference:
  workers: 1
output:
  insight:
    host: 127.0.0.1
    max_visible_streams: 16
""",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="cannot exceed stream count"):
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

    def test_load_app_config_rejects_non_shared_worker_count(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
            workers=2,
        )

        with pytest.raises(ValueError, match="set inference.workers to 1"):
            load_app_config(config_path)

    def test_load_app_config_validates_max_inflight_per_stream(self, tmp_path: Path):
        from main import load_app_config

        tuned_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="  max_inflight_per_stream: 4",
        )
        assert load_app_config(tuned_path).max_inflight_per_stream == 4

        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        invalid_path = write_config(
            invalid_dir,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="  max_inflight_per_stream: 0",
        )
        with pytest.raises(ValueError, match="max_inflight_per_stream must be > 0"):
            load_app_config(invalid_path)

    def test_load_app_config_rejects_skip_probe_without_caps(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            input_extra="  skip_rtsp_probe: true",
        )

        with pytest.raises(ValueError, match="skip_rtsp_probe requires"):
            load_app_config(config_path)

    def test_load_app_config_rejects_fps_scheduler_knobs(self, tmp_path: Path):
        from main import load_app_config

        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            inference_extra="  target_fps: 15",
        )

        with pytest.raises(ValueError, match="target_fps is not supported"):
            load_app_config(config_path)

    def test_validate_config_only_reports_graph_native_settings(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [
                "rtsp://127.0.0.1:8554/src1",
                "rtsp://127.0.0.1:8554/src2",
            ],
            workers=1,
            inference_extra="  queue_depth: 4",
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
        assert "workers=1" in result.stdout
        assert "queue_depth=4" in result.stdout
        assert "max_inflight_per_stream=4" in result.stdout
        assert "fan_in_policy=latest" in result.stdout
        assert "insight_visible_streams=2" in result.stdout
        assert "decoder_admission=core" in result.stdout


class TestRuntimeOptions:
    def test_source_options_keep_decoded_handoff_device_visible(self, monkeypatch):
        import main

        class FakeRtspDecodedInputOptions:
            def __init__(self):
                self.output_caps = SimpleNamespace()
                self.h264_parse_config_interval = -1
                self.h264_fps = -1
                self.h264_width = -1
                self.h264_height = -1
                self.buffer_mode = ""
                self.sync_mode = False
                self.sima_allocator_type = 2

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

        opt, fps, width, height = main.make_source_options(
            cfg, cfg.rtsp_urls[0], fps=30, width=640, height=480
        )

        assert opt.out_format == "NV12"
        assert opt.decoder_raw_output is True
        assert opt.decoder_next_element == "CVU"
        assert opt.auto_caps_from_stream is True
        assert opt.num_buffers == main.DEFAULT_DECODER_BUFFERS
        assert opt.output_caps.enable is True
        assert opt.output_caps.format == "NV12"
        assert opt.output_caps.width == 640
        assert opt.output_caps.height == 480
        assert opt.output_caps.fps == 30
        assert opt.output_caps.memory == "Any"
        assert (fps, width, height) == (30, 640, 480)

    def test_source_options_explicit_caps_override_probe_caps(self, monkeypatch):
        import main

        class FakeRtspDecodedInputOptions:
            def __init__(self):
                self.output_caps = SimpleNamespace()
                self.h264_parse_config_interval = -1
                self.h264_fps = -1
                self.h264_width = -1
                self.h264_height = -1
                self.buffer_mode = ""
                self.sync_mode = False
                self.sima_allocator_type = 2

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
            input_width=1280,
            input_height=720,
            input_fps=20,
            decoder_tuning="throughput-low-latency",
        )

        opt, fps, width, height = main.make_source_options(
            cfg, cfg.rtsp_urls[0], fps=30, width=640, height=480
        )

        assert opt.output_caps.width == 1280
        assert opt.output_caps.height == 720
        assert opt.output_caps.fps == 20
        assert (fps, width, height) == (20, 1280, 720)

    def test_realtime_options_matches_cpp_runtime_defaults(self, monkeypatch):
        import main

        class FakeRunOptions:
            pass

        fake_pyneat = SimpleNamespace(
            RunOptions=FakeRunOptions,
            RunPreset=SimpleNamespace(Realtime="Realtime"),
            OverflowPolicy=SimpleNamespace(KeepLatest="KeepLatest"),
            OutputMemory=SimpleNamespace(ZeroCopy="ZeroCopy"),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        options = main.realtime_options(7)

        assert options.preset == "Realtime"
        assert options.queue_depth == 7
        assert options.overflow_policy == "KeepLatest"
        assert options.output_memory == "ZeroCopy"

    def test_graph_options_apply_internal_queue_depth_and_sync_mla(self, monkeypatch):
        import main

        class FakeGraphOptions:
            def __init__(self):
                self.advanced_execution = SimpleNamespace(
                    internal_queue_depth=None, inference_async=None
                )

        monkeypatch.setattr(main, "pyneat", SimpleNamespace(GraphOptions=FakeGraphOptions))

        options = main.graph_options(2)

        assert options.advanced_execution.internal_queue_depth == 2
        assert options.advanced_execution.inference_async is False

    def test_graph_options_require_sync_mla_public_surface(self, monkeypatch):
        import main

        class FakeGraphOptions:
            def __init__(self):
                self.advanced_execution = SimpleNamespace(internal_queue_depth=None)

        monkeypatch.setattr(main, "pyneat", SimpleNamespace(GraphOptions=FakeGraphOptions))

        with pytest.raises(RuntimeError, match="inference_async"):
            main.graph_options(2)

    def test_decode_options_apply_input_pool_and_tuning(self, monkeypatch):
        import main

        class FakeGraph:
            def __init__(self):
                self.nodes = []

            def add(self, node):
                self.nodes.append(node)

        class FakeInputOptions:
            pass

        class FakeDecodeOptions:
            pass

        fake_pyneat = SimpleNamespace(
            Graph=FakeGraph,
            InputOptions=FakeInputOptions,
            PayloadType=SimpleNamespace(Encoded="Encoded"),
            Format=SimpleNamespace(H264="H264", NV12="NV12"),
            SimaDecodeOptions=FakeDecodeOptions,
            SimaDecodeType=SimpleNamespace(H264="H264"),
            nodes=SimpleNamespace(
                input=lambda name, options: ("input", name, options),
                sima_decode=lambda options: options,
            ),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        source_options = SimpleNamespace(
            h264_width=1280,
            h264_height=720,
            h264_fps=20,
            fallback_h264_width=-1,
            fallback_h264_height=-1,
            fallback_h264_fps=-1,
            sima_allocator_type=2,
            decoder_name="decoder",
            decoder_raw_output=True,
            decoder_next_element="CVU",
            output_caps=SimpleNamespace(enable=False),
        )

        graph = main.make_rtsp_decoded_input(
            source_options,
            decoder_buffers=16,
            input_name="detector_h264",
            decoder_input_buffers=2,
            decoder_tuning="throughput-low-latency",
        )

        decode = graph.nodes[1]
        assert decode.num_buffers == 16
        assert decode.input_buffers == 2
        assert decode.decoder_tuning == "throughput-low-latency"

    def test_encoded_input_uses_the_same_16_mib_au_limit_as_cpp(self, monkeypatch):
        import main

        class FakeInputOptions:
            def __init__(self):
                self.memory_policy = None

        fake_pyneat = SimpleNamespace(
            InputOptions=FakeInputOptions,
            PayloadType=SimpleNamespace(Encoded="Encoded"),
            Format=SimpleNamespace(H264="H264"),
            InputMemoryPolicy=SimpleNamespace(SystemMemory="SystemMemory"),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        options = main.make_encoded_h264_input_options(False)

        assert options.max_bytes == main.MAX_ENCODED_AU_BYTES == 16 * 1024 * 1024
        assert options.memory_policy == "SystemMemory"

    def test_graph_realtime_link_stamps_stream_id(self, monkeypatch):
        import main

        class FakeGraphLinkOptions:
            pass

        fake_pyneat = SimpleNamespace(
            GraphLinkOptions=FakeGraphLinkOptions,
            GraphLinkPolicy=SimpleNamespace(
                RealtimeLatestByStream="latest-by-stream",
                RealtimeEveryFrameByStream="every-frame-by-stream",
            ),
        )
        monkeypatch.setattr(main, "pyneat", fake_pyneat)

        link = main.graph_realtime_link(3, "stream7")

        assert link.policy == "latest-by-stream"
        assert link.queue_depth == 3
        assert link.stream_id == "stream7"
        assert link.max_inflight_per_stream == 4

        tuned = main.graph_realtime_link(3, "stream7", 4)
        assert tuned.max_inflight_per_stream == 4

        every_frame = main.graph_realtime_link(3, "stream7", 4, "every_frame")
        assert every_frame.policy == "every-frame-by-stream"

    def test_stream_index_from_detection_validates_route_metadata(self):
        import main

        assert main.stream_index_from_detection(SimpleNamespace(stream_id="stream2"), 4) == 2
        assert main.stream_index_from_detection(SimpleNamespace(stream_id=""), 1) == 0
        with pytest.raises(RuntimeError, match="missing stream id"):
            main.stream_index_from_detection(SimpleNamespace(stream_id=""), 2)
        with pytest.raises(RuntimeError, match="invalid detection stream id"):
            main.stream_index_from_detection(SimpleNamespace(stream_id="streamx"), 4)
        with pytest.raises(RuntimeError, match="out of range"):
            main.stream_index_from_detection(SimpleNamespace(stream_id="stream4"), 4)


class TestHandoffGates:
    def test_stability_gate_selects_current_metadata_owner_over_stale_video_peer(self):
        gate = load_script_module("app16_insight_stability_gate")

        def candidate(
            peer_id: int,
            browser_report_at: str,
            metadata_sent_at: str | None,
        ) -> dict:
            metadata = {"messages_sent": peer_id * 100}
            if metadata_sent_at is not None:
                metadata["last_sent_at"] = metadata_sent_at
            return {
                "id": peer_id,
                "active": True,
                "connection_state": "connected",
                "data_channel_state": "open",
                "browser": {"inbound_rtp": {"frames_decoded": peer_id * 1000}},
                "last_browser_report_at": browser_report_at,
                "metadata": metadata,
            }

        stale_video_peer = candidate(
            1,
            "2026-07-12T23:10:05Z",
            "2026-07-12T23:09:55Z",
        )
        current_metadata_owner = candidate(
            2,
            "2026-07-12T23:10:04Z",
            "2026-07-12T23:10:06Z",
        )
        payload = {
            "channels": [
                {"channel": 7, "peers": [stale_video_peer, current_metadata_owner]}
            ]
        }

        assert gate.selected_browser_peers(payload)[7]["id"] == 2

    def test_stability_gate_falls_back_to_newest_browser_report_before_metadata(self):
        gate = load_script_module("app16_insight_stability_gate")
        payload = {
            "channels": [
                {
                    "channel": 3,
                    "peers": [
                        {
                            "id": 10,
                            "active": True,
                            "connection_state": "connected",
                            "data_channel_state": "open",
                            "browser": {"video": {}},
                            "last_browser_report_at": "2026-07-12T23:10:04Z",
                            "metadata": {"messages_sent": 0},
                        },
                        {
                            "id": 11,
                            "active": True,
                            "connection_state": "connected",
                            "data_channel_state": "open",
                            "browser": {"video": {}},
                            "last_browser_report_at": "2026-07-12T23:10:05Z",
                            "metadata": {"messages_sent": 0},
                        },
                    ],
                }
            ]
        }

        assert gate.selected_browser_peers(payload)[3]["id"] == 11

    def test_stability_gate_accounts_for_target_rates(self):
        gate = load_script_module("app16_insight_stability_gate")

        def ingest(packets: int, metadata: int):
            return {
                "active": True,
                "rtp": {"packets_received": packets},
                "metadata": {"active": True, "messages_received": metadata},
            }

        def peer(frames: int, metadata: int, report_time: str):
            return {
                "id": 7,
                "browser": {
                    "time": report_time,
                    "video": {"active": True},
                    "inbound_rtp": {"frames_decoded": frames},
                },
                "metadata": {"messages_sent": metadata},
            }

        record = gate.sample(
            [0],
            {0: ingest(100, 100)},
            {0: ingest(200, 200)},
            {0: peer(100, 100, "2026-07-13T08:18:00Z")},
            {0: peer(200, 200, "2026-07-13T08:18:05Z")},
            5.0,
            18.0,
            18.0,
        )

        assert record["passed"] is True
        assert record["channels"][0]["browser_video_fps"] == 20.0
        distribution = gate.rate_summary([18.0, 19.0, 20.0, 20.0, 21.0], 18.0)
        assert distribution["median"] == 20.0
        assert distribution["rate_misses"] == 0

    def test_stability_gate_uses_each_browser_report_interval_for_video_only(self):
        gate = load_script_module("app16_insight_stability_gate")

        def ingest(packets: int, metadata: int):
            return {
                "active": True,
                "rtp": {"packets_received": packets},
                "metadata": {"active": True, "messages_received": metadata},
            }

        def peer(frames: int, metadata: int, report_time: str):
            return {
                "id": 7,
                "browser": {
                    "time": report_time,
                    "video": {"active": True},
                    "inbound_rtp": {"frames_decoded": frames},
                },
                "metadata": {"messages_sent": metadata},
            }

        record = gate.sample(
            [0, 1],
            {0: ingest(100, 100), 1: ingest(100, 100)},
            {0: ingest(200, 149), 1: ingest(200, 149)},
            {
                0: peer(100, 100, "2026-07-13T08:18:00.000Z"),
                1: peer(100, 100, "2026-07-13T08:18:01.000Z"),
            },
            {
                0: peer(150, 149, "2026-07-13T08:18:05.000Z"),
                1: peer(140, 149, "2026-07-13T08:18:05.000Z"),
            },
            4.9,
            9.0,
            9.0,
        )

        assert record["passed"] is True
        first, staggered = record["channels"]
        assert first["browser_report_elapsed_s"] == 5.0
        assert staggered["browser_report_elapsed_s"] == 4.0
        assert first["browser_video_fps"] == 10.0
        assert staggered["browser_video_fps"] == 10.0
        assert first["browser_report_time_source"] == "browser.time"
        assert staggered["browser_report_time_source"] == "browser.time"
        assert first["browser_report_time_valid"] is True
        assert staggered["browser_report_time_valid"] is True
        assert first["ingest_metadata_fps"] == pytest.approx(10.0)
        assert staggered["egress_metadata_fps"] == pytest.approx(10.0)

    def test_stability_gate_falls_back_from_nonadvancing_browser_time(self):
        gate = load_script_module("app16_insight_stability_gate")
        previous = {
            "id": 7,
            "browser": {"time": "2026-07-13T08:18:00Z"},
            "last_browser_report_at": "2026-07-13T08:18:00.100Z",
        }
        current = {
            "id": 7,
            "browser": {"time": "2026-07-13T08:18:00Z"},
            "last_browser_report_at": "2026-07-13T08:18:04.100Z",
        }

        interval, source, valid = gate.browser_report_interval(previous, current, 4.9)

        assert interval == 4.0
        assert source == "last_browser_report_at"
        assert valid is True

    @pytest.mark.parametrize(
        ("before_time", "after_time"),
        [
            ("not-a-time", "still-not-a-time"),
            ("2026-07-13T08:18:00Z", "2026-07-13T08:18:00Z"),
            ("2026-07-13T08:18:01Z", "2026-07-13T08:18:00Z"),
        ],
    )
    def test_stability_gate_uses_wall_fallback_for_invalid_report_intervals(
        self, before_time: str, after_time: str
    ):
        gate = load_script_module("app16_insight_stability_gate")
        previous = {
            "id": 7,
            "browser": {"time": before_time},
            "last_browser_report_at": before_time,
        }
        current = {
            "id": 7,
            "browser": {"time": after_time},
            "last_browser_report_at": after_time,
        }

        assert gate.browser_report_interval(previous, current, 4.9) == (
            4.9,
            "wall_elapsed",
            False,
        )

    def test_stability_gate_invalid_report_clock_cannot_false_pass(self):
        gate = load_script_module("app16_insight_stability_gate")
        ingest_before = {
            "active": True,
            "rtp": {"packets_received": 100},
            "metadata": {"active": True, "messages_received": 100},
        }
        ingest_after = {
            "active": True,
            "rtp": {"packets_received": 200},
            "metadata": {"active": True, "messages_received": 200},
        }

        def peer(frames: int, metadata: int):
            return {
                "id": 7,
                "browser": {
                    "video": {"active": True},
                    "inbound_rtp": {"frames_decoded": frames},
                },
                "metadata": {"messages_sent": metadata},
            }

        record = gate.sample(
            [0],
            {0: ingest_before},
            {0: ingest_after},
            {0: peer(100, 100)},
            {0: peer(200, 200)},
            5.0,
            18.0,
            18.0,
        )

        channel = record["channels"][0]
        assert channel["browser_video_fps"] == 20.0
        assert channel["browser_report_time_valid"] is False
        assert channel["checks"]["browser_video_rate"] is False
        assert record["passed"] is False

    def test_stability_gate_does_not_accept_a_replaced_browser_peer(self):
        gate = load_script_module("app16_insight_stability_gate")

        def ingest(count: int):
            return {
                "active": True,
                "rtp": {"packets_received": count},
                "metadata": {"active": True, "messages_received": count},
            }

        def peer(peer_id: int, count: int, report_time: str):
            return {
                "id": peer_id,
                "browser": {
                    "time": report_time,
                    "video": {"active": True},
                    "inbound_rtp": {"frames_decoded": count},
                },
                "metadata": {"messages_sent": count},
            }

        record = gate.sample(
            [0],
            {0: ingest(100)},
            {0: ingest(200)},
            {0: peer(7, 100, "2026-07-13T08:18:00Z")},
            {0: peer(8, 200, "2026-07-13T08:18:05Z")},
            5.0,
            18.0,
            18.0,
        )

        channel = record["channels"][0]
        assert record["passed"] is False
        assert channel["checks"]["operator_peer_stable"] is False
        assert channel["checks"]["browser_video_rate"] is False
        assert channel["checks"]["browser_metadata_rate"] is False
        assert channel["browser_report_time_source"] == "wall_elapsed"
        assert channel["browser_report_time_valid"] is False

    def test_visual_gate_loads_exact_channel_identity_manifest(self, tmp_path: Path):
        gate = load_script_module("app16_insight_visual_gate")
        manifest_path = tmp_path / "identity.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "width": 1280,
                    "height": 720,
                    "marker": {"x": 8, "y": 8, "width": 24, "height": 24},
                    "temporal": {
                        "x": 8,
                        "y": 70,
                        "bit_width": 8,
                        "bit_height": 20,
                        "bit_stride": 12,
                        "bits": 12,
                        "period_frames": 2400,
                        "fps": 20,
                        "luma_threshold": 128,
                        "sync_tolerance_frames": 6,
                    },
                    "channels": {
                        "0": {"rgb": [230, 34, 34]},
                        "1": {"rgb": [34, 230, 230]},
                    },
                }
            ),
            encoding="utf-8",
        )

        identity = gate.load_identity_manifest(manifest_path, [0, 1], 1280, 720)

        assert identity["colors"][0] == [230.0, 34.0, 34.0]
        assert identity["temporal"]["period_frames"] == 2400
        assert gate.rgb_distance([230.0, 35.0, 35.0], identity["colors"][0]) < 2.0
        with pytest.raises(ValueError, match="no valid RGB marker for channel 2"):
            gate.load_identity_manifest(manifest_path, [0, 2], 1280, 720)

    def test_visual_gate_rejects_backward_or_stale_temporal_metadata(self):
        gate = load_script_module("app16_insight_visual_gate")
        temporal = {
            "period_frames": 2400,
            "fps": 20.0,
            "sync_tolerance_frames": 6.0,
        }

        def metadata(frame: int, count: int):
            pts_ns = frame * 1_000_000_000 // 20
            return {
                "count": count,
                "ptsNs": pts_ns,
                "rtpTimestamp": (pts_ns * 90000 // 1_000_000_000) & 0xFFFFFFFF,
            }

        def sample(code: int, frame: int, count: int, callback_count: int, rtp_translation=0):
            row = metadata(frame, count)
            return {
                "temporalCode": code,
                "metadata": row,
                "videoFrame": {
                    "callbackCount": callback_count,
                    "rtpTimestamp": None,
                },
                "videoRtp": {
                    "rtpTimestamp": (row["rtpTimestamp"] + rtp_translation) & 0xFFFFFFFF,
                    "source": "receiver-synchronization-source",
                },
            }

        # The publisher was already 40 seconds (800 frames) into the fixture
        # when App16 joined, and WebRTC applied an arbitrary RTP translation.
        # Both stable offsets are legitimate; translation drift is not.
        rtp_translation = 1_100_000_000
        forward = [
            sample(0, 800, 10, 10, rtp_translation),
            sample(20, 820, 30, 30, rtp_translation),
            sample(40, 840, 50, 50, rtp_translation),
        ]
        forward_result = gate.analyze_temporal_samples(
            forward, [0.0, 1.0, 2.0], temporal
        )
        assert forward_result["media_origin_offset_stable"]
        assert forward_result["media_origin_offsets_frames"] == [800, 800, 800]
        assert forward_result["video_metadata_rtp_translation_baseline"] == rtp_translation
        assert forward_result["video_metadata_rtp_translation_deviations"] == [0, 0, 0]
        assert forward_result["video_metadata_rtp_translation_range_ticks"] == 0
        assert forward_result["video_metadata_rtp_translation_stable"]
        assert forward_result["passed"]

        backward = [
            forward[0],
            forward[1],
            sample(10, 840, 50, 50, rtp_translation),
        ]
        assert not gate.analyze_temporal_samples(backward, [0.0, 1.0, 2.0], temporal)[
            "passed"
        ]

        stale = [
            forward[0],
            forward[1],
            sample(40, 820, 30, 50, rtp_translation + 20 * 4500),
        ]
        assert not gate.analyze_temporal_samples(stale, [0.0, 1.0, 2.0], temporal)["passed"]

        rtp_translation_jump = [
            forward[0],
            forward[1],
            sample(40, 840, 50, 50, rtp_translation + 100 * 4500),
        ]
        mismatch_result = gate.analyze_temporal_samples(
            rtp_translation_jump, [0.0, 1.0, 2.0], temporal
        )
        assert mismatch_result["media_origin_offset_stable"]
        assert mismatch_result["video_metadata_rtp_translation_range_ticks"] == 100 * 4500
        assert not mismatch_result["video_metadata_rtp_translation_stable"]
        assert not mismatch_result["passed"]

    def test_visual_gate_hooks_local_and_remote_data_channels(self, monkeypatch):
        gate = load_script_module("app16_insight_visual_gate")
        assert "createDataChannel" in gate.CANVAS_HOOK_JS
        assert "addEventListener('datachannel'" in gate.CANVAS_HOOK_JS
        assert "setRemoteDescription" in gate.CANVAS_HOOK_JS
        assert "requestVideoFrameCallback" in gate.CANVAS_HOOK_JS
        assert "window.__app16Peers.push(this)" in gate.CANVAS_HOOK_JS

        temporal = {
            "x": 8,
            "y": 70,
            "bit_width": 8,
            "bit_height": 20,
            "bit_stride": 12,
            "bits": 12,
            "period_frames": 2400,
            "fps": 20,
            "luma_threshold": 128,
            "sync_tolerance_frames": 6,
        }
        sample_script = gate.sample_js([0], None, temporal)
        assert sample_script.count("bitCtx.drawImage(") == 1
        assert "getReceivers()" in sample_script
        assert "getSynchronizationSources()" in sample_script
        assert "presented-video-frame" in sample_script

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "app16_insight_visual_gate.py",
                "--viewer-url",
                "https://viewer",
                "--keep-target-on-success",
            ],
        )
        assert gate.parse_args().keep_target_on_success

    def test_identity_fixture_plan_is_unique_and_forces_no_b_frames(self):
        fixture = load_script_module("app16_make_identity_fixtures")
        colors = fixture.channel_colors(list(range(48)))
        args = SimpleNamespace(
            ffmpeg="ffmpeg",
            overwrite=True,
            input=Path("moving.mp4"),
            duration_seconds=120.0,
            fps=20,
            width=1280,
            height=720,
            font_file=Path("font.ttf"),
        )
        command = fixture.ffmpeg_command(
            args, 7, colors[7], Path("channel-07-1280x720p20-no-b.mp4")
        )

        assert len({tuple(rgb) for rgb in colors.values()}) == 48
        assert command[command.index("-bf") + 1] == "0"
        assert "bframes=0" in command[command.index("-x264-params") + 1]
        assert "text=CH07" in command[command.index("-vf") + 1]
        assert "floor(t*20)" in command[command.index("-vf") + 1]
        assert "enable=" in command[command.index("-vf") + 1]

    def test_identity_fixture_calibrates_decoded_marker_rgb(self, monkeypatch):
        fixture = load_script_module("app16_make_identity_fixtures")
        args = SimpleNamespace(ffmpeg="ffmpeg")
        marker = {"x": 8, "y": 8, "width": 2, "height": 1}
        captured = []

        def fake_check_output(command):
            captured.append(command)
            return bytes([10, 20, 30, 30, 40, 50])

        monkeypatch.setattr(fixture.subprocess, "check_output", fake_check_output)
        assert fixture.measure_decoded_marker_rgb(
            args, Path("encoded.mp4"), marker, sample_frames=1
        ) == [20, 30, 40]
        assert "crop=2:1:8:8" in captured[0]

        unique = {
            str(channel): {"rgb": rgb}
            for channel, rgb in fixture.channel_colors(list(range(48))).items()
        }
        fixture.require_unique_marker_colors(unique)
        unique["47"]["rgb"] = unique["0"]["rgb"]
        with pytest.raises(RuntimeError, match="not unique"):
            fixture.require_unique_marker_colors(unique)


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
    def test_send_metadata_noops_when_stream_metadata_disabled(self):
        from main import SourceRuntime, StreamProfile, send_metadata

        runtime = SourceRuntime(
            index=16,
            url="rtsp://127.0.0.1:8554/src17",
            metadata_sender=None,
            labels=["person"],
            source_options=None,
            profile=StreamProfile(False, 16),
            frame_w=100,
            frame_h=100,
            source_fps=30,
        )

        send_metadata(runtime, FakeSample(), [])

    def test_send_metadata_uses_object_detection_contract(self):
        from main import SourceRuntime, StreamProfile, send_metadata

        sender = FakeMetadataSender()
        runtime = SourceRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            metadata_sender=sender,
            labels=["person"],
            source_options=None,
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
