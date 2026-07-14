"""Unit tests for the high-density multi-stream object detector."""

from __future__ import annotations

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
MODEL_PATH = "assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


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
            ("config.yaml", 16, 25, 8, 2, "auto", 16, 1, 1, "latest"),
            (
                "config-24x720p20fps.yaml",
                24,
                20,
                16,
                2,
                "auto",
                4,
                1,
                4,
                "every_frame",
            ),
            (
                "config-48x720p10fps.yaml",
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

    def test_default_config_is_the_16x25_profile(self):
        default = yaml.safe_load((COMMON_DIR / "config.yaml").read_text(encoding="utf-8"))
        assert len(default["streams"]) == 16
        assert default["input"]["fps"] == 25

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

    def test_graph_options_apply_internal_queue_depth_and_async_mla(self, monkeypatch):
        import main

        class FakeGraphOptions:
            def __init__(self):
                self.advanced_execution = SimpleNamespace(
                    internal_queue_depth=None, inference_async=None
                )

        monkeypatch.setattr(main, "pyneat", SimpleNamespace(GraphOptions=FakeGraphOptions))

        options = main.graph_options(2)

        assert options.advanced_execution.internal_queue_depth == 2
        assert options.advanced_execution.inference_async is True

    def test_graph_options_require_async_mla_public_surface(self, monkeypatch):
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

    def test_send_metadata_preserves_missing_sample_identity(self):
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
        )

        send_metadata(runtime, SimpleNamespace(pts_ns=-1, frame_id=-1), [])

        _, _, timestamp_ms, frame_id = sender.calls[0]
        assert timestamp_ms == -1
        assert frame_id == ""
