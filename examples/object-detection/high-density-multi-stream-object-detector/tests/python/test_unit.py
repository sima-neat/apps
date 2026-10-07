"""Unit tests for the high-density multi-stream object detector."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tests.utils.config_cases import config_writer, load_example_main
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"
MODEL_PATH = "models/yolo26n-det-int8-b1.tar.gz"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "high_density_multi_stream_object_detector_main")

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
    runtime_extra: str = "",
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
        + (["runtime:"] + _extra_lines(runtime_extra) if runtime_extra else [])
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
            "max_inflight_per_stream",
            "max_inflight_total",
        ),
        [
            ("config.yaml", 16, 0, 8, 2, "throughput-low-latency", 16, 1, 1, 8),
            (
                "config-24x720p20fps.yaml",
                24,
                20,
                16,
                2,
                "throughput-low-latency",
                4,
                2,
                4,
                24,
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
                8,
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
        max_inflight_per_stream: int,
        max_inflight_total: int,
    ):
        path = COMMON_DIR / filename
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = main_module.load_app_config(path)

        assert len(cfg.rtsp_urls) == streams
        assert (cfg.input_width, cfg.input_height, cfg.input_fps) == (1280, 720, fps)
        assert cfg.decoder_buffers == decoder_buffers
        assert cfg.decoder_input_buffers == decoder_input_buffers
        assert cfg.decoder_tuning == decoder_tuning
        assert cfg.queue_depth == queue_depth
        assert cfg.internal_queue_depth == internal_queue_depth
        assert cfg.max_inflight_per_stream == max_inflight_per_stream
        assert cfg.max_inflight_total == max_inflight_total
        assert cfg.stream_detection_timeout_ms == 30_000
        assert cfg.no_detection_timeout_ms == 30_000
        assert main_module.effective_insight_visible_streams(cfg) == streams
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

    def test_default_config_uses_16_streams_and_auto_fps(self):
        default = yaml.safe_load((COMMON_DIR / "config.yaml").read_text(encoding="utf-8"))
        assert len(default["streams"]) == 16
        assert default["input"]["fps"] == 0
        assert default["input"]["skip_rtsp_probe"] is False

    def test_config_rejects_overlapping_insight_port_ranges(self, tmp_path: Path):
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
            main_module.load_app_config(config_path)

    def test_load_app_config_accepts_twenty_four_streams(self, tmp_path: Path):
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

        cfg = main_module.load_app_config(config_path)

        assert cfg.model_path == str(tmp_path / MODEL_PATH)
        assert cfg.decode_type == "yolo26"
        assert len(cfg.rtsp_urls) == 24
        assert cfg.workers == 1
        assert cfg.queue_depth == 4
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 8
        assert cfg.skip_rtsp_probe is True
        assert cfg.input_width == 1280
        assert cfg.input_height == 720
        assert cfg.input_fps == 20
        assert cfg.insight_host == "127.0.0.1"
        assert cfg.warmup_frames == 30
        assert cfg.stream_detection_timeout_ms == 30_000
        assert cfg.no_detection_timeout_ms == 30_000

    @pytest.mark.parametrize(
        ("setting", "message"),
        [
            ("no_detection_timeout_ms", "no_detection_timeout_ms must be > 0"),
            (
                "stream_detection_timeout_ms",
                "stream_detection_timeout_ms must be > 0",
            ),
        ],
    )
    def test_config_rejects_invalid_liveness_settings(
        self, tmp_path: Path, setting: str, message: str
    ):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            runtime_extra=f"{setting}: 0",
        )

        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(config_path)

    def test_config_rejects_legacy_fan_in_policy(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="fan_in_policy: realtime-latest-by-stream",
        )

        with pytest.raises(ValueError, match=r"fan_in_policy was removed.*connect\(\)/build\(\)"):
            main_module.load_app_config(config_path)

    def test_load_app_config_accepts_insight_visible_limit(self, tmp_path: Path):
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

        cfg = main_module.load_app_config(config_path)

        assert cfg.insight_visible_streams == 16
        assert main_module.effective_insight_visible_streams(cfg) == 16
        assert main_module.is_insight_visible_stream(cfg, 15) is True
        assert main_module.is_insight_visible_stream(cfg, 16) is False
        assert main_module.should_send_metadata(cfg, 15) is True
        assert main_module.should_send_metadata(cfg, 16) is False

    def test_load_app_config_resolves_model_and_labels_by_their_owners(
        self, tmp_path: Path
    ):
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

        cfg = main_module.load_app_config(config_path)

        assert cfg.model_path == str(config_dir / "models" / "detector.mpk")
        assert cfg.labels_path == Path("labels/coco.txt")

    def test_load_app_config_accepts_forty_streams(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 41)],
            workers=1,
        )

        cfg = main_module.load_app_config(config_path)

        assert len(cfg.rtsp_urls) == 40

    def test_load_app_config_rejects_removed_output_paths(self, tmp_path: Path):
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
            main_module.load_app_config(config_path)

    def test_load_app_config_accepts_yolov8_decode_type(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            decode_type="yolov8",
        )

        cfg = main_module.load_app_config(config_path)

        assert cfg.decode_type == "yolov8"

    def test_load_app_config_accepts_decoder_tuning_and_aliases(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            input_extra=textwrap.dedent(
                """
                  codec: hevc
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

        cfg = main_module.load_app_config(config_path)

        assert cfg.decoder_buffers == 7
        assert cfg.decoder_input_buffers == 2
        assert cfg.decoder_tuning == "throughput-low-latency"
        assert cfg.codec == "h265"

    @pytest.mark.parametrize("depth", [-1, 33])
    def test_load_app_config_rejects_invalid_internal_queue_depth(
        self, tmp_path: Path, depth: int
    ):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra=f"internal_queue_depth: {depth}",
        )

        with pytest.raises(ValueError, match="inference.internal_queue_depth"):
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_too_many_streams(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            [f"rtsp://127.0.0.1:8554/src{index}" for index in range(1, 82)],
            workers=1,
        )

        with pytest.raises(ValueError, match="up to 80 streams"):
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_insight_visible_limit_above_stream_count(
        self, tmp_path: Path
    ):
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
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_empty_streams(self, tmp_path: Path):
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
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_non_shared_worker_count(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
            workers=2,
        )

        with pytest.raises(ValueError, match="set inference.workers to 1"):
            main_module.load_app_config(config_path)

    def test_load_app_config_validates_max_inflight_limits(self, tmp_path: Path):
        tuned_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="  max_inflight_per_stream: 4\n  max_inflight_total: 12",
        )
        tuned = main_module.load_app_config(tuned_path)
        assert tuned.max_inflight_per_stream == 4
        assert tuned.max_inflight_total == 12

        invalid_per_stream_dir = tmp_path / "invalid_per_stream"
        invalid_per_stream_dir.mkdir()
        invalid_per_stream_path = write_config(
            invalid_per_stream_dir,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="  max_inflight_per_stream: 0",
        )
        with pytest.raises(ValueError, match="max_inflight_per_stream must be > 0"):
            main_module.load_app_config(invalid_per_stream_path)

        invalid_total_dir = tmp_path / "invalid_total"
        invalid_total_dir.mkdir()
        invalid_total_path = write_config(
            invalid_total_dir,
            ["rtsp://127.0.0.1:8554/src1"],
            inference_extra="  max_inflight_total: 0",
        )
        with pytest.raises(ValueError, match="max_inflight_total must be > 0"):
            main_module.load_app_config(invalid_total_path)

    def test_load_app_config_rejects_skip_probe_without_caps(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            input_extra="  skip_rtsp_probe: true",
        )

        with pytest.raises(ValueError, match="skip_rtsp_probe requires"):
            main_module.load_app_config(config_path)

    def test_load_app_config_rejects_fps_scheduler_knobs(self, tmp_path: Path):
        config_path = write_config(
            tmp_path,
            ["rtsp://127.0.0.1:8554/src1"],
            workers=1,
            inference_extra="  target_fps: 15",
        )

        with pytest.raises(ValueError, match="target_fps is not supported"):
            main_module.load_app_config(config_path)

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
        assert "max_inflight_total=8" in result.stdout
        assert "stream_detection_timeout_ms=30000" in result.stdout
        assert "no_detection_timeout_ms=30000" in result.stdout
        assert "insight_visible_streams=2" in result.stdout
        assert "decoder_admission=core" in result.stdout


class TestRuntimeOptions:
    def test_source_options_keep_decoded_handoff_device_visible(self, monkeypatch):
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
            RtspCodec=SimpleNamespace(H264="H264", H265="H265"),
            Format=SimpleNamespace(NV12="NV12"),
            CapsMemory=SimpleNamespace(Any="Any"),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        cfg = main_module.AppConfig(
            model_path=MODEL_PATH,
            labels_path=Path("labels.txt"),
            rtsp_urls=["rtsp://127.0.0.1:8554/src1"],
            codec="h265",
        )

        opt, fps, width, height = main_module.make_source_options(
            cfg, cfg.rtsp_urls[0], fps=30, width=640, height=480
        )

        assert opt.out_format == "NV12"
        assert opt.decoder_raw_output is True
        assert opt.decoder_next_element == "CVU"
        assert opt.codec == "H265"
        assert opt.dec_width == 640
        assert opt.dec_height == 480
        assert opt.source_fps == 0
        assert opt.dec_fps == 30
        assert opt.auto_caps_from_stream is True
        assert opt.num_buffers == main_module.DEFAULT_DECODER_BUFFERS
        assert opt.output_caps.enable is True
        assert opt.output_caps.format == "NV12"
        assert opt.output_caps.width == 640
        assert opt.output_caps.height == 480
        assert opt.output_caps.fps == 0
        assert opt.output_caps.memory == "Any"
        assert (fps, width, height) == (30, 640, 480)

    def test_source_options_explicit_caps_override_probe_caps(self, monkeypatch):
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
            RtspCodec=SimpleNamespace(H264="H264", H265="H265"),
            Format=SimpleNamespace(NV12="NV12"),
            CapsMemory=SimpleNamespace(Any="Any"),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        cfg = main_module.AppConfig(
            model_path=MODEL_PATH,
            labels_path=Path("labels.txt"),
            rtsp_urls=["rtsp://127.0.0.1:8554/src1"],
            input_width=1280,
            input_height=720,
            input_fps=20,
            decoder_tuning="throughput-low-latency",
        )
        monkeypatch.setattr(
            main_module,
            "probe_rtsp",
            lambda _url: pytest.fail("fully configured source must not be probed"),
        )

        opt, fps, width, height = main_module.make_source_options(cfg, cfg.rtsp_urls[0])

        assert opt.codec == "H264"
        assert opt.dec_width == 1280
        assert opt.dec_height == 720
        assert opt.source_fps == 20
        assert opt.dec_fps == 20
        assert opt.fallback_h264_width == 1280
        assert opt.fallback_h264_height == 720
        assert getattr(opt, "fallback_h264_fps", -1) == -1
        assert opt.output_caps.width == 1280
        assert opt.output_caps.height == 720
        assert opt.output_caps.fps == 20
        assert (fps, width, height) == (20, 1280, 720)

    def test_realtime_options_matches_cpp_runtime_defaults(self, monkeypatch):
        class FakeRunOptions:
            pass

        fake_pyneat = SimpleNamespace(
            RunOptions=FakeRunOptions,
            RunPreset=SimpleNamespace(Realtime="Realtime"),
            OverflowPolicy=SimpleNamespace(KeepLatest="KeepLatest"),
            OutputMemory=SimpleNamespace(ZeroCopy="ZeroCopy"),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        options = main_module.realtime_options(7)

        assert options.preset == "Realtime"
        assert options.queue_depth == 7
        assert options.overflow_policy == "KeepLatest"
        assert options.output_memory == "ZeroCopy"

    def test_video_sender_does_not_participate_in_shared_preroll(self, monkeypatch):
        class FakeVideoSenderOptions:
            @staticmethod
            def passthrough(codec):
                return SimpleNamespace(codec=codec, async_=True)

        monkeypatch.setattr(
            main_module,
            "pyneat",
            SimpleNamespace(
                VideoSenderOptions=FakeVideoSenderOptions,
                RtspCodec=SimpleNamespace(H264="h264", H265="h265"),
            ),
        )
        cfg = main_module.AppConfig(
            model_path=MODEL_PATH,
            labels_path=Path("labels.txt"),
            rtsp_urls=["rtsp://127.0.0.1:8554/src1"],
            codec="h265",
            insight_host="192.0.2.10",
            video_port_base=9200,
        )

        options = main_module.make_video_options(cfg, SimpleNamespace(index=3))

        assert options.async_ is False
        assert options.codec == "h265"
        assert options.host == "192.0.2.10"
        assert options.channel == 3
        assert options.video_port_base == 9200

    def test_graph_options_apply_internal_queue_depth_and_async_mla(self, monkeypatch):
        class FakeGraphOptions:
            def __init__(self):
                self.advanced_execution = SimpleNamespace(
                    internal_queue_depth=None, inference_async=None
                )

        monkeypatch.setattr(main_module, "pyneat", SimpleNamespace(GraphOptions=FakeGraphOptions))

        options = main_module.graph_options(2)

        assert options.advanced_execution.internal_queue_depth == 2
        assert options.advanced_execution.inference_async is True

    def test_graph_options_require_async_mla_public_surface(self, monkeypatch):
        class FakeGraphOptions:
            def __init__(self):
                self.advanced_execution = SimpleNamespace(internal_queue_depth=None)

        monkeypatch.setattr(main_module, "pyneat", SimpleNamespace(GraphOptions=FakeGraphOptions))

        with pytest.raises(RuntimeError, match="inference_async"):
            main_module.graph_options(2)

    def test_decode_options_apply_input_pool_and_tuning(self, monkeypatch):
        class FakeGraph:
            def __init__(self, _name=""):
                self.nodes = []

            def add(self, node):
                self.nodes.append(node)

        class FakeDecodeOptions:
            pass

        fake_pyneat = SimpleNamespace(
            Graph=FakeGraph,
            Format=SimpleNamespace(NV12="NV12"),
            SimaDecodeOptions=FakeDecodeOptions,
            SimaDecodeType=SimpleNamespace(H264="H264", H265="H265"),
            RtspCodec=SimpleNamespace(H264="H264", H265="H265"),
            nodes=SimpleNamespace(
                sima_decode=lambda options: options,
                output=lambda name: ("output", name),
            ),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        source_options = SimpleNamespace(
            codec="H264",
            dec_width=1280,
            dec_height=720,
            source_fps=0,
            dec_fps=20,
            sima_allocator_type=2,
            decoder_name="decoder",
            decoder_raw_output=True,
            decoder_next_element="CVU",
            output_caps=SimpleNamespace(enable=False),
        )

        graph = main_module.make_decoder(
            source_options,
            decoder_buffers=16,
            decoder_input_buffers=2,
            decoder_tuning="throughput-low-latency",
        )

        decode = graph.nodes[0]
        assert decode.dec_width == 1280
        assert decode.dec_height == 720
        assert decode.dec_fps == 20
        assert decode.num_buffers == 16
        assert decode.input_buffers == 2
        assert decode.decoder_tuning == "throughput-low-latency"
        assert decode.memory_opt is True
        assert graph.nodes[1] == ("output", "detector_frame")

        default_graph = main_module.make_decoder(
            source_options,
            decoder_buffers=8,
            decoder_input_buffers=2,
            decoder_tuning="auto",
        )
        assert default_graph.nodes[0].memory_opt is False

    def test_graph_realtime_link_stamps_stream_id(self, monkeypatch):
        class FakeGraphLinkOptions:
            pass

        fake_pyneat = SimpleNamespace(
            GraphLinkOptions=FakeGraphLinkOptions,
            GraphLinkPolicy=SimpleNamespace(
                RealtimeLatestByStream="latest-by-stream",
            ),
        )
        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)

        link = main_module.graph_realtime_link(3, "stream7")

        assert link.policy == "latest-by-stream"
        assert link.queue_depth == 3
        assert link.stream_id == "stream7"
        assert link.max_inflight_per_stream == 4
        assert link.max_inflight_total == 8

        tuned = main_module.graph_realtime_link(3, "stream7", 4, 12)
        assert tuned.max_inflight_per_stream == 4
        assert tuned.max_inflight_total == 12

    def test_stream_index_from_detection_validates_route_metadata(self):
        assert main_module.stream_index_from_detection(SimpleNamespace(stream_id="stream2"), 4) == 2
        assert main_module.stream_index_from_detection(SimpleNamespace(stream_id=""), 1) == 0
        with pytest.raises(RuntimeError, match="missing stream id"):
            main_module.stream_index_from_detection(SimpleNamespace(stream_id=""), 2)
        with pytest.raises(RuntimeError, match="invalid detection stream id"):
            main_module.stream_index_from_detection(SimpleNamespace(stream_id="streamx"), 4)
        with pytest.raises(RuntimeError, match="out of range"):
            main_module.stream_index_from_detection(SimpleNamespace(stream_id="stream4"), 4)


class TestRuntimeDelivery:
    def test_unexpected_clean_detector_close_is_not_reported_as_success(self):
        class ClosedRun:
            def pull(self, _name, _timeout_ms):
                return None

            def running(self):
                return False

            def last_error(self):
                return ""

        main_module._STOP_REQUESTED = False
        source = main_module.SourceRuntime(
            0,
            "rtsp://src0",
            None,
            [],
            None,
            main_module.StreamProfile(False, 0),
            1280,
            720,
            20,
        )
        app = main_module.AppRuntime(model=None, graph=None, run=ClosedRun(), sources=[source])
        cfg = SimpleNamespace(
            initial_detection_timeout_ms=1000,
            stream_detection_timeout_ms=1000,
            no_detection_timeout_ms=1000,
        )
        with pytest.raises(RuntimeError, match="detections output closed unexpectedly"):
            main_module.pull_detections(app, cfg, main_module.AggregateProfile(False, 0))

    def test_target_completion_cannot_hide_latched_starvation(self, monkeypatch):
        detections = [0, 0, 1, 1, 0, 1]

        class FakeRun:
            def pull(self, _name, _timeout_ms):
                return detections.pop(0)

        target_checks = 0

        def reached_target(_sources):
            nonlocal target_checks
            target_checks += 1
            return not detections

        monkeypatch.setattr(
            main_module, "stream_index_from_detection", lambda sample, _count: sample
        )
        monkeypatch.setattr(main_module, "complete_detection", lambda *_args: None)
        monkeypatch.setattr(main_module, "target_reached", reached_target)
        monotonic_values = iter((0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.9, 1.4, 1.4))
        monkeypatch.setattr(main_module.time, "monotonic", lambda: next(monotonic_values))
        main_module._STOP_REQUESTED = False
        app = main_module.AppRuntime(
            model=None, graph=None, run=FakeRun(), sources=[object(), object()]
        )
        cfg = SimpleNamespace(
            initial_detection_timeout_ms=10_000,
            stream_detection_timeout_ms=1000,
            no_detection_timeout_ms=10_000,
        )

        with pytest.raises(
            RuntimeError, match="timed out waiting for detector progress from streams: 1"
        ):
            main_module.pull_detections(app, cfg, object())
        assert target_checks == 6

    def test_detection_watchdog_tracks_deadlines(self):
        watchdog = main_module.DetectionWatchdog(
            3,
            priming_observations=2,
            startup_timeout_s=10.0,
            stream_timeout_s=20.0,
            no_progress_timeout_s=50.0,
            start=0.0,
        )
        watchdog.observe(0, 1.0)
        watchdog.observe(0, 1.1)
        watchdog.observe(2, 2.0)
        watchdog.observe(1, 3.0)
        watchdog.observe(1, 3.1)
        assert not watchdog.check(9.99)
        startup_failure = watchdog.check(10.0)
        assert startup_failure.kind is main_module.DetectionFailureKind.STARTUP
        assert startup_failure.streams == (2,)

        watchdog.observe(2, 10.0)
        assert not watchdog.startup_complete()
        late_startup_failure = watchdog.check(10.0)
        assert late_startup_failure.kind is main_module.DetectionFailureKind.STARTUP
        assert late_startup_failure.streams == (2,)

        watchdog = main_module.DetectionWatchdog(
            3,
            priming_observations=2,
            startup_timeout_s=100.0,
            stream_timeout_s=5.0,
            no_progress_timeout_s=50.0,
            start=0.0,
        )
        watchdog.observe(0, 1.0)
        watchdog.observe(0, 1.1)
        watchdog.observe(2, 2.0)
        watchdog.observe(1, 10.1)
        watchdog.observe(1, 10.2)
        watchdog.observe(0, 10.5)
        watchdog.observe(2, 10.5)
        assert watchdog.startup_complete()
        watchdog.observe(0, 14.0)
        watchdog.observe(2, 14.1)
        assert not watchdog.check(15.49)
        watchdog.observe(0, 15.5)
        watchdog.observe(1, 15.5)
        starvation = watchdog.check(15.5)
        assert starvation.kind is main_module.DetectionFailureKind.STREAM_STARVATION
        assert starvation.streams == (1,)

        global_stall = watchdog.check(65.5)
        assert global_stall.kind is main_module.DetectionFailureKind.GLOBAL_STALL
        assert global_stall.streams == ()

        startup_stall_watchdog = main_module.DetectionWatchdog(
            3,
            priming_observations=2,
            startup_timeout_s=100.0,
            stream_timeout_s=20.0,
            no_progress_timeout_s=5.0,
            start=0.0,
        )
        startup_stall_watchdog.observe(0, 1.0)
        assert not startup_stall_watchdog.check(5.999)
        startup_stall = startup_stall_watchdog.check(6.0)
        assert startup_stall.kind is main_module.DetectionFailureKind.GLOBAL_STALL
        assert startup_stall.streams == ()

        recovered_stall_watchdog = main_module.DetectionWatchdog(
            1,
            priming_observations=1,
            startup_timeout_s=100.0,
            stream_timeout_s=20.0,
            no_progress_timeout_s=5.0,
            start=0.0,
        )
        recovered_stall_watchdog.observe(0, 5.0)
        recovered_stall = recovered_stall_watchdog.check(5.0)
        assert recovered_stall.kind is main_module.DetectionFailureKind.GLOBAL_STALL
        assert recovered_stall.streams == ()

    def test_detection_watchdog_allows_sustained_48_stream_scheduler_skew(self):
        stream_count = 48
        watchdog = main_module.DetectionWatchdog(
            stream_count,
            priming_observations=2,
            startup_timeout_s=60.0,
            stream_timeout_s=5.0,
            no_progress_timeout_s=30.0,
            start=0.0,
        )
        for stream_index in range(stream_count):
            watchdog.observe(stream_index, stream_index * 0.01)
            watchdog.observe(stream_index, stream_index * 0.01 + 0.001)
        assert watchdog.startup_complete()

        now = 1.0
        for _ in range(10):
            watchdog.observe(0, now)
            now += 0.01
            for _ in range(5):
                for stream_index in range(1, stream_count):
                    watchdog.observe(stream_index, now)
                    now += 0.01

        assert not watchdog.check(now)

    def test_source_topology_connects_encoded_video_with_a_distinct_latest_link(
        self, monkeypatch
    ):
        class FakeGraph:
            def __init__(self, name=""):
                self.name = name

            def set_name(self, name):
                self.name = name

        class FakeGraphLinkOptions:
            def __init__(self):
                self.policy = "default"
                self.queue_depth = 16
                self.stream_id = ""
                self.max_inflight_per_stream = -1
                self.max_inflight_total = -1

        class RecordingGraph:
            def __init__(self):
                self.connections = []

            def connect(self, source, destination, options=None):
                self.connections.append((source, destination, options))

        fake_pyneat = SimpleNamespace(
            GraphLinkOptions=FakeGraphLinkOptions,
            GraphLinkPolicy=SimpleNamespace(RealtimeLatestByStream="latest-by-stream"),
            groups=SimpleNamespace(video_sender=lambda _options: FakeGraph("video_sender")),
        )
        rtsp_graph = object()
        rtsp_calls = []

        def make_rtsp_encoded_input(options):
            rtsp_calls.append(options)
            return rtsp_graph

        monkeypatch.setattr(main_module, "pyneat", fake_pyneat)
        monkeypatch.setattr(main_module, "make_rtsp_encoded_input", make_rtsp_encoded_input)
        monkeypatch.setattr(main_module, "make_decoder", lambda *_args: "decoder")
        monkeypatch.setattr(main_module, "graph_realtime_link", lambda *_args: "latest")
        monkeypatch.setattr(
            main_module, "make_video_options", lambda *_args: SimpleNamespace(video_port=9000)
        )

        cfg = main_module.AppConfig("model", Path("labels"), ["rtsp://src0"])
        source_options = object()
        source = main_module.SourceRuntime(
            0,
            "rtsp://src0",
            None,
            [],
            source_options,
            main_module.StreamProfile(False, 0),
            1280,
            720,
            20,
        )
        graph = RecordingGraph()
        app = main_module.AppRuntime(None, graph, None, [source])

        main_module.connect_source_graph(app, cfg, source, "detector")

        assert rtsp_calls == [source_options]
        assert len(graph.connections) == 3
        assert graph.connections[0][0] is rtsp_graph
        assert graph.connections[0][1:] == ("decoder", None)
        assert graph.connections[1] == ("decoder", "detector", "latest")
        encoded_sender = graph.connections[2][1]
        video_link = graph.connections[2][2]
        assert graph.connections[2][0] is rtsp_graph
        assert encoded_sender.name == "encoded_insight_video_sender_0"
        assert video_link.policy == "latest-by-stream"
        assert video_link.queue_depth == 16
        assert video_link.stream_id == ""
        assert video_link.max_inflight_per_stream == -1
        assert video_link.max_inflight_total == -1
        assert source.video_port == 9000


class FakeMetadataSender:
    def __init__(self, accepted: bool = True):
        self.calls = []
        self.accepted = accepted

    def send_raw_json(self, payload):
        self.calls.append(payload)
        return self.accepted


class FakeSample:
    frame_id = 42
    pts_ns = 1_234_000_000
    dts_ns = 1_200_000_000
    duration_ns = 50_000_000
    input_seq = 17
    orig_input_seq = 12
    stream_id = "stream0"


class TestMetadata:
    def test_send_metadata_noops_when_stream_metadata_disabled(self):
        runtime = main_module.SourceRuntime(
            index=16,
            url="rtsp://127.0.0.1:8554/src17",
            metadata_sender=None,
            labels=["person"],
            source_options=None,
            profile=main_module.StreamProfile(False, 16),
            frame_w=100,
            frame_h=100,
            source_fps=30,
        )

        main_module.send_metadata(runtime, FakeSample(), [])

    def test_nonblocking_metadata_exception_is_counted_without_stopping_pipeline(self):
        class FailingSender:
            def send_raw_json(self, _payload):
                raise RuntimeError("udp send failed")

        runtime = main_module.SourceRuntime(
            index=3,
            url="rtsp://127.0.0.1:8554/src4",
            metadata_sender=FailingSender(),
            labels=[],
            source_options=None,
            profile=main_module.StreamProfile(False, 3),
            frame_w=1280,
            frame_h=720,
            source_fps=20,
        )

        main_module.send_metadata_nonblocking(runtime, "{}")

        assert runtime.metadata_send_ok == 0
        assert runtime.metadata_send_fail == 1

    def test_send_metadata_uses_object_detection_contract(self):
        sender = FakeMetadataSender()
        runtime = main_module.SourceRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            metadata_sender=sender,
            labels=["person"],
            source_options=None,
            profile=main_module.StreamProfile(False, 0),
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

        main_module.send_metadata(runtime, FakeSample(), boxes)

        assert len(sender.calls) == 1
        payload = json.loads(sender.calls[0])
        assert payload == {
            "type": "object-detection",
            "data": {
                "objects": [
                    {
                        "id": "obj_1",
                        "label": "person",
                        "confidence": 0.75,
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                    }
                ]
            },
            "timestamp": 1234,
            "frame_id": "42",
            "stream_id": "stream0",
            "stream_index": 0,
            "pts_ns": 1_234_000_000,
            "dts_ns": 1_200_000_000,
            "duration_ns": 50_000_000,
            "input_seq": 17,
            "orig_input_seq": 12,
            "rtp_timestamp": 111_060,
        }

    def test_send_metadata_preserves_missing_sample_identity(self):
        sender = FakeMetadataSender()
        runtime = main_module.SourceRuntime(
            index=0,
            url="rtsp://127.0.0.1:8554/src1",
            metadata_sender=sender,
            labels=["person"],
            source_options=None,
            profile=main_module.StreamProfile(False, 0),
            frame_w=100,
            frame_h=100,
            source_fps=30,
        )

        main_module.send_metadata(runtime, SimpleNamespace(pts_ns=-1, frame_id=-1), [])

        payload = json.loads(sender.calls[0])
        assert payload["timestamp"] == -1
        assert payload["frame_id"] == ""
        assert "rtp_timestamp" not in payload


def test_measurement_excludes_warmup_and_failed_sends():
    from metadata_measurement import MetadataMeasurement

    measurement = MetadataMeasurement(2, 100, 3)
    assert not measurement.observe(0, 100, False, False, 1.0)
    assert not measurement.observe(0, 101, True, False, 2.0)
    assert not measurement.observe(1, 100, False, False, 3.0)
    assert measurement.total == 0
    assert not measurement.observe(1, 101, False, True, 4.0)
    assert not measurement.observe(0, 102, True, False, 5.0)
    assert not measurement.observe(1, 102, True, False, 6.0)
    assert measurement.observe(0, 103, True, False, 7.0)
    assert measurement.summary() == dict(frames=3, elapsed_s=4.0, aggregate_fps=0.75,
                                        per_stream_frames=[2, 1], per_stream_send_failures=[0, 1])


# ---------------------------------------------------------------------------
# Configuration handling and option validation (Refs #526).
# Each test starts from VALID_CONFIG and breaks exactly one thing, so a failure
# names the rule that fired rather than "config rejected". The rules already
# owned by TestConfigLoading above (the 80-stream cap, the empty stream list,
# the worker count, the inflight and internal-queue limits, the liveness
# timeouts, the removed keys, the probe-skip requirements, the port overlap and
# the visible-stream cap) are not repeated here.
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "model": {"path": MODEL_PATH, "labels": "coco_label.txt", "decode_type": "yolo26"},
    "streams": ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
    "input": {
        "codec": "h264",
        "tcp": True,
        "latency_ms": 100,
        "drop_on_latency": False,
        "skip_rtsp_probe": False,
        "width": 0,
        "height": 0,
        "fps": 0,
        "decoder_buffers": 16,
        "decoder_input_buffers": 2,
        "decoder_tuning": "auto",
    },
    "runtime": {
        "profile": False,
        "warmup_frames": 30,
        "initial_detection_timeout_ms": 30000,
        "stream_detection_timeout_ms": 30000,
        "no_detection_timeout_ms": 30000,
    },
    "inference": {
        "workers": 1,
        "queue_depth": 4,
        "internal_queue_depth": 1,
        "max_inflight_per_stream": 4,
        "max_inflight_total": 8,
        "min_score": 0.55,
        "nms_iou": 0.6,
        "max_detections": 50,
    },
    "output": {
        "video_enabled": True,
        "insight": {
            "host": "127.0.0.1",
            "video_port_base": 9000,
            "metadata_port_base": 9100,
            "max_visible_streams": -1,
        },
    },
}


# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_full_config = config_writer(VALID_CONFIG)


class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main_module.load_app_config(write_full_config(tmp_path))

        assert cfg.model_path.endswith(MODEL_PATH)
        assert len(cfg.rtsp_urls) == 2
        assert cfg.codec == "h264"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": MODEL_PATH},
            "streams": ["rtsp://127.0.0.1:8554/src1"],
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main_module.load_app_config(write_full_config(tmp_path, root=raw))

        assert cfg.codec == "h264"
        assert cfg.workers == 1
        assert cfg.queue_depth == main_module.DEFAULT_QUEUE_DEPTH
        assert cfg.internal_queue_depth == main_module.DEFAULT_INTERNAL_QUEUE_DEPTH
        assert cfg.max_inflight_per_stream == main_module.DEFAULT_MAX_INFLIGHT_PER_STREAM
        assert cfg.max_inflight_total == main_module.DEFAULT_MAX_INFLIGHT_TOTAL
        assert cfg.decoder_buffers == main_module.DEFAULT_DECODER_BUFFERS
        assert cfg.decoder_input_buffers == main_module.DEFAULT_DECODER_INPUT_BUFFERS
        assert cfg.decoder_tuning == main_module.normalize_decoder_tuning("auto")
        assert cfg.input_width == 0
        assert cfg.input_height == 0
        assert cfg.input_fps == 0
        assert cfg.latency_ms == 100
        assert cfg.tcp == True
        assert cfg.skip_rtsp_probe == False
        assert cfg.min_score == pytest.approx(0.55)
        assert cfg.nms_iou == pytest.approx(0.60)
        assert cfg.max_detections == 50
        assert cfg.warmup_frames == 30
        assert cfg.initial_detection_timeout_ms == main_module.DEFAULT_INITIAL_DETECTION_TIMEOUT_MS
        assert cfg.stream_detection_timeout_ms == main_module.DEFAULT_STREAM_DETECTION_TIMEOUT_MS
        assert cfg.no_detection_timeout_ms == main_module.DEFAULT_NO_DETECTION_TIMEOUT_MS
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100
        assert cfg.insight_visible_streams == main_module.ALL_INSIGHT_STREAMS
        assert cfg.video_enabled == True
        assert cfg.decode_type == main_module.normalize_box_decode_type("yolo26")
        assert cfg.labels_path == Path("coco_label.txt")


REJECTED = [
    pytest.param(('model', 'path'), '', 'model.path must be set', id='model-path-empty'),
    pytest.param(('model', 'labels'), '', 'model.labels must be set', id='model-labels-empty'),
    pytest.param(('model', 'decode_type'), 'ssd', 'model.decode_type must be one of', id='decode-type-unsupported'),
    pytest.param(('output', 'insight', 'host'), '', 'output.insight.host must be set', id='insight-host-empty'),
    pytest.param(('input', 'codec'), 'vp9', 'input.codec must be h264/avc or h265/hevc', id='codec-unsupported'),
    pytest.param(('input', 'decoder_tuning'), 'fast', 'input.decoder_tuning must be one of', id='decoder-tuning-unsupported'),
    pytest.param(('input', 'latency_ms'), -1, 'input.latency_ms must be >= 0', id='latency-negative'),
    pytest.param(('input', 'width'), -1, 'input.width must be >= 0', id='width-negative'),
    pytest.param(('input', 'height'), -1, 'input.height must be >= 0', id='height-negative'),
    pytest.param(('input', 'fps'), -1, 'input.fps must be >= 0', id='fps-negative'),
    pytest.param(('input', 'width'), 640, 'input.width and input.height must be set together', id='width-without-height'),
    pytest.param(('input', 'decoder_buffers'), 0, 'input.decoder_buffers must be > 0', id='decoder-buffers-zero'),
    pytest.param(('input', 'decoder_buffers'), 65, 'input.decoder_buffers must be <= 64', id='decoder-buffers-above'),
    pytest.param(('input', 'decoder_input_buffers'), 0, 'input.decoder_input_buffers must be > 0', id='decoder-input-buffers-zero'),
    pytest.param(('inference', 'queue_depth'), 0, 'inference.queue_depth must be > 0', id='queue-depth-zero'),
    pytest.param(('inference', 'queue_depth'), 33, 'inference.queue_depth must be <= 32', id='queue-depth-above'),
    pytest.param(('inference', 'max_inflight_per_stream'), 33, 'inference.max_inflight_per_stream must be <= 32', id='inflight-per-stream-above'),
    pytest.param(('inference', 'min_score'), -0.01, 'inference.min_score must be between 0 and 1', id='min-score-below'),
    pytest.param(('inference', 'min_score'), 1.01, 'inference.min_score must be between 0 and 1', id='min-score-above'),
    pytest.param(('inference', 'nms_iou'), -0.01, 'inference.nms_iou must be between 0 and 1', id='nms-below'),
    pytest.param(('inference', 'nms_iou'), 1.01, 'inference.nms_iou must be between 0 and 1', id='nms-above'),
    pytest.param(('inference', 'max_detections'), 0, 'inference.max_detections must be > 0', id='max-detections-zero'),
    pytest.param(('runtime', 'warmup_frames'), -1, 'runtime.warmup_frames must be >= 0', id='warmup-negative'),
    pytest.param(('runtime', 'initial_detection_timeout_ms'), 0, 'runtime.initial_detection_timeout_ms must be > 0', id='initial-timeout-zero'),
    pytest.param(('output', 'insight', 'video_port_base'), 0, 'output.insight.video_port_base must be > 0', id='video-port-base-zero'),
    pytest.param(('output', 'insight', 'video_port_base'), 65536, 'output.insight.video_port_base must be <= 65535', id='video-port-base-above'),
    pytest.param(('output', 'insight', 'video_port_base'), 65535, 'output.insight video port range exceeds 65535', id='video-port-range-exceeds'),
    pytest.param(('output', 'insight', 'metadata_port_base'), 0, 'output.insight.metadata_port_base must be > 0', id='metadata-port-base-zero'),
    pytest.param(('output', 'insight', 'metadata_port_base'), 65536, 'output.insight.metadata_port_base must be <= 65535', id='metadata-port-base-above'),
    pytest.param(('output', 'insight', 'metadata_port_base'), 65535, 'output.insight metadata port range exceeds 65535', id='metadata-port-range-exceeds'),
    pytest.param(('output', 'insight', 'max_visible_streams'), -2, 'output.insight.max_visible_streams must be >= -1', id='visible-streams-below-minus-one'),
]


class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        with pytest.raises(ValueError) as excinfo:
            main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))

        assert message in str(excinfo.value)


class TestBoundariesAreAccepted:
    """An off-by-one that rejects a legal value is the failure nobody writes a test for."""

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (('inference', 'min_score'), 0.0),
            (('inference', 'min_score'), 1.0),
            (('inference', 'nms_iou'), 0.0),
            (('inference', 'nms_iou'), 1.0),
            (('input', 'latency_ms'), 0),
            (('input', 'fps'), 0),
            (('input', 'decoder_buffers'), 1),
            (('input', 'decoder_buffers'), 64),
            (('input', 'decoder_input_buffers'), 1),
            (('inference', 'queue_depth'), 1),
            (('inference', 'queue_depth'), 32),
            (('inference', 'internal_queue_depth'), 0),
            (('inference', 'internal_queue_depth'), 32),
            (('inference', 'max_inflight_per_stream'), 1),
            (('inference', 'max_inflight_per_stream'), 32),
            (('inference', 'max_inflight_total'), 1),
            (('inference', 'max_detections'), 1),
            (('runtime', 'warmup_frames'), 0),
            (('runtime', 'initial_detection_timeout_ms'), 1),
            (('runtime', 'stream_detection_timeout_ms'), 1),
            (('runtime', 'no_detection_timeout_ms'), 1),
            (('output', 'insight', 'video_port_base'), 1),
            (('output', 'insight', 'video_port_base'), 65534),
            (('output', 'insight', 'metadata_port_base'), 1),
            (('output', 'insight', 'max_visible_streams'), -1),
            (('output', 'insight', 'max_visible_streams'), 2),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))


class TestStreamList:
    def test_a_scalar_instead_of_a_list_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(
                write_full_config(tmp_path, [(("streams",), "rtsp://127.0.0.1:8554/src1")])
            )

    @pytest.mark.parametrize(
        ("streams", "bad_index"),
        [([""], 0), (["   "], 0), (["rtsp://127.0.0.1:8554/src1", ""], 1), (["rtsp://127.0.0.1:8554/src1", 42], 1)],
    )
    def test_a_blank_or_non_string_entry_names_its_index(self, tmp_path, streams, bad_index):
        """The index matters: with eighty streams, "one of them is wrong" is not actionable."""
        with pytest.raises(ValueError, match=rf"streams\[{bad_index}\] must be a non-empty string"):
            main_module.load_app_config(write_full_config(tmp_path, [(("streams",), streams)]))

    def test_stream_order_is_preserved(self, tmp_path):
        """Stream order decides which Insight port each stream publishes on."""
        streams = ["rtsp://host/c", "rtsp://host/a", "rtsp://host/b"]

        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("streams",), streams)]))

        assert cfg.rtsp_urls == streams


class TestCodecAliases:
    @pytest.mark.parametrize(
        ("alias", "codec"), [("h264", "h264"), ("avc", "h264"), ("h265", "h265"), ("hevc", "h265")]
    )
    def test_every_documented_alias_is_accepted(self, tmp_path, alias, codec):
        cfg = main_module.load_app_config(write_full_config(tmp_path, [(("input", "codec"), alias)]))

        assert cfg.codec == codec


class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main_module.load_app_config(config_path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(config_path)
