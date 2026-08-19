"""Unit tests for the adaptive-resolution-object-detector example."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"          # entry point: --mode dispatcher
ADAPTIVE_PY = PYTHON_DIR / "adaptive_app.py"
FUSED_PY = PYTHON_DIR / "fused_app.py"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

pytestmark = pytest.mark.unit


def write_config(tmp_path: Path, streams_block: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  path: assets/models/yolo26m-det-int8-b1.tar.gz",
                streams_block,
                "output:",
                "  insight:",
                "    host: 127.0.0.1",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


RICH_TWO = textwrap.dedent(
    """\
    streams:
      max_streams: 8
      sources:
        - id: cam-1
          rtsp_url: rtsp://127.0.0.1:8554/src1
        - id: cam-2
          rtsp_url: rtsp://127.0.0.1:8554/src2"""
)

BARE_THREE = textwrap.dedent(
    """\
    streams:
      - rtsp://127.0.0.1:8554/src1
      - rtsp://127.0.0.1:8554/src2
      - rtsp://127.0.0.1:8554/src3"""
)


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

    def test_validate_config_only_reports_stream_count(self, tmp_path: Path):
        config_path = write_config(tmp_path, RICH_TWO)
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path), "--validate-config-only"],
            capture_output=True,
            text=True,
            cwd=str(EXAMPLE_DIR),
            timeout=20,
        )
        assert result.returncode == 0
        assert "streams=2" in result.stdout


class TestConfigLoading:
    def test_rich_sources(self, tmp_path: Path):
        from adaptive_app import load_app_config

        cfg = load_app_config(write_config(tmp_path, RICH_TWO))
        assert cfg.model_path == "assets/models/yolo26m-det-int8-b1.tar.gz"
        assert [s.id for s in cfg.sources] == ["cam-1", "cam-2"]
        assert cfg.max_streams == 8

    def test_bare_list_autonames(self, tmp_path: Path):
        from adaptive_app import load_app_config

        cfg = load_app_config(write_config(tmp_path, BARE_THREE))
        assert [s.id for s in cfg.sources] == ["cam-1", "cam-2", "cam-3"]
        assert [s.rtsp_url for s in cfg.sources][0] == "rtsp://127.0.0.1:8554/src1"

    def test_rejects_over_max_streams(self, tmp_path: Path):
        from adaptive_app import load_app_config

        block = RICH_TWO.replace("max_streams: 8", "max_streams: 1")
        with pytest.raises(ValueError, match="max_streams"):
            load_app_config(write_config(tmp_path, block))

    def test_rejects_empty_streams(self, tmp_path: Path):
        from adaptive_app import load_app_config

        with pytest.raises(ValueError, match="streams"):
            load_app_config(write_config(tmp_path, "streams: []"))

    def test_reload_sources_matches_load(self, tmp_path: Path):
        from adaptive_app import load_app_config, reload_sources

        config_path = write_config(tmp_path, RICH_TWO)
        cfg = load_app_config(config_path)
        reloaded = reload_sources(config_path)
        assert [s.id for s in reloaded] == [s.id for s in cfg.sources]



class TestOutputPolicy:
    HEIGHTS = [2160, 1080, 720, 480]






class FakeMetadataSender:
    def __init__(self):
        self.calls = []
        self.raw = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
        self.calls.append((metadata_type, data_json, timestamp_ms, frame_id))
        return True

    # The adaptive app publishes here, not through send_metadata: Insight matches
    # a held frame by an exact rtp_timestamp key that the convenience API cannot
    # carry. send_metadata() swallows exceptions so a dropped datagram never
    # stalls detection - which also means a fake missing this method would make
    # the test silently assert nothing.
    def send_raw_json(self, payload_json):
        self.raw.append(payload_json)
        return True


class FakeSample:
    frame_id = 42
    pts_ns = 1_234_000_000


class TestMetadata:

    def test_send_metadata_uses_object_detection_contract(self):
        from adaptive_app import ProfileWindow, StreamRuntime, send_metadata

        sender = FakeMetadataSender()
        runtime = StreamRuntime(
            channel=0,
            id="cam-1",
            url="rtsp://127.0.0.1:8554/src1",
            labels=["person"],
            frame_w=100,
            frame_h=100,
            metadata_sender=sender,
            profile=ProfileWindow(False, "cam-1"),
        )
        boxes = [{"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": 0.75, "class_id": 0}]

        send_metadata(runtime, FakeSample(), boxes, stream_count=2)

        assert len(sender.raw) == 1, "detections must be published via send_raw_json"
        envelope = json.loads(sender.raw[0])
        assert envelope["type"] == "object-detection"
        assert envelope["stream_id"] == "cam-1"
        assert envelope["stream_index"] == 0
        assert envelope["frame_id"] == "42"
        assert envelope["pts_ns"] == FakeSample.pts_ns
        assert envelope["timestamp"] == 1234
        # `data` is already an object in the envelope, not a JSON string.
        data = envelope["data"]
        assert data["stream_count"] == 2
        assert data["objects"][0]["label"] == "person"


class TestModeDispatch:
    """`main.py` is an entry point, not an implementation.

    It selects a topology with --mode and forwards everything else unchanged.
    The C++ entry point (src/cpp/main.cpp) takes the same flag, which is what
    lets the pipelines chooser switch language without changing anything else.
    """

    def test_help_lists_both_modes(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "adaptive" in result.stdout and "fused" in result.stdout

    def test_rejects_unknown_mode(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--mode", "nonsense",
             "--config", str(EXAMPLE_DIR / "src" / "common" / "config.yaml"),
             "--validate-config-only"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode != 0

    def test_both_implementations_are_importable(self):
        """Each mode's module must load on its own - main.py only dispatches."""
        import importlib

        for module_name, entry in (("adaptive_app", "load_app_config"),
                                   ("fused_app", "load_app_config")):
            module = importlib.import_module(module_name)
            assert hasattr(module, entry), f"{module_name}.{entry} missing"
            assert callable(module.main)

    def test_modes_do_not_share_a_config_schema(self, tmp_path):
        """Handing a mode the other's config must fail, not half-run.

        `adaptive` reads streams.sources plus the adaptive: policy sections;
        `fused` reads a bare streams: list. Silently accepting the wrong shape
        would start a pipeline with settings the user never asked for.
        """
        import adaptive_app
        import fused_app

        adaptive_cfg = tmp_path / "adaptive.yaml"
        adaptive_cfg.write_text(textwrap.dedent("""
            model:
              path: /tmp/model.tar.gz
              labels: /tmp/labels.txt
            adaptive:
              resolutions: [640]
            streams:
              max_streams: 4
              sources:
                - id: cam-1
                  rtsp_url: rtsp://127.0.0.1:8554/src1
            output:
              insight:
                host: 127.0.0.1
        """).strip() + "\n", encoding="utf-8")

        # The adaptive schema is not a valid fused config.
        with pytest.raises(Exception):
            fused_app.load_app_config(adaptive_cfg)

        fused_cfg = tmp_path / "fused.yaml"
        fused_cfg.write_text(textwrap.dedent("""
            model:
              path: /tmp/model.tar.gz
              labels: /tmp/labels.txt
            streams:
              - rtsp://127.0.0.1:8554/src1
            output:
              insight:
                host: 127.0.0.1
        """).strip() + "\n", encoding="utf-8")

        # The reverse is NOT symmetric, and that is deliberate: the adaptive
        # loader also accepts a bare list and auto-names the streams (see
        # TestConfigLoading::test_bare_list_autonames). So a fused config loads
        # under adaptive - what it will not do is silently pick up fused-only
        # settings, since there are none in that shape.
        adaptive_cfg_from_fused = adaptive_app.load_app_config(fused_cfg)
        assert len(adaptive_cfg_from_fused.sources) == 1
        assert adaptive_cfg_from_fused.sources[0].rtsp_url.endswith("/src1")

    def test_fused_accepts_more_than_four_streams(self, tmp_path):
        """The old "up to four streams" guard was a development-phase limit.

        This topology is the one the high-density example uses for 16/24/48
        streams, so the ceiling is measured capacity, not a placeholder. The C++
        side carries the same limit.
        """
        import fused_app

        urls = "\n".join(f"  - rtsp://127.0.0.1:8554/src{i}" for i in range(1, 9))
        cfg_path = tmp_path / "fused8.yaml"
        cfg_path.write_text(
            "model:\n  path: /tmp/model.tar.gz\n  labels: /tmp/labels.txt\n"
            f"streams:\n{urls}\n"
            "output:\n  insight:\n    host: 127.0.0.1\n",
            encoding="utf-8",
        )
        cfg = fused_app.load_app_config(cfg_path)
        assert len(cfg.rtsp_urls) == 8
