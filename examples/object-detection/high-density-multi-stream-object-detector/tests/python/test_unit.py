"""Unit tests for the high-density multi-stream object detector."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(
    EXAMPLE_DIR, "high_density_multi_stream_object_detector_main"
)

pytestmark = pytest.mark.unit

# Every key the application reads, with test values.
CONFIG_LINES = {
    "model.path": "model:\n  path: models/yolo26n_w8a8_mpk.tar.gz",
    "model.labels": "  labels: coco_label.txt",
    "input.source": "input:\n  source:\n    - rtsp://127.0.0.1:8554/src1\n"
    "    - rtsp://127.0.0.1:8554/src2",
    "input.codec": "  codec: h264",
    "input.width": "  width: 1280",
    "input.height": "  height: 720",
    "input.fps": "  fps: 30",
    "input.rtsp.tcp": "  rtsp:\n    tcp: true",
    "input.rtsp.latency_ms": "    latency_ms: 100",
    "inference.min_score": "inference:\n  min_score: 0.3",
    "inference.nms_iou": "  nms_iou: 0.6",
    "inference.max_detections": "  max_detections: 50",
    "inference.max_inflight_per_stream": "  max_inflight_per_stream: 1",
    "inference.max_inflight_total": "  max_inflight_total: 8",
    "output.video_enabled": "output:\n  video_enabled: true",
    "output.insight.host": "  insight:\n    host: 127.0.0.1",
    "output.insight.video_port_base": "    video_port_base: 9000",
    "output.insight.metadata_port_base": "    metadata_port_base: 9100",
    "app.decoder_buffers": "app:\n  decoder_buffers: 8",
    "app.queue_depth": "  queue_depth: 16",
    "app.internal_queue_depth": "  internal_queue_depth: 1",
}


def write_config(tmp_path: Path, extra: str = "", **replace: str) -> Path:
    lines = {**CONFIG_LINES, **replace}
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines.values()) + "\n" + extra, encoding="utf-8")
    return path


def test_help_lists_the_options():
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--validate-config-only" in result.stdout


def test_missing_config_file_exits_with_code_2(tmp_path, capsys):
    assert main_module.main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "config file not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("filename", "streams", "fps"),
    [
        ("config.yaml", 20, 0),
        ("config-30x720p20fps.yaml", 30, 20),
        ("config-60x720p10fps.yaml", 60, 10),
    ],
)
def test_shipped_profiles_load(filename, streams, fps):
    cfg = main_module.load_app_config(COMMON_DIR / filename)
    assert len(cfg.sources) == streams
    assert cfg.fps == fps
    assert (cfg.width, cfg.height) == (1280, 720)


def test_config_reads_every_key(tmp_path):
    path = write_config(tmp_path)
    cfg = main_module.load_app_config(path)
    assert cfg.sources == ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"]
    assert cfg.model_path == tmp_path.resolve() / "models/yolo26n_w8a8_mpk.tar.gz"
    assert cfg.rtsp_tcp is True and cfg.rtsp_latency_ms == 100
    assert cfg.queue_depth == 16 and cfg.internal_queue_depth == 1


def test_unknown_key_is_rejected(tmp_path):
    path = write_config(tmp_path, extra="streams:\n  - rtsp://old\n")
    with pytest.raises(ValueError, match="unknown keys: streams"):
        main_module.load_app_config(path)


def test_codec_must_be_h264_or_h265(tmp_path):
    path = write_config(tmp_path, **{"input.codec": "  codec: mjpeg"})
    with pytest.raises(ValueError, match="input.codec must be h264 or h265"):
        main_module.load_app_config(path)
