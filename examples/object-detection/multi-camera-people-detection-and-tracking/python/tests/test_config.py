"""Config loading tests for the multi-camera people tracking example."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


@pytest.mark.unit
class TestConfigLoading:
    def test_load_app_config_parses_dynamic_stream_list(self, tmp_path: Path):
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model: assets/models/yolo_v8m_mpk.tar.gz
input:
  tcp: true
  latency_ms: 200
inference:
  frames: 0
  fps: 0
  bitrate_kbps: 2500
  profile: true
  person_class_id: 0
  detection_threshold: null
  nms_iou_threshold: null
  top_k: null
tracking:
  iou_threshold: 0.3
  max_missing_frames: 15
output:
  optiview:
    host: 192.168.0.107
    video_port_base: 9000
    json_port_base: 9100
  debug_dir: null
  save_every: 0
streams:
  - rtsp://192.168.0.235:8554/src1
  - rtsp://192.168.0.235:8554/src2
  - rtsp://192.168.0.235:8554/src3
""".strip(),
            encoding="utf-8",
        )

        cfg = load_app_config(config_path)

        assert cfg.model == "assets/models/yolo_v8m_mpk.tar.gz"
        assert cfg.optiview_host == "192.168.0.107"
        assert cfg.optiview_video_port_base == 9000
        assert cfg.optiview_json_port_base == 9100
        assert cfg.tcp is True
        assert cfg.save_every == 0
        assert cfg.rtsp_urls == [
            "rtsp://192.168.0.235:8554/src1",
            "rtsp://192.168.0.235:8554/src2",
            "rtsp://192.168.0.235:8554/src3",
        ]

    def test_load_app_config_rejects_missing_streams(self, tmp_path: Path):
        from utils.config import load_app_config

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            """
model: assets/models/yolo_v8m_mpk.tar.gz
input: {}
inference: {}
tracking: {}
output:
  optiview:
    host: 127.0.0.1
    video_port_base: 9000
    json_port_base: 9100
  debug_dir: null
  save_every: 0
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="streams"):
            load_app_config(config_path)

    def test_common_config_yaml_uses_supported_shape(self):
        from utils.config import load_app_config

        config_path = EXAMPLE_DIR / "common" / "config.yaml"
        cfg = load_app_config(config_path)

        assert cfg.model
        assert cfg.optiview_video_port_base > 0
        assert cfg.optiview_json_port_base > 0
        assert cfg.save_every >= 0
        assert isinstance(cfg.rtsp_urls, list)
