"""Unit tests for the single-stream object detector: the shipped config and each input type's
documented keys load. tests/cpp/test_unit.cpp runs the same cases against the C++ binary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "single_stream_object_detector_main")

pytestmark = pytest.mark.unit

# The input keys each input type reads, with test values.
INPUTS = {
    "rtsp": {
        "type": "rtsp",
        "source": "rtsp://127.0.0.1:8554/src1",
        "codec": "h264",
        "width": 1280,
        "height": 720,
        "fps": 0,
        "rtsp": {"tcp": True, "latency_ms": 100},
    },
    "http": {
        "type": "http",
        "source": "http://127.0.0.1:8080/stream.mjpg",
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "usb": {"type": "usb", "source": "usb-0000:00:14.0-1", "width": 1280, "height": 720, "fps": 30},
}


def load(tmp_path: Path, input_type: str):
    """load_app_config on a config of `input_type` with the keys the config comments list."""
    config = {
        "model": {"path": "models/yolo26m_w16a16_mpk.tar.gz", "labels": "coco_label.txt"},
        "input": INPUTS[input_type],
        "inference": {"frames": 0, "min_score": 0.3, "nms_iou": 0.6, "max_detections": 50},
        "output": {
            "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100}
        },
    }
    if input_type == "http":
        config["app"] = {"ssl_strict": False}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return main_module.load_app_config(path)


def test_shipped_config_loads():
    cfg = main_module.load_app_config(COMMON_DIR / "config.yaml")
    assert cfg.input_type == "rtsp"
    assert (cfg.codec, cfg.width, cfg.height, cfg.fps) == ("h264", 1280, 720, 0)


@pytest.mark.parametrize(
    ("input_type", "codec", "ssl_strict"),
    [("rtsp", "h264", True), ("http", "mjpeg", False), ("usb", "mjpeg", True)],
)
def test_each_input_type_loads_with_its_documented_keys(tmp_path, input_type, codec, ssl_strict):
    cfg = load(tmp_path, input_type)
    assert (cfg.input_type, cfg.codec, cfg.ssl_strict) == (input_type, codec, ssl_strict)
