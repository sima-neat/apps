"""Unit tests for the single-stream object detector's own config rules.

input.type selects the input, and the example reads only the keys that type takes, each with
the codecs and frame sizes it can carry. tests/cpp/test_unit.cpp runs the same cases against
the C++ binary.
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


def load(tmp_path: Path, source_type: str, app: dict | None = None, **inputs):
    """load_app_config on a config of `source_type`, with `inputs` replacing input keys."""
    config = {
        "model": {"path": "models/yolo26m_w16a16_mpk.tar.gz", "labels": "coco_label.txt"},
        "input": {**INPUTS[source_type], **inputs},
        "inference": {"frames": 0, "min_score": 0.3, "nms_iou": 0.6, "max_detections": 50},
        "output": {
            "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100}
        },
    }
    if source_type == "http" and app is None:
        app = {"ssl_strict": False}
    if app is not None:
        config["app"] = app
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return main_module.load_app_config(path)


def test_shipped_config_loads():
    cfg = main_module.load_app_config(COMMON_DIR / "config.yaml")
    assert cfg.source_type == "rtsp"
    assert (cfg.codec, cfg.width, cfg.height, cfg.fps) == ("h264", 1280, 720, 0)


@pytest.mark.parametrize(
    ("source_type", "codec", "ssl_strict"),
    [("rtsp", "h264", True), ("http", "mjpeg", False), ("usb", "mjpeg", True)],
)
def test_each_input_type_reads_its_own_keys(tmp_path, source_type, codec, ssl_strict):
    cfg = load(tmp_path, source_type)
    assert (cfg.source_type, cfg.codec, cfg.ssl_strict) == (source_type, codec, ssl_strict)


@pytest.mark.parametrize(
    ("source_type", "app", "extra", "key"),
    [
        ("http", None, {"codec": "mjpeg"}, "input.codec"),
        ("usb", None, {"rtsp": {"tcp": True}}, "input.rtsp.tcp"),
        ("rtsp", {"ssl_strict": True}, {}, "app.ssl_strict"),
    ],
)
def test_a_key_of_another_input_type_is_rejected(tmp_path, source_type, app, extra, key):
    with pytest.raises(ValueError, match=f"unknown keys: {key}"):
        load(tmp_path, source_type, app, **extra)


def test_unknown_input_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="input.type must be rtsp, http, or usb"):
        load(tmp_path, "rtsp", type="rtsps")


def test_codec_must_be_h264_h265_or_mjpeg(tmp_path):
    with pytest.raises(ValueError, match="input.codec must be h264, h265, or mjpeg"):
        load(tmp_path, "rtsp", codec="vp9")


@pytest.mark.parametrize("source_type", ["http", "usb"])
def test_http_and_usb_sources_need_a_frame_rate(tmp_path, source_type):
    with pytest.raises(ValueError, match="input.fps must be above 0 for an HTTP or USB source"):
        load(tmp_path, source_type, fps=0)


@pytest.mark.parametrize(("width", "height"), [(1284, 720), (3840, 2160)])
def test_usb_frame_size_must_fit_rtp_jpeg(tmp_path, width, height):
    with pytest.raises(ValueError, match="multiples of 8 and at most 2040"):
        load(tmp_path, "usb", width=width, height=height)
