"""Unit tests for the single-stream object detector: the shipped config and each input type's
documented keys load, and the values the example rejects fail with their message.
tests/cpp/test_unit.cpp runs the same cases against the C++ binary.
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
        "width": 0,
        "height": 0,
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


def validate(config_path: Path, capsys) -> tuple[int, str]:
    """main --validate-config-only on `config_path`: the exit code and stdout."""
    code = main_module.main(["--config", str(config_path), "--validate-config-only"])
    return code, capsys.readouterr().out


# The values the example rejects before it runs: (section, key, value, message).
REJECTED = [
    ("input", "type", "rtp", "input.type must be rtsp, http, or usb"),
    ("input", "codec", "vp8", "input.codec must be h264, h265, or mjpeg"),
    ("output", "host", "", "output.insight.host must not be empty"),
    ("inference", "frames", -1, "inference.frames must be 0 or more"),
]


def config_path(tmp_path: Path, input_type: str, change=None) -> Path:
    """A config of `input_type` with the keys the config comments list, and `change`, a
    (section, key, value) of REJECTED, applied."""
    config = {
        "model": {"path": "models/yolo26m_w16a16_mpk.tar.gz", "labels": "coco_label.txt"},
        "input": dict(INPUTS[input_type]),
        "inference": {"frames": 0, "min_score": 0.3, "nms_iou": 0.6, "max_detections": 50},
        "output": {
            "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100}
        },
    }
    if input_type == "http":
        config["app"] = {"ssl_strict": False}
    if change:
        section, key, value = change
        (config["output"]["insight"] if section == "output" else config[section])[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_shipped_config_loads(capsys):
    code, out = validate(COMMON_DIR / "config.yaml", capsys)
    assert code == 0, "config.yaml loads"
    assert "input=rtsp, codec=h264, 0x0, fps=0" in out, "config.yaml reads its values"


@pytest.mark.parametrize("input_type", ["rtsp", "http", "usb"])
def test_each_input_type_loads_with_its_documented_keys(tmp_path, capsys, input_type):
    code, out = validate(config_path(tmp_path, input_type), capsys)
    assert code == 0, f"{input_type} config loads"
    assert f"input={input_type}" in out, f"{input_type} config reads as {input_type}"


@pytest.mark.parametrize("section,key,value,message", REJECTED, ids=[c[3] for c in REJECTED])
def test_rejected_values_fail_with_their_message(tmp_path, section, key, value, message):
    with pytest.raises(ValueError, match=message):
        main_module.main(
            ["--config", str(config_path(tmp_path, "rtsp", (section, key, value)))]
            + ["--validate-config-only"]
        )
