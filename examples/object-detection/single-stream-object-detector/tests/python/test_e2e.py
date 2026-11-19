"""End-to-end tests: each source type reaches Insight as video and detection metadata."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
import yaml

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.pytest_fixtures import usb_camera_cases

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
LABELS = (EXAMPLE_DIR / "src" / "common" / "coco_label.txt").read_text().split("\n")
FRAMES = 100
# Above the shipped 0.30, so the scores show that the example applies inference.min_score.
MIN_SCORE = 0.5
# The USB case asks the camera for this MJPEG mode.
USB_MODE = {"width": 1280, "height": 720, "fps": 30}

SOURCE_CASES = [
    pytest.param({"type": "rtsp", "fixture": "rtsp_h264_url", "codec": "h264"}, id="rtsp-h264"),
    pytest.param({"type": "rtsp", "fixture": "rtsp_h265_url", "codec": "h265"}, id="rtsp-h265"),
    pytest.param({"type": "rtsp", "fixture": "rtsp_mjpeg_url", "codec": "mjpeg"}, id="rtsp-mjpeg"),
    # HTTP MJPEG reports no frame rate, and the test server uses a self-signed certificate.
    pytest.param(
        {"type": "http", "fixture": "http_mjpeg_url", "fps": 30, "ssl_strict": False},
        id="http-mjpeg",
    ),
] + usb_camera_cases({"type": "usb", "fixture": "usb_camera_id"})


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _frame_size(url: str) -> tuple[int, int]:
    """The test stream's frame size, which the config must state, read over TCP as the
    example reads RTSP."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        pytest.fail(f"cannot read the frame size of {url}")
    return width, height


def _input_section(source: dict, value: str) -> dict:
    """The input keys of the case's input type."""
    if source["type"] == "usb":
        return {"type": "usb", "source": value, **USB_MODE}
    width, height = _frame_size(value)
    inputs = {"type": source["type"], "source": value, "width": width, "height": height}
    inputs["fps"] = source.get("fps", 0)
    if source["type"] == "rtsp":
        inputs["codec"] = source["codec"]
    return inputs


def _keep_only_input_type_keys(config_path: Path, source: dict) -> None:
    """Drop the RTSP keys the shipped config carries from an HTTP or USB case's config."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if source["type"] != "rtsp":
        config["input"].pop("codec")
        config["input"].pop("rtsp")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


@pytest.mark.e2e
@pytest.mark.parametrize("source", SOURCE_CASES)
def test_publishes_insight_video_and_metadata(
    request, source, e2e_model_path, test_timeout_ms, e2e_config_writer
):
    value = request.getfixturevalue(source["fixture"])
    inputs = _input_section(source, value)
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            # Absolute, because the example resolves a relative model.path from its config.
            "model": {"path": str(Path(e2e_model_path).resolve())},
            "input": inputs,
            "inference": {"frames": FRAMES, "min_score": MIN_SCORE},
            # This test is the Insight receiver, so the example publishes to loopback.
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
        }
        | ({"app": {"ssl_strict": source["ssl_strict"]}} if source["type"] == "http" else {})
    )
    _keep_only_input_type_keys(config_path, source)

    with (
        MetadataJsonListener(
            "127.0.0.1",
            metadata_port,
            num_ports=1,
            require_all_ports=True,
            min_object_count=1,
        ) as listener,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video,
    ):
        video.bind(("127.0.0.1", video_port))
        video.settimeout(5.0)
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
            check=False,
        )
        assert result.returncode == 0, (
            f"exited with {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        metadata = listener.wait_for_messages(5.0)
        following = listener.wait_for_messages(5.0)
        # The rest of the run's messages, so the checks below see every frame it sent.
        rest = listener.drain_pending()
        video_packet = video.recv(65536)

    assert f"processed={FRAMES} " in result.stdout, result.stdout
    assert metadata.success, metadata.error
    assert following.success, following.error
    first, last = metadata.messages[-1], following.messages[-1]
    assert int(last.frame_id) > int(first.frame_id)
    for message in metadata.messages + following.messages + rest.messages:
        for entry in json.loads(message.payload)["data"]["objects"]:
            assert entry["label"] in LABELS, entry
            assert entry["confidence"] >= MIN_SCORE, entry
    # H.265 goes to Insight as received (RTP payload type 98), and so does a USB camera's JPEG
    # (26). H.264 is passed through and decoded MJPEG and HTTP frames are encoded as H.264 (96).
    payload_type = 26 if source["type"] == "usb" else 98 if source.get("codec") == "h265" else 96
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
    assert video_packet[1] & 0x7F == payload_type
