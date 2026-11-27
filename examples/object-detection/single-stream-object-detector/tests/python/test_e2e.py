"""End-to-end tests: each source type reaches Insight as video and detection metadata."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import pytest
import yaml

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.pytest_fixtures import usb_camera_cases

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
APPS_ROOT = EXAMPLE_DIR.parents[2]
LABELS = (APPS_ROOT / "support" / "labels" / "coco80.txt").read_text().splitlines()
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
    # An RTSP stream's size and rate come from the application's probe.
    width, height = (0, 0) if source["type"] == "rtsp" else _frame_size(value)
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


def _drain_video(sock: socket.socket, stop: threading.Event, headers: list[bytes]) -> None:
    """Keep each video packet's RTP header until `stop` is set and the socket is empty."""
    sock.settimeout(0.2)
    while True:
        try:
            headers.append(sock.recv(65536)[:12])
        except TimeoutError:
            if stop.is_set():
                return


def _pairs(timestamp_ms: int, rtp_timestamps: set[int]) -> bool:
    """Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a
    video frame's RTP timestamp."""
    ticks = (timestamp_ms * 90) & 0xFFFFFFFF
    return any(abs(ticks - rtp) <= 90 for rtp in rtp_timestamps)


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
        headers: list[bytes] = []
        stop = threading.Event()
        drain = threading.Thread(target=_drain_video, args=(video, stop, headers))
        drain.start()
        try:
            result = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                capture_output=True,
                text=True,
                timeout=test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
                check=False,
            )
        finally:
            stop.set()
            drain.join()
        assert result.returncode == 0, (
            f"exited with {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        metadata = listener.wait_for_messages(5.0)
        following = listener.wait_for_messages(5.0)
        # The rest of the run's messages, so the checks below see every frame it sent.
        rest = listener.drain_pending()

    assert f"processed={FRAMES} " in result.stdout, result.stdout
    assert metadata.success, metadata.error
    assert following.success, following.error
    first, last = metadata.messages[-1], following.messages[-1]
    assert int(last.frame_id) > int(first.frame_id)
    messages = metadata.messages + following.messages + rest.messages
    for message in messages:
        for entry in json.loads(message.payload)["data"]["objects"]:
            assert entry["label"] in LABELS, entry
            assert entry["confidence"] >= MIN_SCORE, entry
    # RTSP H.264 (RTP payload type 96), H.265 (98), and MJPEG (26) and a USB camera's JPEG (26)
    # go to Insight as received; decoded HTTP frames are encoded as H.264 (96).
    codec = "mjpeg" if source["type"] == "usb" else source.get("codec")
    payload_type = 96 if source["type"] == "http" else {"h264": 96, "h265": 98, "mjpeg": 26}[codec]
    assert headers and len(headers[0]) == 12 and headers[0][0] >> 6 == 2
    assert headers[0][1] & 0x7F == payload_type
    # Insight draws each message's boxes on the video frame its timestamp names.
    rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers}
    unpaired = [m.timestamp_ms for m in messages if not _pairs(m.timestamp_ms, rtp_timestamps)]
    assert not unpaired, f"metadata timestamps with no video frame: {unpaired[:5]}"
