"""Hardware-gated MIPI multi-model smoke tests."""

from __future__ import annotations

import importlib.util
import json
import os
import selectors
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DEFAULT_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"
sys.path.insert(0, str(MAIN_PY.parent))

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
INSIGHT_FRAMES = 60
INSIGHT_DEADLINE_S = 120.0
# VideoSenderOptions.from_raw with H.264 sends RTP payload type 96.
H264_PAYLOAD_TYPE = 96
H264_IDR = 5
H264_SPS = 7
H264_STAP_A = 24
H264_FU_A = 28
# Neat's MetadataSender splits JSON above 1200 bytes into datagrams with a
# 12-byte header: magic 0x4e, version 0x01, big-endian message id, index, count.
METADATA_CHUNK_MAGIC = b"\x4e\x01"
METADATA_CHUNK_HEADER = 12
# Insight metadata type and data key the application sends for each profile.
INSIGHT_PROFILES = {
    "detect": ("object-detection", "objects"),
    "segment": ("segmentation", "segments"),
}

requires_mipi_camera = pytest.mark.skipif(
    os.environ.get("SIMA_NEAT_TEST_MIPI_CAMERA") != "1",
    reason="set SIMA_NEAT_TEST_MIPI_CAMERA=1 on a board with a configured camera",
)


def require_strict_zero_copy(backend: str) -> None:
    """Apply the application's own strict zero-copy check to the described backend."""
    spec = importlib.util.spec_from_file_location("mipi_multi_model_e2e", MAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.require_strict_zero_copy(backend)


def write_config(tmp_path: Path, frames: int) -> tuple[str, Path]:
    profile = os.environ.get("SIMA_NEAT_MIPI_PROFILE", "detect")
    model = os.environ.get("SIMA_NEAT_MIPI_MODEL", "")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["model"] = {"profile": profile, "path": model}
    raw["runtime"] = {"frames": frames, "timeout_ms": 30_000}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return profile, config


def h264_nal_types(packet: bytes) -> set[int]:
    """Validate one RTP packet and return the H.264 NAL unit types it starts."""
    assert len(packet) > 12, f"short RTP packet: {len(packet)} bytes"
    assert packet[0] >> 6 == 2, "RTP version is not 2"
    assert packet[1] & 0x7F == H264_PAYLOAD_TYPE, f"unexpected RTP payload type {packet[1] & 0x7F}"
    offset = 12 + 4 * (packet[0] & 0x0F)
    if packet[0] & 0x10:
        offset += 4 + 4 * int.from_bytes(packet[offset + 2 : offset + 4], "big")
    payload = packet[offset:]
    assert payload, "RTP packet has no H.264 payload"
    nal_type = payload[0] & 0x1F
    if nal_type == H264_STAP_A:
        types = set()
        position = 1
        while position + 2 < len(payload):
            size = int.from_bytes(payload[position : position + 2], "big")
            types.add(payload[position + 2] & 0x1F)
            position += 2 + size
        return types
    if nal_type == H264_FU_A:
        assert len(payload) > 1, "truncated FU-A packet"
        return {payload[1] & 0x1F} if payload[1] & 0x80 else set()
    return {nal_type}


class MetadataReassembler:
    """Rebuild MetadataSender JSON messages from single or chunked datagrams."""

    def __init__(self) -> None:
        self._partial: dict[bytes, dict[int, bytes]] = {}

    def feed(self, datagram: bytes) -> dict | None:
        if not datagram.startswith(METADATA_CHUNK_MAGIC):
            return json.loads(datagram)
        assert len(datagram) > METADATA_CHUNK_HEADER, "empty metadata chunk"
        message_id = datagram[2:10]
        index = datagram[10]
        count = datagram[11]
        assert 0 <= index < count, f"metadata chunk index {index} outside count {count}"
        parts = self._partial.setdefault(message_id, {})
        parts[index] = datagram[METADATA_CHUNK_HEADER:]
        if len(parts) < count:
            return None
        del self._partial[message_id]
        return json.loads(b"".join(parts[i] for i in range(count)))


def check_point(x, y, name: str) -> None:
    assert isinstance(x, (int, float)) and isinstance(y, (int, float)), name
    assert 0 <= x <= CAMERA_WIDTH and 0 <= y <= CAMERA_HEIGHT, f"{name} outside frame: {x},{y}"


def check_bbox(bbox, name: str) -> None:
    assert isinstance(bbox, list) and len(bbox) == 4, f"{name} bbox is not [x, y, w, h]"
    x, y, width, height = bbox
    check_point(x, y, f"{name} bbox origin")
    assert width >= 0 and height >= 0, f"{name} bbox has negative size"
    check_point(x + width, y + height, f"{name} bbox corner")


def check_item(item: dict, name: str) -> None:
    assert isinstance(item.get("id"), str) and item["id"], f"{name} has no id"
    assert isinstance(item.get("label"), str) and item["label"], f"{name} has no label"
    assert 0.0 <= item.get("confidence", -1.0) <= 1.0, f"{name} confidence out of range"
    check_bbox(item.get("bbox"), name)


def check_segment(segment: dict, name: str) -> None:
    check_item(segment, name)
    assert segment.get("mask_format") == "polygon", f"{name} mask is not a polygon"
    polygon = segment.get("mask")
    assert isinstance(polygon, list) and len(polygon) >= 3, f"{name} polygon has < 3 points"
    for point in polygon:
        assert isinstance(point, list) and len(point) == 2, f"{name} point is not [x, y]"
        check_point(point[0], point[1], f"{name} polygon point")


@pytest.mark.e2e
@requires_mipi_camera
def test_selected_profile_uses_strict_zero_copy(tmp_path: Path) -> None:
    _, config = write_config(tmp_path, frames=2)

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--describe"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"MIPI inference failed with {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )
    require_strict_zero_copy(result.stdout)
    assert "PASS strict-zero-copy MIPI" in result.stdout


@pytest.mark.e2e
@requires_mipi_camera
def test_selected_profile_streams_video_and_metadata_to_insight(tmp_path: Path) -> None:
    """Receive the Insight video and metadata streams on local UDP ports.

    Precondition: a person or common object must be in view of the camera in a
    lit scene. An empty or dark scene yields empty object or segment lists, and
    the test fails.
    """
    profile, config = write_config(tmp_path, frames=INSIGHT_FRAMES)
    if profile not in INSIGHT_PROFILES:
        pytest.skip(f"Insight streaming supports detect and segment, not {profile}")
    metadata_type, data_key = INSIGHT_PROFILES[profile]

    sockets = {}
    for name in ("video", "metadata"):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        receiver.bind(("127.0.0.1", 0))
        receiver.setblocking(False)
        sockets[name] = receiver
    video_port = sockets["video"].getsockname()[1]
    metadata_port = sockets["metadata"].getsockname()[1]

    nal_types: set[int] = set()
    video_packets = 0
    messages: list[dict] = []
    reassembler = MetadataReassembler()

    def drain(timeout: float) -> None:
        nonlocal video_packets
        for key, _ in selector.select(timeout):
            while True:
                try:
                    datagram = key.fileobj.recv(65_535)
                except BlockingIOError:
                    break
                if key.data == "video":
                    video_packets += 1
                    nal_types.update(h264_nal_types(datagram))
                else:
                    message = reassembler.feed(datagram)
                    if message is not None:
                        messages.append(message)

    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    selector = selectors.DefaultSelector()
    for name, receiver in sockets.items():
        selector.register(receiver, selectors.EVENT_READ, name)
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                str(MAIN_PY),
                "--config",
                str(config),
                "--insight-host",
                "127.0.0.1",
                "--insight-video-port",
                str(video_port),
                "--insight-metadata-port",
                str(metadata_port),
                "--insight-channel",
                "0",
            ],
            stdout=stdout,
            stderr=stderr,
        )
        try:
            deadline = time.monotonic() + INSIGHT_DEADLINE_S
            while process.poll() is None:
                assert time.monotonic() < deadline, "application did not finish in time"
                drain(0.2)
            drain(0.5)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            selector.close()
            for receiver in sockets.values():
                receiver.close()

    output = (
        f"--- STDOUT ---\n{stdout_path.read_text()}\n--- STDERR ---\n{stderr_path.read_text()}"
    )
    assert process.returncode == 0, f"MIPI Insight run failed with {process.returncode}\n{output}"
    assert "PASS strict-zero-copy MIPI" in output and "+ Insight" in output, output

    assert video_packets > 0, f"no RTP video packets arrived on port {video_port}"
    assert {H264_SPS, H264_IDR} <= nal_types, f"no H.264 SPS and IDR in NAL types {nal_types}"

    assert messages, f"no metadata arrived on port {metadata_port}"
    items = []
    for message in messages:
        assert message.get("type") == metadata_type, f"unexpected metadata type: {message}"
        entries = message.get("data", {}).get(data_key)
        assert isinstance(entries, list), f"metadata has no {data_key} list: {message}"
        items.extend(entries)
    assert items, f"{len(messages)} metadata messages had no {data_key}; is anything in view?"
    for index, item in enumerate(items):
        if profile == "segment":
            check_segment(item, f"segment {index}")
        else:
            check_item(item, f"object {index}")
