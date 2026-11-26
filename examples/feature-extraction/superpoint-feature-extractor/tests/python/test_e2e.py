"""End-to-end test: an H.264 RTSP stream reaches Insight as video and feature-point metadata."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener, MetadataJsonResult

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 100
# SuperPoint's input frame, which the decoder reports its keypoints in.
MODEL_SIZE = (640, 480)
# The decoder keeps at most 600 points per frame, Core's default top_k for every profile.
MAX_POINTS = 600


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _drain_video(sock: socket.socket, stop: threading.Event, headers: list[bytes]) -> None:
    """Keep each video packet's RTP header until `stop` is set and the socket is empty."""
    sock.settimeout(0.2)
    while True:
        try:
            headers.append(sock.recv(65536)[:12])
        except TimeoutError:
            if stop.is_set():
                return


@pytest.mark.e2e
def test_publishes_insight_video_and_feature_points(
    rtsp_h264_url, e2e_model_path, test_timeout_ms, e2e_config_writer, tmp_output_dir
):
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    # e2e_config_writer writes the config next to the test's out/ folder, and the model path is
    # relative to the config's folder. The app runs from EXAMPLE_DIR, so the run passes only if
    # the app resolves model.path from its config's folder.
    model_path = os.path.relpath(Path(e2e_model_path).resolve(), tmp_output_dir.parent.resolve())
    assert not (EXAMPLE_DIR / model_path).exists(), "model.path also resolves from the cwd"
    config_path = e2e_config_writer(
        {
            "model": {"path": model_path},
            "input": {"source": rtsp_h264_url},
            "inference": {"frames": FRAMES},
            # This test is the Insight receiver, so the example publishes to loopback.
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
        }
    )

    with (
        MetadataJsonListener(
            "127.0.0.1",
            metadata_port,
            num_ports=1,
            metadata_type="pose-estimation",
            data_array_key="poses",
        ) as listener,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video,
    ):
        video.bind(("127.0.0.1", video_port))
        headers: list[bytes] = []
        stop = threading.Event()
        drain = threading.Thread(target=_drain_video, args=(video, stop, headers))
        drain.start()
        # Read the metadata while the app runs: a socket buffer holds only part of a run's
        # messages. Each read returns at its first valid message, or times out.
        reads: list[MetadataJsonResult] = []

        def read_until_stopped() -> None:
            while not stop.is_set():
                reads.append(listener.wait_for_messages(0.2))

        read_metadata = threading.Thread(target=read_until_stopped)
        read_metadata.start()
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
            read_metadata.join()
        assert result.returncode == 0, (
            f"exited with {result.returncode}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        # The datagrams that arrived after the last read.
        rest = listener.drain_pending()

    summary = re.search(
        rf"frames={FRAMES} average_points=([0-9.]+) descriptor_dim=256 "
        rf"video_sender=127\.0\.0\.1:{video_port}$",
        result.stdout,
        re.MULTILINE,
    )
    assert summary, result.stdout
    assert 0 < float(summary.group(1)) <= MAX_POINTS
    stream = re.search(r"stream=(\d+)x(\d+)@", result.stdout)
    assert stream, result.stdout
    width, height = map(int, stream.groups())

    errors = [read.error for read in reads if read.error and not read.timed_out]
    assert not errors and rest.success, errors[:3] or rest.error
    messages = [message for read in reads for message in read.messages] + rest.messages
    # Every frame's message arrives, except the ones the app reports as dropped.
    dropped = re.search(r"^(\d+) metadata messages dropped$", result.stdout, re.MULTILINE)
    assert len(messages) == FRAMES - (int(dropped.group(1)) if dropped else 0), len(messages)
    assert all(message.frame_id for message in messages), "metadata without a frame_id"
    frame_ids = [int(message.frame_id) for message in messages]
    assert all(a < b for a, b in zip(frame_ids, frame_ids[1:])), frame_ids[:10]
    points = [
        keypoint
        for message in messages
        for pose in json.loads(message.payload)["data"]["poses"]
        for keypoint in pose["keypoints"]
    ]
    assert points and all(0 <= p["x"] < width and 0 <= p["y"] < height for p in points)
    # The mapping from model-input pixels shows only on a stream larger than the model input.
    assert width > MODEL_SIZE[0] or height > MODEL_SIZE[1], (
        f"the {width}x{height} test stream must be larger than the "
        f"{MODEL_SIZE[0]}x{MODEL_SIZE[1]} model input"
    )
    # Source-frame points reach past the model frame; model-pixel points never would.
    assert any(p["x"] >= MODEL_SIZE[0] or p["y"] >= MODEL_SIZE[1] for p in points)
    # The model input is the whole frame stretched, so points reach the frame's top and bottom
    # tenth; with a letterbox those rows would be padding, and the stretch mapping would leave them
    # empty. This needs a scene with texture near the top and bottom edges, as the test stream has.
    assert any(p["y"] < height // 10 for p in points)
    assert any(p["y"] >= height - height // 10 for p in points)
    # The H.264 stream goes to Insight as received (RTP payload type 96).
    assert headers and len(headers[0]) == 12 and headers[0][0] >> 6 == 2
    assert headers[0][1] & 0x7F == 96
