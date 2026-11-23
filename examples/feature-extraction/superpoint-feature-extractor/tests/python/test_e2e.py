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

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.rtp_video import drain_video, pairs, rtp_timestamps

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 100
# SuperPoint's input frame, which the decoder reports its keypoints in.
MODEL_SIZE = (640, 480)
# The decoder keeps at most 600 points per frame (the A65V1 profile's top_k).
MAX_POINTS = 600


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@pytest.mark.e2e
def test_publishes_insight_video_and_feature_points(
    rtsp_h264_url, e2e_model_path, test_timeout_ms, e2e_config_writer
):
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            # Absolute, because the example resolves a relative model.path from its config.
            "model": {"path": str(Path(e2e_model_path).resolve())},
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
            require_all_ports=True,
            min_object_count=1,
        ) as listener,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video,
    ):
        video.bind(("127.0.0.1", video_port))
        headers: list[bytes] = []
        stop = threading.Event()
        drain = threading.Thread(target=drain_video, args=(video, stop, headers))
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

    summary = re.search(
        rf"frames={FRAMES} average_points=([0-9.]+) descriptor_dim=256 "
        rf"video_sender=127\.0\.0\.1:{video_port}$",
        result.stdout,
        re.MULTILINE,
    )
    assert summary, result.stdout
    assert 0 < float(summary.group(1)) <= MAX_POINTS
    width, height = map(int, re.search(r"stream=(\d+)x(\d+)@", result.stdout).groups())

    assert metadata.success, metadata.error
    assert following.success, following.error
    first, last = metadata.messages[-1], following.messages[-1]
    assert first.frame_id and last.frame_id, "metadata without a frame_id"
    assert int(last.frame_id) > int(first.frame_id)
    messages = metadata.messages + following.messages + rest.messages
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
    # Insight draws each message's points on the video frame its timestamp names.
    timestamps = rtp_timestamps(headers)
    unpaired = [m.timestamp_ms for m in messages if not pairs(m.timestamp_ms, timestamps)]
    assert not unpaired, f"metadata timestamps with no video frame: {unpaired[:5]}"
