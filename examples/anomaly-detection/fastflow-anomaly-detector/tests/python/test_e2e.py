"""End-to-end tests: each RTSP codec reaches Insight as video and defect-region metadata."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 60
THRESHOLD = 0.5
MIN_REGION_PX = 300


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@pytest.mark.e2e
@pytest.mark.parametrize("codec", ["h264", "h265", "mjpeg"])
def test_publishes_insight_video_and_regions(
    request, codec, e2e_model_path, test_timeout_ms, e2e_config_writer
):
    rtsp_url = request.getfixturevalue(f"rtsp_{codec}_url")
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            # Absolute, because the example resolves a relative model.path from its config.
            "model": {"path": str(Path(e2e_model_path).resolve())},
            "input": {"source": rtsp_url, "codec": codec},
            "inference": {"frames": FRAMES},
            # This test is the Insight receiver, so the example publishes to loopback.
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
            "app": {"anomaly_threshold": THRESHOLD, "min_region_px": MIN_REGION_PX},
        }
    )

    with (
        MetadataJsonListener(
            "127.0.0.1",
            metadata_port,
            num_ports=1,
            metadata_type="segmentation",
            data_array_key="segments",
            require_all_ports=True,
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

    assert re.search(rf"processed={FRAMES} flagged=\d+ ", result.stdout), result.stdout
    assert metadata.success, metadata.error
    assert following.success, following.error
    first, last = metadata.messages[-1], following.messages[-1]
    assert int(last.frame_id) > int(first.frame_id)
    width, height = (int(value) for value in re.search(r"stream=(\d+)x(\d+)@", result.stdout).groups())
    for message in metadata.messages + following.messages + rest.messages:
        for segment in json.loads(message.payload)["data"]["segments"]:
            assert segment["label"] == "anomaly", segment
            assert THRESHOLD <= segment["confidence"] <= 1.0, segment
            x, y, box_width, box_height = segment["bbox"]
            assert 0 <= x and 0 <= y and x + box_width <= width and y + box_height <= height
            assert segment["mask_format"] == "polygon"
            assert all(0 <= px < width and 0 <= py < height for px, py in segment["mask"])
    # VideoSender forwards the stream as received: RTP H.264 (96), H.265 (98), or JPEG (26).
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
    assert video_packet[1] & 0x7F == {"h264": 96, "h265": 98, "mjpeg": 26}[codec]
