"""End-to-end test: an RTSP H.264 stream in, H.264 video and person segments out to Insight."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
MODEL = "efficientsam3_mpk.tar.gz"
TEXT_ENCODER = "efficientsam3_text_encoder_mpk.tar.gz"
FRAMES = 40


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@pytest.mark.e2e
def test_publishes_insight_video_and_segments(
    rtsp_h264_url, models_dir, test_timeout_ms, skip_unless_e2e_ready, e2e_config_writer
):
    model, text_encoder = models_dir / MODEL, models_dir / TEXT_ENCODER
    for path in (model, text_encoder):
        skip_unless_e2e_ready(path.exists(), f"missing EfficientSAM3 artifact: {path}")
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            "model": {"path": str(model), "text_encoder": str(text_encoder)},
            "prompt": {"text": "person"},
            "source": {"rtsp_url": rtsp_h264_url},
            "inference": {"frames": FRAMES},
            "output": {
                "insight": {"host": "127.0.0.1", "video_port": video_port, "metadata_port": metadata_port}
            },
        }
    )

    with MetadataJsonListener(
        "127.0.0.1",
        metadata_port,
        num_ports=1,
        metadata_type="segmentation",
        data_array_key="segments",
        require_all_ports=True,
        min_object_count=1,
    ) as listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video:
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
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        metadata = listener.wait_for_messages(5.0)
        following = listener.wait_for_messages(5.0)
        video_packet = video.recv(65536)

    assert metadata.success, metadata.error
    assert following.success, following.error
    assert f"processed={FRAMES} " in result.stdout
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2 and video_packet[1] & 0x7F == 96
    first, last = metadata.messages[-1], following.messages[-1]
    assert int(last.frame_id) > int(first.frame_id)
    assert last.timestamp_ms > first.timestamp_ms
    for message in metadata.messages + following.messages:
        for segment in json.loads(message.payload)["data"]["segments"]:
            assert segment["label"] == "person"
            assert 0.3 < segment["confidence"] <= 1.0
            assert len(segment["bbox"]) == 4 and all(value >= 0 for value in segment["bbox"])
            assert segment["mask_format"] == "polygon" and len(segment["mask"]) >= 3
            assert all(len(point) == 2 and min(point) >= 0 for point in segment["mask"])
