"""End-to-end RF-DETR detection and segmentation tests."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
LABELS = set((EXAMPLE_DIR / "src" / "common" / "coco-labels.txt").read_text().splitlines())
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def _runtime_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("numpy", "pyneat")
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


# The registry build profile each segmentation size ships with in config.yaml.
SEGMENTATION_PROFILES = {
    "nano": "w8a16_w8a8_mixed",
    "small": "w8a16_w8a8_mixed",
    "medium": "w8a16_w8a8_mixed",
}


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("task", "variant", "codec", "stream_fixture"),
    [
        ("detection", variant, codec, f"rtsp_{codec}_url")
        for codec in ("h264", "h265", "mjpeg")
        for variant in ("nano", "small", "medium", "large")
    ]
    + [
        ("segmentation", variant, codec, f"rtsp_{codec}_url")
        for codec in ("h264", "h265", "mjpeg")
        for variant in ("nano", "small", "medium")
    ],
)
def test_publishes_insight_video_and_metadata(
    task,
    variant,
    codec,
    stream_fixture,
    models_dir,
    test_timeout_ms,
    skip_unless_e2e_ready,
    e2e_config_writer,
    request,
):
    skip_unless_e2e_ready(_runtime_ready(), "numpy and pyneat are required")
    package = (
        f"rf_detr_{variant}_w8a8_mixed_w16a16"
        if task == "detection"
        else f"rf_detr_seg_{variant}_{SEGMENTATION_PROFILES[variant]}"
    )
    backbone = models_dir / f"{package}_backbone_mpk.tar.gz"
    for path in (backbone, models_dir / f"{package}_decoder_mpk.tar.gz"):
        skip_unless_e2e_ready(path.exists(), f"missing RF-DETR artifact: {path}")
    rtsp_url = request.getfixturevalue(stream_fixture)
    model_id = f"{'det' if task == 'detection' else 'seg'}-{variant}"

    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            "model": {"id": model_id, "path": str(backbone)},
            "input": {"source": rtsp_url, "codec": codec},
            # The config comments recommend 100 results per frame for detection, 24 for
            # segmentation.
            "inference": {
                "frames": 20,
                "min_score": 0.2,
                "max_detections": 100 if task == "detection" else 24,
            },
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
        }
    )

    metadata_type = "object-detection" if task == "detection" else "segmentation"
    data_key = "objects" if task == "detection" else "segments"
    with (
        MetadataJsonListener(
            "127.0.0.1",
            metadata_port,
            num_ports=1,
            metadata_type=metadata_type,
            data_array_key=data_key,
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
            f"RF-DETR {task} {variant} exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        metadata = listener.wait_for_messages(5.0)
        following = listener.wait_for_messages(5.0)
        video_packet = video.recv(65536)

    assert metadata.success, metadata.error
    assert following.success, following.error
    assert f"RF-DETR {model_id}: sent 20 results to Insight" in result.stdout
    # VideoSender uses RTP H.265 (98) or H.264 (96). MJPEG is re-encoded as H.264.
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
    assert video_packet[1] & 0x7F == (98 if codec == "h265" else 96)
    for message in metadata.messages + following.messages:
        assert message.frame_id.isdecimal() and message.timestamp_ms >= 0
    first, last = metadata.messages[-1], following.messages[-1]
    assert int(last.frame_id) > int(first.frame_id)
    assert last.timestamp_ms > first.timestamp_ms
    entries = json.loads(last.payload)["data"][data_key]
    for entry in entries:
        assert entry["label"] in LABELS, f"label not in coco-labels.txt: {entry['label']}"
        assert 0.2 <= entry["confidence"] <= 1.0, "confidence below inference.min_score"
        assert len(entry["bbox"]) == 4
        if task == "segmentation":
            assert entry["mask_format"] == "polygon" and len(entry["mask"]) >= 3
