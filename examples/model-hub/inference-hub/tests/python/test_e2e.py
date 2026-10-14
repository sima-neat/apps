"""End-to-end inference hub tests."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
MODELS_DIR = EXAMPLE_DIR / "src" / "python" / "hub_models"
METADATA = {"classification": "top_classes", "object-detection": "objects"}


def _runtime_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _installed(models_dir: Path, name: str, variant: str) -> bool:
    model_cfg = yaml.safe_load((MODELS_DIR / name / "config.yaml").read_text(encoding="utf-8"))
    model_id = model_cfg["variants"][variant]
    return all(
        any((models_dir / model_id / "components" / component).glob("*_mpk.tar.gz"))
        for component in model_cfg["components"]
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("name", "variant", "metadata_type", "codec"),
    [
        ("yolo26", "n", "object-detection", "h264"),
        ("rf_detr", "nano", "object-detection", "h264"),
        ("resnet50", "default", "classification", "h264"),
        ("yolo26", "n", "object-detection", "h265"),
        ("yolo26", "n", "object-detection", "mjpeg"),
    ],
)
def test_publishes_insight_video_and_metadata(
    name,
    variant,
    metadata_type,
    codec,
    models_dir,
    test_timeout_ms,
    skip_unless_e2e_ready,
    e2e_config_writer,
    request,
):
    skip_unless_e2e_ready(_runtime_ready(), "cv2, numpy, and pyneat are required")
    skip_unless_e2e_ready(_installed(models_dir, name, variant), f"{name} {variant} is not installed in {models_dir}")
    rtsp_url = request.getfixturevalue(f"rtsp_{codec}_url")

    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            "model": {"name": name, "variant": variant, "dir": str(models_dir)},
            "source": {"type": "rtsp", "rtsp": {"url": rtsp_url, "codec": codec}},
            "inference": {"frames": 20},
            "insight": {"host": "127.0.0.1", "video_port": video_port, "metadata_port": metadata_port},
        }
    )

    data_key = METADATA[metadata_type]
    with MetadataJsonListener(
        "127.0.0.1",
        metadata_port,
        num_ports=1,
        metadata_type=metadata_type,
        data_array_key=data_key,
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
        assert result.returncode == 0, (
            f"{name} {variant} exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        metadata = listener.wait_for_messages(5.0)
        video_packet = video.recv(65536)

    assert metadata.success, metadata.error
    assert "published=20 " in result.stdout
    # VideoSender uses RTP H.265 (98) or H.264 (96). MJPEG is re-encoded as H.264.
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
    assert video_packet[1] & 0x7F == (98 if codec == "h265" else 96)
    last = metadata.messages[-1]
    assert last.frame_id.isdecimal() and last.timestamp_ms >= 0
    for entry in json.loads(last.payload)["data"][data_key]:
        assert entry.get("label")
        assert 0.0 <= entry.get("confidence", -1.0) <= 1.0
        if metadata_type == "object-detection":
            assert len(entry.get("bbox", [])) == 4
