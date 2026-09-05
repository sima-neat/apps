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
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def _runtime_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("numpy", "pyneat")
    )


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("task", "variant", "codec", "stream_fixture"),
    [
        ("detection", variant, codec, f"rtsp_{codec}_url")
        for codec in ("h264", "h265", "mjpeg")
        for variant in ("small", "medium")
    ]
    + [
        ("segmentation", "medium", codec, f"rtsp_{codec}_url")
        for codec in ("h264", "h265", "mjpeg")
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
    model_name = f"rfdetr-{'seg-' if task == 'segmentation' else ''}{variant}"
    paths = {
        "backbone": models_dir / f"{model_name}-backbone.tar.gz",
        "transformer": models_dir / f"{model_name}-transformer.tar.gz",
    }
    for path in paths.values():
        skip_unless_e2e_ready(path.exists(), f"missing RF-DETR artifact: {path}")
    rtsp_url = request.getfixturevalue(stream_fixture)

    model_paths = {key: str(value) for key, value in paths.items()}
    if task == "detection":
        model_config = {"task": task, task: {"variant": variant, variant: model_paths}}
    else:
        model_config = {"task": task, task: model_paths}

    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = e2e_config_writer(
        {
            "model": model_config,
            "source": {"rtsp_url": rtsp_url, "codec": codec},
            "inference": {
                "frames": 20,
                task: {"min_score": 0.2},
            },
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port": video_port,
                    "metadata_port": metadata_port,
                }
            },
        }
    )

    metadata_type = "object-detection" if task == "detection" else "segmentation"
    data_key = "objects" if task == "detection" else "segments"
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
            f"RF-DETR {task} {variant} exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        metadata = listener.wait_for_messages(5.0)
        following = listener.wait_for_messages(5.0)
        video_packet = video.recv(65536)

    assert metadata.success, metadata.error
    assert following.success, following.error
    assert f"RF-DETR {task}: completed=20 " in result.stdout
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
        assert entry.get("label")
        assert 0.0 <= entry.get("confidence", -1.0) <= 1.0
        assert len(entry.get("bbox", [])) == 4
        assert all(value >= 0.0 for value in entry["bbox"])
        if task == "segmentation":
            assert entry.get("mask_format") == "polygon"
            assert len(entry.get("mask", [])) >= 3
            assert all(
                len(point) == 2 and all(value >= 0.0 for value in point)
                for point in entry["mask"]
            )
