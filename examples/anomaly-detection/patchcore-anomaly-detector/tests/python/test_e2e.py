"""End-to-end tests: --calibrate builds the bank, the shipped images get the right verdicts, and
each RTSP codec reaches Insight as video and defect-region metadata."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
APPS_ROOT = EXAMPLE_DIR.parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DATASET_DIR = APPS_ROOT / "assets" / "datasets" / "patchcore"
FRAMES = 60
IMAGE_RE = re.compile(r"^(?P<name>\S+): (?P<verdict>ANOMALOUS|normal) score=", re.MULTILINE)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _run(config_path: Path, *args: str, timeout_s: float) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(APPS_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _config(e2e_config_writer, e2e_model_path, tmp_output_dir: Path, overrides: dict) -> Path:
    """A config whose bank is calibrated into this test's folder from the shipped nominal set."""
    return e2e_config_writer(
        {
            # Absolute, because the example resolves a relative model.path from its config.
            "model": {"path": str(Path(e2e_model_path).resolve())},
            "app": {
                "memory_bank_dir": str(tmp_output_dir / "bank"),
                "nominal_images_dir": str(DATASET_DIR / "nominal"),
            },
            **overrides,
        }
    )


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("folder", "anomalous"),
    [
        ("images", {"scratch_0.png", "scratch_1.png", "scratch_2.png", "scratch_3.png"}),
        # Known-good images the bank never saw score below the threshold.
        ("held_out_normal", set()),
    ],
)
def test_scores_images_and_outlines_defects(
    folder, anomalous, e2e_model_path, test_timeout_ms, e2e_config_writer, tmp_output_dir
):
    output_dir = tmp_output_dir / "outlined"
    config_path = _config(
        e2e_config_writer,
        e2e_model_path,
        tmp_output_dir,
        {
            "input": {"type": "image_dir", "source": str(DATASET_DIR / folder)},
            "output": {"dir": str(output_dir)},
        },
    )
    _run(config_path, "--calibrate", timeout_s=test_timeout_ms / 1000)
    result = _run(config_path, timeout_s=test_timeout_ms / 1000)

    verdicts = {match["name"]: match["verdict"] for match in IMAGE_RE.finditer(result.stdout)}
    inputs = sorted(path.name for path in (DATASET_DIR / folder).glob("*.png"))
    assert sorted(verdicts) == inputs, result.stdout
    assert {name for name, verdict in verdicts.items() if verdict == "ANOMALOUS"} == anomalous
    for name in inputs:
        source = cv2.imread(str(DATASET_DIR / folder / name))
        outlined = cv2.imread(str(output_dir / name))
        assert outlined is not None and outlined.shape == source.shape, name
        # Only an anomalous image is drawn on, and its outline covers a part of it.
        changed = float(np.any(source != outlined, axis=2).mean())
        assert (0.0 < changed < 0.5) if name in anomalous else changed == 0.0, (name, changed)


@pytest.mark.e2e
@pytest.mark.parametrize("codec", ["h264", "h265", "mjpeg"])
def test_publishes_insight_video_and_regions(
    request, codec, e2e_model_path, test_timeout_ms, e2e_config_writer, tmp_output_dir
):
    rtsp_url = request.getfixturevalue(f"rtsp_{codec}_url")
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_path = _config(
        e2e_config_writer,
        e2e_model_path,
        tmp_output_dir,
        {
            "input": {"type": "rtsp", "source": rtsp_url, "codec": codec},
            "inference": {"frames": FRAMES},
            # This test is the Insight receiver, so the example publishes to loopback.
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
        },
    )
    _run(config_path, "--calibrate", timeout_s=test_timeout_ms / 1000)

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
        result = _run(config_path, timeout_s=test_timeout_ms / 1000)
        metadata = listener.wait_for_messages(5.0)
        rest = listener.drain_pending()
        video_packet = video.recv(65536)

    flagged = int(re.search(rf"processed={FRAMES} flagged=(\d+) ", result.stdout).group(1))
    # The test streams show no part from the nominal set, so the model flags them.
    assert flagged > 0, result.stdout
    assert metadata.success, metadata.error
    width, height = (int(value) for value in re.search(r"stream=(\d+)x(\d+)@", result.stdout).groups())
    segments = [
        segment
        for message in metadata.messages + rest.messages
        for segment in json.loads(message.payload)["data"]["segments"]
    ]
    assert segments, "no frame carried a defect region"
    for segment in segments:
        assert segment["label"] == "anomaly", segment
        # A region lies above the patch threshold, so its confidence is above 0.5.
        assert 0.5 < segment["confidence"] <= 1.0, segment
        x, y, box_width, box_height = segment["bbox"]
        assert 0 <= x and 0 <= y and x + box_width <= width and y + box_height <= height
        assert segment["mask_format"] == "polygon"
        assert all(0 <= px < width and 0 <= py < height for px, py in segment["mask"])
    # VideoSender forwards the stream as received: RTP H.264 (96), H.265 (98), or JPEG (26).
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
    assert video_packet[1] & 0x7F == {"h264": 96, "h265": 98, "mjpeg": 26}[codec]
