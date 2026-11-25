"""End-to-end test: the RTSP stream's faces reach Insight as pose-estimation metadata, next to
H.264 video that the metadata's timestamps pair with.

Local UDP listeners stand in for Insight, so only an RTSP H.264 source
(SIMANEAT_TEST_RTSP_H264_URL) and the model are needed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
LABELS = (EXAMPLE_DIR / "src" / "common" / "face_label.txt").read_text().splitlines()
LANDMARK_NAMES = ["eye_l", "eye_r", "nose_tip", "mouth_l", "mouth_r"]
E2E_INSIGHT_HOST = "127.0.0.1"
# The shipped config's inference values, which every pose must respect.
MIN_SCORE = 0.25
MAX_DETECTIONS = 50


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


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    def test_single_stream_metadata_pipeline(
        self,
        e2e_model_path,
        rtsp_h264_url,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        metadata_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
        video_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)

        config_path = e2e_config_writer(
            {
                "input": {"source": rtsp_h264_url},
                "inference": {
                    "frames": 140,
                    "min_score": MIN_SCORE,
                    "max_detections": MAX_DETECTIONS,
                },
                "output": {
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port_base": video_port,
                        "metadata_port_base": metadata_port,
                    }
                },
            }
        )

        cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path)]
        with (
            MetadataJsonListener(
                E2E_INSIGHT_HOST,
                metadata_port,
                num_ports=1,
                metadata_type="pose-estimation",
                data_array_key="poses",
                require_all_ports=True,
            ) as metadata_listener,
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video,
        ):
            video.bind((E2E_INSIGHT_HOST, video_port))
            headers: list[bytes] = []
            stop = threading.Event()
            drain = threading.Thread(target=_drain_video, args=(video, stop, headers))
            drain.start()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=test_timeout_ms / 1000,
                    cwd=str(EXAMPLE_DIR),
                )
            finally:
                stop.set()
                drain.join()
            metadata = metadata_listener.wait_for_messages(5.0)
            # The rest of the run's messages, so the checks below see every frame it sent.
            rest = metadata_listener.drain_pending()

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert metadata.success, f"pose-estimation metadata was not received: {metadata.error}"
        assert "processed=140" in result.stdout, result.stdout
        messages = metadata.messages + rest.messages
        for message in messages:
            poses = json.loads(message.payload)["data"]["poses"]
            assert len(poses) <= MAX_DETECTIONS
            for pose in poses:
                # The label comes from the label file, and each keypoint carries the face's score.
                assert pose["label"] == LABELS[0]
                assert [point["name"] for point in pose["keypoints"]] == LANDMARK_NAMES
                assert all(point["confidence"] > MIN_SCORE for point in pose["keypoints"])
        # The example encodes the decoded frames as H.264 (RTP payload type 96) for Insight.
        assert headers and len(headers[0]) == 12 and headers[0][0] >> 6 == 2
        assert headers[0][1] & 0x7F == 96
        # Insight draws each message's poses on the video frame its timestamp names.
        rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers}
        unpaired = [m.timestamp_ms for m in messages if not _pairs(m.timestamp_ms, rtp_timestamps)]
        assert not unpaired, f"metadata timestamps with no video frame: {unpaired[:5]}"
