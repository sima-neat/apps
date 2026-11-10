"""E2E tests for face-recognizer (Python).

The bundled clip (tests/fixtures/face_clip.mp4) shows the two people of the shared COCO
images 000000493864.jpg (PersonA) and 000000116439.jpg (PersonB) in every frame.
"""

from __future__ import annotations

import bisect
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.pytest_fixtures import usb_camera_cases

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CLIP = EXAMPLE_DIR / "tests" / "fixtures" / "face_clip.mp4"
COCO = EXAMPLE_DIR.parents[2] / "assets" / "datasets" / "coco"
CLIP_FRAMES = 89
LIVE_FRAMES = 60
PAIRED_MESSAGES = 10
SOURCE_CASES = [
    pytest.param({"type": "rtsp", "url_fixture": "rtsp_h264_url"}, id="rtsp-h264"),
] + usb_camera_cases({"type": "usb", "url_fixture": "usb_camera_id"})


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _run_main(config_path: Path, *args: str, timeout_ms: int, while_running=None):
    """Runs main.py to completion; `while_running` is called every 100 ms meanwhile."""
    cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path), *args]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(EXAMPLE_DIR)
    )
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if while_running is not None:
                while_running()
            if time.monotonic() > deadline:
                process.kill()
                stdout, stderr = process.communicate()
                break
    assert process.returncode == 0, (
        f"main.py exited with code {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    return stdout


def _names_per_frame(stdout: str) -> list[list[str]]:
    """The names printed for each frame, from lines such as
    "frame 3: PersonA 0.972, Unknown 0.412"."""
    frames = []
    for line in stdout.splitlines():
        match = re.fullmatch(r"frame \d+: (.*)", line)
        if match:
            frames.append(sorted(re.findall(r"(\S+) -?\d+\.\d+", match.group(1))))
    return frames


@pytest.fixture
def base_config(models_dir, e2e_config_section, skip_unless_e2e_ready, tmp_path):
    """The model and gallery settings every run of a test shares."""
    shipped = e2e_config_section("face-recognizer", "model")
    models = {name: models_dir / Path(shipped[name]).name for name in ("scrfd", "arcface")}
    skip_unless_e2e_ready(
        all(path.is_file() for path in models.values()),
        f"SCRFD or ArcFace model not found under {models_dir}",
    )
    return {
        "model": {name: str(path) for name, path in models.items()},
        "gallery": {"path": str(tmp_path / "gallery.bin")},
    }


@pytest.mark.e2e
class TestE2E:
    def test_enrolled_people_are_named_in_the_clip(
        self, base_config, e2e_config_writer, tmp_path, test_timeout_ms
    ):
        def enroll(person: str, image: str) -> None:
            images = tmp_path / f"images_{person}"
            (images / person).mkdir(parents=True)
            shutil.copy(COCO / image, images / person / image)
            config = e2e_config_writer({**base_config, "enrollment": {"images_dir": str(images)}})
            stdout = _run_main(config, "--enroll", timeout_ms=test_timeout_ms)
            assert "enrolled=1 skipped=0" in stdout, stdout

        def names_in_clip() -> list[list[str]]:
            config = e2e_config_writer(
                {**base_config, "source": {"type": "video_file", "video_file": {"path": str(CLIP)}}}
            )
            return _names_per_frame(_run_main(config, timeout_ms=test_timeout_ms))

        enroll("PersonA", "000000493864.jpg")
        assert names_in_clip() == [["PersonA", "Unknown"]] * CLIP_FRAMES

        # A second enrollment adds to the people already in the gallery.
        enroll("PersonB", "000000116439.jpg")
        assert names_in_clip() == [["PersonA", "PersonB"]] * CLIP_FRAMES

    def test_enrollment_from_a_video(
        self, base_config, e2e_config_writer, tmp_path, test_timeout_ms
    ):
        config = e2e_config_writer(
            {
                **base_config,
                "enrollment": {"video_path": str(CLIP), "name": "PersonC", "sample_every": 10},
            }
        )

        stdout = _run_main(config, "--enroll", timeout_ms=test_timeout_ms)

        # Every tenth of the clip's 89 frames is sampled.
        assert "enrolled=8 skipped=0 people=1" in stdout, stdout
        assert (tmp_path / "gallery.bin").is_file()

    @pytest.mark.parametrize("source", SOURCE_CASES)
    def test_live_source_reaches_insight(
        self, request, source, base_config, e2e_config_writer, test_timeout_ms
    ):
        source_value = request.getfixturevalue(source["url_fixture"])
        # The case fills its source type's section; a USB case's "URL" is the camera's stable ID.
        usb = source["type"] == "usb"
        section = {"camera" if usb else "url": source_value}
        # This test is the Insight receiver, so the application publishes to loopback.
        insight_host = "127.0.0.1"
        video_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
        metadata_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
        config = e2e_config_writer(
            {
                **base_config,
                "source": {"type": source["type"], source["type"]: section},
                "inference": {"frames": LIVE_FRAMES},
                "output": {
                    "insight": {
                        "host": insight_host,
                        "video_port": video_port,
                        "metadata_port": metadata_port,
                    }
                },
            }
        )

        # An RTSP stream is forwarded as RTP H.264 (payload type 96); a USB camera's own JPEG
        # is forwarded as RTP/JPEG (payload type 26).
        payload_type = 26 if usb else 96
        video_timestamps: set[int] = set()

        def read_video() -> None:
            while True:
                try:
                    packet = video.recv(65536)
                except BlockingIOError:
                    return
                if len(packet) > 12 and packet[0] >> 6 == 2 and packet[1] & 0x7F == payload_type:
                    video_timestamps.add(struct.unpack(">I", packet[4:8])[0])

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video, MetadataJsonListener(
            insight_host, metadata_port, num_ports=1
        ) as metadata_listener:
            video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            video.bind((insight_host, video_port))
            video.setblocking(False)
            # The video is read while the application runs: a whole run does not fit in the
            # socket buffer, and the frames kept must be the ones the metadata describes.
            stdout = _run_main(config, timeout_ms=test_timeout_ms, while_running=read_video)
            read_video()
            metadata = metadata_listener.drain_pending()

        assert f"processed={LIVE_FRAMES}" in stdout, stdout
        assert video_timestamps, f"no RTP video with payload type {payload_type} reached Insight"

        # A message pairs with a frame when its millisecond timestamp, in 90 kHz RTP units, is
        # within one millisecond of that frame's RTP timestamp.
        frames = sorted(video_timestamps)

        def paired(message) -> bool:
            rtp = (message.timestamp_ms * 90) & 0xFFFFFFFF
            index = bisect.bisect_left(frames, rtp - 90)
            return message.timestamp_ms >= 0 and index < len(frames) and frames[index] <= rtp + 90

        paired_count = sum(1 for message in metadata.messages if paired(message))
        assert paired_count >= PAIRED_MESSAGES, (
            f"{paired_count} of {len(metadata.messages)} metadata messages carry the timestamp "
            f"of a received video frame ({len(frames)} frames received): {metadata.error}"
        )
