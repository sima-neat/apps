"""E2E tests for face-recognizer (Python).

1. Enrollment from a PersonA/PersonB folder of the shared COCO images, then the bundled
   face clip (tests/assets/face_clip.mp4) recognized with that gallery: every frame must
   name both PersonA and PersonB, and no other name. Enrollment from the clip itself
   must also write a gallery.
2. RTSP stream (test_live_pipeline[rtsp]): the live-stream graph must process 60 frames
   and exit 0. The shared CI stream content is not controlled, so detections are not
   asserted there. With Insight on loopback, this test listens where Insight would: the
   stream must arrive as RTP/H.264 and face-metadata messages must carry the timestamp
   of a video frame, which is how Insight pairs a result with its frame.
3. USB camera (collected only when SIMANEAT_TEST_USB_CAMERA names one; CI has no camera):
   the headless graph must process 60 frames, and with Insight on loopback it must also
   forward the camera's JPEG as RTP/JPEG (payload type 26).

Env vars (missing → skipped locally, failed in strict mode):
  SIMANEAT_APPS_TEST_MODELS_DIR  directory containing the model tar.gz files
  SIMANEAT_TEST_RTSP_H264_URL    live H.264 RTSP stream URL (RTSP test only)
  SIMANEAT_APPS_TEST_TIMEOUT_MS  per-run timeout in ms
  SIMANEAT_TEST_USB_CAMERA       USB camera ID from usb_camera_ids() (USB test only)
  SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT / _METADATA_PORT  loopback Insight ports
"""

from __future__ import annotations

import bisect
import importlib.util
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
ASSETS = EXAMPLE_DIR / "tests" / "assets"
CLIP = ASSETS / "face_clip.mp4"
COCO = EXAMPLE_DIR.parents[2] / "assets" / "datasets" / "coco"
# The enrolled identities and the shared COCO image each is enrolled from.
IDENTITIES = {"PersonA": "000000493864.jpg", "PersonB": "000000116439.jpg"}
FRAMES = 60

_SCRFD_FILE   = "scrfd_2.5g_bnkps.mla_mpk.tar.gz"
_ARCFACE_FILE = "w600k_r50.surgery_mpk.tar.gz"
_FACE_LINE = re.compile(r"face\[\d+\] → (\S+)")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "").strip() or default)


def _find_model(models_dir: Path, filename: str) -> Path | None:
    direct = models_dir / filename
    if direct.exists():
        return direct.resolve()
    for p in models_dir.rglob(filename):
        return p.resolve()
    return None


def _config(models_dir: Path, e2e_config_writer, skip_unless_e2e_ready, gallery: Path,
            input_uri: str = "", usb_camera: str = "",
            insight_ports: tuple[int, int] | None = None) -> Path:
    skip_unless_e2e_ready(
        all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy", "pyneat")),
        "Python runtime dependencies (cv2, numpy, pyneat) are not available",
    )
    scrfd_path   = _find_model(models_dir, _SCRFD_FILE)
    arcface_path = _find_model(models_dir, _ARCFACE_FILE)
    skip_unless_e2e_ready(
        scrfd_path is not None and arcface_path is not None,
        f"SCRFD or ArcFace model not found under {models_dir}",
    )
    assert CLIP.is_file(), f"bundled test clip missing under {ASSETS}"
    video_port, metadata_port = insight_ports or (9000, 9100)
    return e2e_config_writer({
        "scrfd":   {"model": str(scrfd_path)},
        "arcface": {"model": str(arcface_path)},
        "gallery": {"path": str(gallery)},
        "input":   {"uri": input_uri, "usb": {"camera": usb_camera}},
        # With Insight enabled this test is the receiver, on its own loopback.
        "output":  {"insight": {"host": "127.0.0.1" if insight_ports else "",
                                "video_port": video_port, "metadata_port": metadata_port}},
    })


def _main(config_path: Path, args: list[str], timeout_ms: int,
          while_running=None) -> subprocess.CompletedProcess:
    """Run main.py to completion. `while_running` is called every 100 ms during the run."""
    cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path), *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=str(EXAMPLE_DIR))
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if while_running:
                while_running()
            if time.monotonic() > deadline:
                proc.kill()
                stdout, stderr = proc.communicate()
                raise AssertionError(
                    f"main.py {' '.join(args)} did not finish in {timeout_ms} ms\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    assert result.returncode == 0, (
        f"main.py {' '.join(args)} exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _recognize(config_path: Path, args: list[str], timeout_ms: int,
               while_running=None) -> subprocess.CompletedProcess:
    result = _main(config_path, [*args, "--max-frames", str(FRAMES)], timeout_ms, while_running)
    processed = re.search(r"processed=(\d+)", result.stdout)
    assert processed and int(processed.group(1)) == FRAMES, (
        f"expected processed={FRAMES}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _gallery_names(path: Path) -> list[str]:
    """The people in a gallery.bin (header, then per person: name, embedding, sample
    count, unnormalized mean)."""
    data = path.read_bytes()
    assert data[:8] == b"FRGAL1\n\x00", f"{path} is not a gallery"
    (count,) = struct.unpack_from("<I", data, 12)
    names, offset = [], 16
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", data, offset)
        names.append(data[offset + 2:offset + 2 + name_len].decode("utf-8"))
        offset += 2 + name_len + 512 * 4 + 4 + 512 * 4
    assert offset == len(data), f"{path} has trailing or missing bytes"
    return names


def _video_socket(port: int) -> socket.socket:
    """A UDP socket bound where Insight would listen for video."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    sock.bind(("127.0.0.1", port))
    sock.setblocking(False)
    return sock


def _rtp_packets(sock: socket.socket) -> list[bytes]:
    packets = []
    while True:
        try:
            packet = sock.recv(65536)
        except BlockingIOError:
            return packets
        if len(packet) > 12 and packet[0] >> 6 == 2:
            packets.append(packet)


@pytest.mark.e2e
class TestE2E:
    def test_enrollment_then_recognition(
        self, tmp_path, e2e_config_writer, models_dir, test_timeout_ms, skip_unless_e2e_ready,
    ):
        """People enrolled from images are recognized in every frame of the bundled clip."""
        enrolled = tmp_path / "gallery_images.bin"
        config_path = _config(models_dir, e2e_config_writer, skip_unless_e2e_ready, enrolled,
                              input_uri=str(CLIP))
        images = tmp_path / "gallery_images"
        for name, filename in IDENTITIES.items():
            assert (COCO / filename).is_file(), f"shared test image missing: {COCO / filename}"
            (images / name).mkdir(parents=True)
            shutil.copy(COCO / filename, images / name / filename)

        _main(config_path, ["--enroll", "--images", str(images)], test_timeout_ms)
        assert sorted(_gallery_names(enrolled)) == sorted(IDENTITIES)

        from_video = tmp_path / "gallery_video.bin"
        _main(config_path, ["--enroll", "--video", str(CLIP), "--name", "TestIdentity",
                            "--sample-every", "10", "--gallery", str(from_video)],
              test_timeout_ms)
        assert _gallery_names(from_video) == ["TestIdentity"]

        result = _recognize(config_path, ["--test"], test_timeout_ms)
        names = _FACE_LINE.findall(result.stdout)
        counts = {name: names.count(name) for name in set(names)}
        assert counts == {name: FRAMES for name in IDENTITIES}, (
            f"expected {tuple(IDENTITIES)} each recognized in all {FRAMES} frames; "
            f"got {counts}\nstdout:\n{result.stdout}"
        )

    @pytest.mark.parametrize(
        "source",
        [pytest.param({"name": "rtsp", "fixture": "rtsp_h264_url"}, id="rtsp")]
        + usb_camera_cases({"name": "usb_camera", "fixture": "usb_camera_id"}),
    )
    def test_live_pipeline(
        self, source, request, tmp_path, e2e_config_writer, models_dir, test_timeout_ms,
        skip_unless_e2e_ready,
    ):
        """The live graph processes the requested frames and exits cleanly, headless and
        with Insight on loopback, where the source's own video and the metadata arrive."""
        value = request.getfixturevalue(source["fixture"])
        is_usb = source["name"] == "usb_camera"
        source_args = {"usb_camera": value} if is_usb else {"input_uri": value}

        # The stream's content is not controlled, so this run needs no enrolled people:
        # without a gallery every face is reported as Unknown.
        gallery = tmp_path / "no_gallery.bin"

        # Headless: --test clears the Insight host.
        headless = _config(models_dir, e2e_config_writer, skip_unless_e2e_ready, gallery,
                           **source_args)
        _recognize(headless, ["--test"], test_timeout_ms)

        ports = (_env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000),
                 _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100))
        with_insight = _config(models_dir, e2e_config_writer, skip_unless_e2e_ready, gallery,
                               insight_ports=ports, **source_args)
        with _video_socket(ports[0]) as video, \
                MetadataJsonListener("127.0.0.1", ports[1], num_ports=1) as metadata_listener:
            # Read the video while the application runs: a whole run does not fit in the
            # socket buffer, and the frames kept must be the ones the metadata describes.
            packets: list[bytes] = []
            _recognize(with_insight, [], test_timeout_ms,
                       while_running=lambda: packets.extend(_rtp_packets(video)))
            packets.extend(_rtp_packets(video))
            metadata = metadata_listener.drain_pending()

        # The source's own stream: RTP/JPEG (26) from a USB camera, RTP/H.264 (96) from RTSP.
        payload_type = 26 if is_usb else 96
        video_timestamps = sorted({struct.unpack(">I", p[4:8])[0]
                                   for p in packets if p[1] & 0x7F == payload_type})
        assert video_timestamps, (
            f"no RTP video with payload type {payload_type} reached the Insight video port"
        )
        assert metadata.messages, f"no face metadata reached Insight: {metadata.error}"
        if is_usb:
            return

        # Insight pairs a metadata message with the video frame whose RTP timestamp
        # (90 kHz) equals the message's millisecond timestamp, within one millisecond.
        def paired(message) -> bool:
            want = (message.timestamp_ms * 90) & 0xFFFFFFFF
            i = bisect.bisect_left(video_timestamps, want - 90)
            return (message.timestamp_ms >= 0 and i < len(video_timestamps)
                    and video_timestamps[i] <= want + 90)

        paired_count = sum(1 for message in metadata.messages if paired(message))
        assert paired_count >= 10, (
            f"only {paired_count} of {len(metadata.messages)} metadata messages carry the "
            f"timestamp of a received video frame ({len(video_timestamps)} frames received)"
        )
