"""E2E tests for face-recognizer (Python).

1. Bundled face clip (tests/assets/face_clip.mp4) with the bundled gallery: every
   frame must contain both enrolled identities, PersonA and PersonB, and no other name.
2. RTSP stream (test_live_pipeline[rtsp]): the live-stream graph must process 60 frames and exit 0. The shared
   CI stream content is not controlled, so detections are not asserted there.
3. USB camera (collected only when SIMANEAT_TEST_USB_CAMERA names one; CI has no camera):
   the headless graph must process 60 frames, and with Insight on loopback it must also
   forward the camera's JPEG as RTP/JPEG (payload type 26).

Env vars (missing → skipped locally, failed in strict mode):
  SIMANEAT_APPS_TEST_MODELS_DIR  directory containing the model tar.gz files
  SIMANEAT_TEST_RTSP_H264_URL    live H.264 RTSP stream URL (RTSP test only)
  SIMANEAT_APPS_TEST_TIMEOUT_MS  per-run timeout in ms
  SIMANEAT_TEST_USB_CAMERA       USB camera ID from usb_camera_ids() (USB test only)
  SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT  loopback Insight video port (USB test, default 9000)
"""

from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.pytest_fixtures import usb_camera_cases

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
ASSETS = EXAMPLE_DIR / "tests" / "assets"
CLIP = ASSETS / "face_clip.mp4"
GALLERY = ASSETS / "gallery.bin"
ENROLLED_NAMES = ("PersonA", "PersonB")
FRAMES = 60

_SCRFD_FILE   = "scrfd_2.5g_bnkps.mla_mpk.tar.gz"
_ARCFACE_FILE = "w600k_r50.surgery_mpk.tar.gz"
_FACE_LINE = re.compile(r"face\[\d+\] → (\S+)")


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy", "pyneat"))


def _find_model(models_dir: Path, filename: str) -> Path | None:
    direct = models_dir / filename
    if direct.exists():
        return direct.resolve()
    for p in models_dir.rglob(filename):
        return p.resolve()
    return None


def _run(input_uri: str, models_dir: Path, e2e_config_writer, timeout_ms: int,
         skip_unless_e2e_ready, usb_camera: str = "", insight_port: int = 0,
         ) -> subprocess.CompletedProcess:
    skip_unless_e2e_ready(
        _runtime_deps_ready(),
        "Python runtime dependencies (cv2, numpy, pyneat) are not available",
    )
    scrfd_path   = _find_model(models_dir, _SCRFD_FILE)
    arcface_path = _find_model(models_dir, _ARCFACE_FILE)
    skip_unless_e2e_ready(
        scrfd_path is not None and arcface_path is not None,
        f"SCRFD or ArcFace model not found under {models_dir}",
    )
    assert CLIP.is_file() and GALLERY.is_file(), f"bundled test assets missing under {ASSETS}"

    config_path = e2e_config_writer({
        "scrfd":   {"model": str(scrfd_path)},
        "arcface": {"model": str(arcface_path)},
        "gallery": {"path": str(GALLERY)},
        "input":   {"uri": input_uri, "usb": {"camera": usb_camera}},
        # This test is the Insight video receiver, so it publishes to its own loopback.
        "output":  {"insight": {"host": "127.0.0.1" if insight_port else "",
                                "video_port": insight_port or 9000}},
    })
    mode = [] if insight_port else ["--test"]  # --test clears the Insight host
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path),
         *mode, "--max-frames", str(FRAMES)],
        capture_output=True, text=True, timeout=timeout_ms / 1000, cwd=str(EXAMPLE_DIR),
    )
    assert result.returncode == 0, (
        f"main.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    processed = re.search(r"processed=(\d+)", result.stdout)
    assert processed and int(processed.group(1)) == FRAMES, (
        f"expected processed={FRAMES}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.mark.e2e
class TestE2E:
    def test_bundled_clip_recognition(
        self, e2e_config_writer, models_dir, test_timeout_ms, skip_unless_e2e_ready,
    ):
        """Every frame of the bundled clip has PersonA and PersonB recognized."""
        result = _run(str(CLIP), models_dir, e2e_config_writer, test_timeout_ms,
                      skip_unless_e2e_ready)
        names = _FACE_LINE.findall(result.stdout)
        counts = {name: names.count(name) for name in set(names)}
        assert counts == {name: FRAMES for name in ENROLLED_NAMES}, (
            f"expected {ENROLLED_NAMES} each recognized in all {FRAMES} frames; "
            f"got {counts}\nstdout:\n{result.stdout}"
        )

    @pytest.mark.parametrize(
        "source",
        [pytest.param({"name": "rtsp", "fixture": "rtsp_h264_url"}, id="rtsp")]
        + usb_camera_cases({"name": "usb_camera", "fixture": "usb_camera_id"}),
    )
    def test_live_pipeline(
        self, source, request, e2e_config_writer, models_dir, test_timeout_ms,
        skip_unless_e2e_ready,
    ):
        """The live graph processes the requested frames and exits cleanly; a USB camera
        also forwards its JPEG to Insight as RTP/JPEG."""
        value = request.getfixturevalue(source["fixture"])
        if source["name"] == "rtsp":
            _run(value, models_dir, e2e_config_writer, test_timeout_ms, skip_unless_e2e_ready)
            return

        _run("", models_dir, e2e_config_writer, test_timeout_ms, skip_unless_e2e_ready,
             usb_camera=value)
        port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "") or 9000)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", port))
            sock.setblocking(False)
            _run("", models_dir, e2e_config_writer, test_timeout_ms, skip_unless_e2e_ready,
                 usb_camera=value, insight_port=port)
            try:
                packet = sock.recv(65536)
            except BlockingIOError:
                packet = b""
        assert len(packet) > 12 and packet[0] >> 6 == 2 and packet[1] & 0x7F == 26, (
            "USB camera run did not publish RTP/JPEG (payload type 26) to Insight"
        )
