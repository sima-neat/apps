"""E2E tests for face-recognizer (Python).

1. Bundled face clip (tests/assets/face_clip.mp4) with the bundled gallery: every
   frame must contain a face recognized as the enrolled identity.
2. RTSP stream: the live-stream graph must process 60 frames and exit 0. The shared
   CI stream content is not controlled, so detections are not asserted there.

Env vars (missing → skipped locally, failed in strict mode):
  SIMANEAT_APPS_TEST_MODELS_DIR  directory containing the model tar.gz files
  SIMANEAT_TEST_RTSP_H264_URL    live H.264 RTSP stream URL (RTSP test only)
  SIMANEAT_APPS_TEST_TIMEOUT_MS  per-run timeout in ms
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
ASSETS = EXAMPLE_DIR / "tests" / "assets"
CLIP = ASSETS / "face_clip.mp4"
GALLERY = ASSETS / "gallery.bin"
ENROLLED_NAME = "Paresh"
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
         skip_unless_e2e_ready) -> subprocess.CompletedProcess:
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
        "input":   {"uri": input_uri},
        "output":  {"insight": {"host": ""}},
    })
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path),
         "--test", "--max-frames", str(FRAMES)],
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
        """Every frame of the bundled clip has a face recognized as the enrolled identity."""
        result = _run(str(CLIP), models_dir, e2e_config_writer, test_timeout_ms,
                      skip_unless_e2e_ready)
        names = _FACE_LINE.findall(result.stdout)
        assert len(names) >= FRAMES and set(names) == {ENROLLED_NAME}, (
            f"expected a face recognized as {ENROLLED_NAME} in all {FRAMES} frames; "
            f"got {len(names)} face results: {sorted(set(names))}\nstdout:\n{result.stdout}"
        )

    def test_rtsp_pipeline(
        self, rtsp_h264_url, e2e_config_writer, models_dir, test_timeout_ms,
        skip_unless_e2e_ready,
    ):
        """The live-stream graph processes the requested frames and exits cleanly."""
        _run(rtsp_h264_url, models_dir, e2e_config_writer, test_timeout_ms,
             skip_unless_e2e_ready)
