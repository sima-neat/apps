"""E2E test for face-recognizer (Python).

Requires a live H.264 RTSP source and compiled models; always skipped in CI.
Run manually on-device:

    export SIMANEAT_APPS_TEST_MODELS_DIR=/path/to/models
    export SIMANEAT_TEST_RTSP_H264_URL=rtsp://<host>:<port>/<stream>
    # optional — enables recognition check:
    export SIMANEAT_APPS_TEST_GALLERY_BIN=/path/to/gallery.bin
    pytest tests/python/test_e2e.py -m e2e -v

Required env vars (any missing → test is skipped or fails in strict mode):
  SIMANEAT_APPS_TEST_MODELS_DIR  directory containing model tar.gz files
  SIMANEAT_TEST_RTSP_H264_URL    live H.264 RTSP stream URL

Optional:
  SIMANEAT_APPS_TEST_GALLERY_BIN  pre-enrolled gallery.bin; when set, at least
                                   one non-Unknown recognition result is required
  SIMANEAT_APPS_TEST_TIMEOUT_MS   per-run timeout in ms (default 120 000)
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

_SCRFD_FILE   = "scrfd_2.5g_bnkps.mla_mpk.tar.gz"
_ARCFACE_FILE = "w600k_r50.surgery_mpk.tar.gz"
_BUNDLED_GALLERY = EXAMPLE_DIR / "tests" / "assets" / "gallery.bin"


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy", "pyneat"))


def _find_model(models_dir: Path, filename: str) -> Path | None:
    direct = models_dir / filename
    if direct.exists():
        return direct.resolve()
    for p in models_dir.rglob(filename):
        return p.resolve()
    return None


@pytest.mark.e2e
class TestE2E:
    def test_rtsp_h264_recognition(
        self,
        rtsp_h264_url,
        e2e_config_writer,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
    ):
        """Pipeline starts, runs 60 frames, exits 0; recognition check when gallery set."""
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

        # Prefer explicit env override; fall back to the bundled test gallery so
        # recognition is verified even in CI without extra setup.
        gallery_raw = os.environ.get("SIMANEAT_APPS_TEST_GALLERY_BIN", "").strip()
        if gallery_raw:
            gallery_path: str | None = str(Path(gallery_raw).resolve())
        elif _BUNDLED_GALLERY.exists():
            gallery_path = str(_BUNDLED_GALLERY)
        else:
            gallery_path = None

        # Pass absolute model paths so config resolution works from the temp config dir.
        overrides: dict = {
            "scrfd":   {"model": str(scrfd_path)},
            "arcface": {"model": str(arcface_path)},
            "input":   {"uri": rtsp_h264_url},
            "output":  {"insight": {"host": ""}},  # headless — no encoder/sender init
        }
        if gallery_path:
            overrides["gallery"] = {"path": gallery_path}

        config_path = e2e_config_writer(overrides)

        cmd = [
            sys.executable, str(MAIN_PY),
            "--config", str(config_path),
            "--test", "--max-frames", "60",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # "processed=N fps=X.X" is always printed at shutdown
        assert "processed=" in result.stdout, (
            "Pipeline summary not found — pipeline may not have started.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        # Optional: at least one non-Unknown recognition result must appear
        if gallery_path:
            has_known = any(
                "→" in line
                and line.split("→", 1)[-1].strip().split()[0] not in ("", "Unknown")
                for line in result.stdout.splitlines()
                if "face[" in line
            )
            assert has_known, (
                "SIMANEAT_APPS_TEST_GALLERY_BIN is set but no non-Unknown recognition "
                "result found in 60 frames.\n"
                "Ensure the RTSP stream contains enrolled faces, or unset "
                "SIMANEAT_APPS_TEST_GALLERY_BIN to skip this check.\n"
                f"stdout:\n{result.stdout}"
            )
