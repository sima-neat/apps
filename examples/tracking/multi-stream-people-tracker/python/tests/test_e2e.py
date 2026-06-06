"""E2E tests for multi-stream-people-tracker (Python)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    for model in models_dir.iterdir():
        if pattern in model.name and "seg" not in model.name and model.name.endswith(".tar.gz"):
            return model
    return None


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    def test_multi_stream_insight_and_save_pipeline(
        self,
        models_dir,
        tmp_output_dir,
        rtsp_urls,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        model = _find_model(models_dir, "yolo_v8")
        skip_unless_e2e_ready(model is not None, "yolo detector model not found in models_dir")
        skip_unless_e2e_ready(len(rtsp_urls) >= 2, "need at least two RTSP URLs for multistream e2e")

        inference = e2e_config_section("multi-stream-people-tracker", "inference")
        tracking = e2e_config_section("multi-stream-people-tracker", "tracking")
        config_path = e2e_config_writer(
            {
                "model": str(model),
                "streams": rtsp_urls[:2],
                "input": {"tcp": True, "latency_ms": 200},
                "inference": {"frames": 10, "profile": False, **inference},
                "tracking": tracking,
                "output": {
                    "insight": {
                        "host": "127.0.0.1",
                        "video_port_base": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port_base": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100
                        ),
                    },
                    "debug_dir": str(tmp_output_dir),
                    "save_every": 2,
                },
            }
        )

        cmd = [
            sys.executable,
            str(MAIN_PY),
            "--config",
            str(config_path),
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

        files = [
            path
            for path in tmp_output_dir.rglob("*")
            if path.is_file() and path.name != "config.yaml"
        ]
        assert files, "Expected sampled output files but output directory is empty"
