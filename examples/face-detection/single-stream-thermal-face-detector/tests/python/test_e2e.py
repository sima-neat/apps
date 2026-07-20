"""E2E tests for single-stream-thermal-face-detector (Python).

Runs the single-stream RTSP pipeline and verifies the app publishes valid
pose-estimation metadata JSON (the 5 facial landmarks) to the Insight metadata
UDP port. A local UDP listener stands in for Insight, so no running viewer is
required -- only an RTSP H.264 source (SIMANEAT_TEST_RTSP_H264_URL) and the model.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"


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
                "source": {"rtsp_url": rtsp_h264_url},
                "inference": {"frames": 140},
                "output": {
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port": video_port,
                        "metadata_port": metadata_port,
                    }
                },
            }
        )

        cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path)]
        with MetadataJsonListener(
            E2E_INSIGHT_HOST,
            metadata_port,
            num_ports=1,
            metadata_type="pose-estimation",
            data_array_key="poses",
            require_all_ports=True,
        ) as metadata_listener:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            metadata = metadata_listener.wait_for_messages(5.0)

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert metadata.success, (
            f"pose-estimation metadata was not received: {metadata.error}"
        )
