"""E2E tests for fastflow-anomaly-detector (Python).

Runs the single-stream RTSP pipeline against a live RTSP H.264 source
(SIMANEAT_TEST_RTSP_H264_URL) and checks that it writes the annotated frames it also
sends to Insight.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    def test_rtsp_h264_to_insight(
        self,
        rtsp_h264_url,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        e2e_config_section,
        e2e_config_writer,
        run_until_output_files,
    ):
        output_cfg = e2e_config_section("fastflow-anomaly-detector", "testing.e2e.output")
        config_path = e2e_config_writer(
            {
                "model": {"path": str(e2e_model_path)},
                "source": {"rtsp_url": rtsp_h264_url},
                "output": {
                    "save_dir": str(tmp_output_dir),
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port": _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000),
                    },
                },
            }
        )
        cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path)]

        result = run_until_output_files(
            cmd,
            tmp_output_dir,
            int(output_cfg["total_saved_frames"]),
            test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
        assert len(output_files) >= int(output_cfg["total_saved_frames"])
        assert all(path.stat().st_size > 0 for path in output_files)
