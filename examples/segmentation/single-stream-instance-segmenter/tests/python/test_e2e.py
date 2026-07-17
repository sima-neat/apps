"""E2E tests for single-stream-instance-segmenter (Python)."""

import os
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_str_or_default(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline_rtsp_h264(
        self,
        e2e_model_path,
        rtsp_h264_url,
        tmp_output_dir,
        test_timeout_ms,
        e2e_config_section,
        e2e_config_writer,
        run_until_output_files,
    ):
        output_cfg = e2e_config_section("single-stream-instance-segmenter", "testing.e2e.output")
        config_path = e2e_config_writer(
            {
                "source": {"type": "rtsp", "codec": "h264", "url": rtsp_h264_url},
                "output": {
                    "save_dir": str(tmp_output_dir),
                    "insight": {
                        "host": _env_str_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_HOST", "127.0.0.1"
                        ),
                        "video_port": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100
                        ),
                    },
                },
            }
        )
        cmd = [
            sys.executable, str(MAIN_PY),
            "--config", str(config_path),
        ]

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
