"""E2E tests for single-stream-object-detector (Python)."""

import os
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
SOURCE_CASES = [
    pytest.param(
        {
            "name": "rtsp_h264",
            "type": "rtsp",
            "codec": "h264",
            "url_fixture": "rtsp_h264_url",
        },
        id="rtsp-h264",
    ),
    pytest.param(
        {
            "name": "rtsp_h265",
            "type": "rtsp",
            "codec": "h265",
            "url_fixture": "rtsp_h265_url",
        },
        id="rtsp-h265",
    ),
    pytest.param(
        {
            "name": "rtsp_mjpeg",
            "type": "rtsp",
            "codec": "mjpeg",
            "url_fixture": "rtsp_mjpeg_url",
        },
        id="rtsp-mjpeg",
    ),
    pytest.param(
        {
            "name": "http_mjpeg",
            "type": "http",
            "codec": "mjpeg",
            "url_fixture": "http_mjpeg_url",
            "fps": 30,
            "ssl_strict": False,
        },
        id="http-mjpeg",
    ),
]


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_str_or_default(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


@pytest.mark.e2e
class TestE2E:
    @pytest.mark.parametrize("source", SOURCE_CASES)
    def test_full_pipeline(
        self,
        request,
        source,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        e2e_config_section,
        e2e_config_writer,
        run_until_output_files,
    ):
        source_url = request.getfixturevalue(source["url_fixture"])
        output_cfg = e2e_config_section("single-stream-object-detector", "testing.e2e.output")
        source_config = {
            "type": source["type"],
            "codec": source["codec"],
            "url": source_url,
            "ssl_strict": source.get("ssl_strict", True),
        }
        if source.get("fps", 0) > 0:
            source_config["fps"] = source["fps"]

        config_path = e2e_config_writer(
            {
                "source": source_config,
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
                    }
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
            f"{source['name']} main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
        assert len(output_files) >= int(output_cfg["total_saved_frames"])
        assert all(path.stat().st_size > 0 for path in output_files)
