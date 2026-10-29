"""E2E tests for single-stream-object-detector (Python)."""

import os
import socket
import sys
from pathlib import Path

import pytest

from tests.utils.pytest_fixtures import usb_camera_cases
from tests.utils.process_assertions import assert_exited_cleanly
from tests.utils.output_assertions import assert_streamed_frames_are_usable

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
] + usb_camera_cases(
    {
        "name": "usb_camera",
        "type": "usb",
        "codec": "mjpeg",
        "url_fixture": "usb_camera_id",
        "fps": 30,
    }
)


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


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
        # The case fills its source type's section; a USB case's "URL" is the camera's stable ID.
        section = {"camera" if source["type"] == "usb" else "url": source_url}
        if source["type"] == "rtsp":
            section["codec"] = source["codec"]
        if source["type"] == "http":
            section["ssl_strict"] = source.get("ssl_strict", True)
        if source.get("fps", 0) > 0:
            section["fps"] = source["fps"]
        source_config = {"type": source["type"], source["type"]: section}

        # This test is the Insight video receiver, so it publishes to its own loopback address.
        insight_host = "127.0.0.1"
        video_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
        config_path = e2e_config_writer(
            {
                "source": source_config,
                "output": {
                    "save_dir": str(tmp_output_dir),
                    "insight": {
                        "host": insight_host,
                        "video_port": video_port,
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

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video:
            video.bind((insight_host, video_port))
            video.settimeout(5.0)
            result = run_until_output_files(
                cmd,
                tmp_output_dir,
                int(output_cfg["total_saved_frames"]),
                test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            video_packet = video.recv(65536)

        assert_exited_cleanly(result)
        assert_streamed_frames_are_usable(
            tmp_output_dir, int(output_cfg["total_saved_frames"])
        )
        # Streams are re-encoded to RTP H.264 (payload type 96); a USB camera's own JPEG is
        # forwarded as RTP/JPEG (payload type 26).
        assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
        assert video_packet[1] & 0x7F == (26 if source["type"] == "usb" else 96)
