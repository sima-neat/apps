"""E2E tests for single-stream-instance-segmenter (Python).

Every case runs one model family over one input path and checks the three outputs the
application advertises: annotated frames, Insight segmentation metadata, and Insight video.
"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

# Model families and the packages tests/test-scope.yaml downloads for them.
FAMILY_CASES = (
    ("yolo26", "yolo26m-seg-bf16-b1.tar.gz"),
    ("yolov8", "yolo_v8n_seg_mpk.tar.gz"),
)

# Input paths the application supports, and the fixture that supplies each URL.
SOURCE_CASES = (
    ("rtsp", "h264", "rtsp_h264_url"),
    ("rtsp", "mjpeg", "rtsp_mjpeg_url"),
    ("http", "mjpeg", "http_mjpeg_url"),
)


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
@pytest.mark.parametrize(("family", "model_file"), FAMILY_CASES)
@pytest.mark.parametrize(("source_type", "codec", "stream_fixture"), SOURCE_CASES)
class TestE2E:
    def test_publishes_frames_video_and_metadata(
        self,
        family,
        model_file,
        source_type,
        codec,
        stream_fixture,
        models_dir,
        tmp_output_dir,
        test_timeout_ms,
        e2e_config_section,
        e2e_config_writer,
        skip_unless_e2e_ready,
        request,
    ):
        model_path = models_dir / model_file
        skip_unless_e2e_ready(model_path.is_file(), f"missing {family} package: {model_path}")
        source_url = request.getfixturevalue(stream_fixture)

        example = "single-stream-instance-segmenter"
        frames = int(e2e_config_section(example, "testing.e2e.inference")["frames"])
        expected_frames = int(
            e2e_config_section(example, "testing.e2e.output")["total_saved_frames"]
        )
        # This test is the Insight receiver, so it publishes to its own loopback address.
        insight_host = "127.0.0.1"
        video_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
        metadata_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
        config_path = e2e_config_writer(
            {
                "model": {"family": family, "path": str(model_path)},
                "source": {
                    "type": source_type,
                    "codec": codec,
                    "url": source_url,
                    # HTTPS MJPEG sources commonly carry a self-signed certificate.
                    "ssl_strict": source_type != "http",
                },
                "output": {
                    "save_dir": str(tmp_output_dir),
                    "insight": {
                        "host": insight_host,
                        "video_port": video_port,
                        "metadata_port": metadata_port,
                    },
                },
            }
        )
        command = [sys.executable, str(MAIN_PY), "--config", str(config_path)]

        with MetadataJsonListener(
            insight_host,
            metadata_port,
            num_ports=1,
            metadata_type="segmentation",
            data_array_key="segments",
            require_all_ports=True,
            min_object_count=1,
        ) as listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video:
            video.bind((insight_host, video_port))
            video.settimeout(5.0)
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
                check=False,
            )
            metadata = listener.wait_for_messages(5.0)
            video_packet = video.recv(65536)

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert f"model={family}" in result.stdout
        assert f"processed={frames}" in result.stdout

        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
        assert len(output_files) >= expected_frames
        assert all(path.stat().st_size > 0 for path in output_files)

        # VideoSender always re-encodes to RTP H.264 (payload type 96) for Insight.
        assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
        assert video_packet[1] & 0x7F == 96

        assert metadata.success, metadata.error
        for message in metadata.messages:
            assert message.frame_id.isdecimal() and message.timestamp_ms >= 0
        segments = json.loads(metadata.messages[-1].payload)["data"]["segments"]
        assert segments
        for segment in segments:
            assert segment["label"]
            assert 0.0 <= segment["confidence"] <= 1.0
            assert len(segment["bbox"]) == 4
            assert all(value >= 0 for value in segment["bbox"])
            assert segment["mask_format"] == "polygon"
            assert len(segment["mask"]) >= 3
            assert all(
                len(point) == 2 and all(value >= 0 for value in point)
                for point in segment["mask"]
            )
