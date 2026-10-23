"""E2E tests for single-stream-instance-segmenter (Python).

Every case runs one model family over one input path and checks the three outputs the
application advertises: annotated frames, Insight segmentation metadata, and Insight video.
"""

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.output_assertions import assert_streamed_frames_are_usable

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
EXAMPLE = "single-stream-instance-segmenter"

# The YOLOv8 package tests/test-scope.yaml downloads. The YOLO26 packages come from the
# scope's selection through e2e_model_path, so the nightly run also exercises the variants.
YOLOV8_MODEL_FILE = "yolo_v8n_seg_mpk.tar.gz"

# One case per source/codec combination the application accepts. The
# validator only allows http with mjpeg, so that is the single http case.
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
            # HTTPS MJPEG sources commonly carry a self-signed certificate.
            "ssl_strict": False,
        },
        id="http-mjpeg",
    ),
]


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _run_and_check(
    family,
    model_path,
    source,
    request,
    tmp_output_dir,
    test_timeout_ms,
    e2e_config_section,
    e2e_config_writer,
):
    source_url = request.getfixturevalue(source["url_fixture"])
    frames = int(e2e_config_section(EXAMPLE, "testing.e2e.inference")["frames"])
    save_every = int(e2e_config_section(EXAMPLE, "testing.e2e.output")["save_every"])
    attempts = frames // save_every
    # This test is the Insight receiver, so it publishes to its own loopback address.
    insight_host = "127.0.0.1"
    video_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    metadata_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
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
            "model": {"family": family, "path": str(model_path)},
            "source": source_config,
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
        f"{source['name']} main.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert f"model={family}" in result.stdout
    assert f"processed={frames}" in result.stdout

    # Every result due a picture is accounted for: written, or reported as having lost the
    # decoded frame it needed. A source faster than the model legitimately produces some of
    # the latter on the host-decoded route, so the run is checked for complete accounting
    # rather than a fixed yield the platform cannot promise.
    accounting = re.search(r"saved=(\d+) unpaired=(\d+)", result.stdout)
    assert accounting, f"run summary missing save accounting\n{result.stdout}"
    saved, unpaired = int(accounting.group(1)), int(accounting.group(2))
    assert saved + unpaired == attempts, f"saved={saved} unpaired={unpaired} of {attempts}"
    assert saved > 0, "no annotated frame was written"

    output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
    assert len(output_files) == saved
    assert_streamed_frames_are_usable(tmp_output_dir, saved)

    # VideoSender always re-encodes to RTP H.264 (payload type 96) for Insight.
    assert len(video_packet) > 12 and video_packet[0] >> 6 == 2
    assert video_packet[1] & 0x7F == 96

    assert metadata.success, metadata.error
    for message in metadata.messages:
        assert message.frame_id.isdecimal() and message.timestamp_ms >= 0
    # Every frame publishes a message, including frames with nothing to report, so the
    # run is checked for a message that carries segments rather than whichever arrived last.
    published = [json.loads(message.payload)["data"]["segments"] for message in metadata.messages]
    segments = next((entry for entry in published if entry), None)
    assert segments, f"no published message carried a segment (messages={len(published)})"
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


@pytest.mark.e2e
@pytest.mark.parametrize("source", SOURCE_CASES)
class TestE2E:
    def test_yolo26_publishes_frames_video_and_metadata(
        self,
        source,
        request,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        e2e_config_section,
        e2e_config_writer,
    ):
        _run_and_check(
            "yolo26",
            e2e_model_path,
            source,
            request,
            tmp_output_dir,
            test_timeout_ms,
            e2e_config_section,
            e2e_config_writer,
        )

    def test_yolov8_publishes_frames_video_and_metadata(
        self,
        source,
        request,
        models_dir,
        tmp_output_dir,
        test_timeout_ms,
        e2e_config_section,
        e2e_config_writer,
        skip_unless_e2e_ready,
    ):
        model_path = models_dir / YOLOV8_MODEL_FILE
        skip_unless_e2e_ready(model_path.is_file(), f"missing yolov8 package: {model_path}")
        _run_and_check(
            "yolov8",
            model_path,
            source,
            request,
            tmp_output_dir,
            test_timeout_ms,
            e2e_config_section,
            e2e_config_writer,
        )
