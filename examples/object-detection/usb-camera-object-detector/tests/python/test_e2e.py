"""E2E tests for usb-camera-object-detector (Python).

There is no USB camera on the test target, so these drive the same graph from
the fixed-image MJPEG source declared under `testing.e2e` in config.yaml. That
covers the MJPEG passthrough video sender, SimaDecode for 4:2:2 and
4:2:0 MJPEG straight into the model, box decode, and metadata send. USB capture requires camera
validation separately.
"""

import re
import json
import os
import socket
import threading

from tests.utils.metadata_json_listener import MetadataJsonListener
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CONFIG_YAML = EXAMPLE_DIR / "src" / "common" / "config.yaml"
FRAMES = 30


def shipped_test_override() -> str:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    return raw["testing"]["e2e"]["source"]["override_fragment"]


class VideoReceiver:
    """Collect complete RTP/JPEG frames while the application runs; fail on loss or bad packets."""
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 19200))))
        self.socket.settimeout(0.1)
        self.port = self.socket.getsockname()[1]
        self.frame_types = []
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.receive)

    def receive(self):
        sequence = None
        assembled = 0  # JPEG bytes of the frame in progress
        try:
            while not self.stop.is_set():
                try:
                    packet = self.socket.recv(65536)
                except socket.timeout:
                    continue
                assert len(packet) > 20 and packet[0] == 0x80
                assert packet[1] & 0x7f == 26, "not an RTP/JPEG packet"
                current = int.from_bytes(packet[2:4], "big")
                if sequence is not None:
                    assert current == (sequence + 1) % 65536, "lost RTP packet"
                sequence = current
                jpeg = packet[12:]  # RFC 2435 main header
                offset, jpeg_type, q = int.from_bytes(jpeg[1:4], "big"), jpeg[4], jpeg[5]
                assert offset == assembled, "incomplete RTP/JPEG frame"
                assert (jpeg[6] * 8, jpeg[7] * 8) == (1920, 1080), "RTP/JPEG frame is not 1920x1080"
                header = 8 + (4 if jpeg_type >= 64 else 0)
                if offset == 0 and q >= 128:
                    header += 4 + int.from_bytes(jpeg[header + 2:header + 4], "big")
                assert len(jpeg) >= header, "truncated RTP/JPEG header"
                assembled += len(jpeg) - header
                if packet[1] & 0x80:  # marker: last packet of the frame
                    self.frame_types.append(jpeg_type)
                    assembled = 0
        except BaseException as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.socket.close()

    def assert_jpeg(self, expected_type):
        """At least 3 complete 1920x1080 frames of the source's RFC 2435 type arrived intact
        (0 is 4:2:2, 1 is 4:2:0; +64 with restart markers), so the camera JPEG was forwarded."""
        assert self.error is None, str(self.error)
        assert len(self.frame_types) >= 3, f"complete RTP/JPEG frames: {len(self.frame_types)}"
        assert {t % 64 for t in self.frame_types} == {expected_type}, f"types: {self.frame_types}"


@pytest.mark.e2e
class TestE2E:
    def run_example(self, config_path, test_timeout_ms) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR.parents[2]),
        )

    # Y42B encodes 4:2:2 MJPEG, the sampling most USB cameras send; I420 encodes 4:2:0.
    @pytest.mark.parametrize("raw_format", ["Y42B", "I420"], ids=["mjpeg422", "mjpeg420"])
    def test_full_pipeline(
        self,
        raw_format,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """The graph builds and publishes detections from the fixed-image source."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )
        fragment = shipped_test_override()
        assert "format=Y42B" in fragment
        fragment = fragment.replace("format=Y42B", f"format={raw_format}")

        metadata_port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 19300))
        with MetadataJsonListener("127.0.0.1", metadata_port, num_ports=1,
                                  metadata_type="object-detection", data_array_key="objects",
                                  min_object_count=1) as listener, VideoReceiver() as video:
            config_path = e2e_config_writer({
                "source": {"override_fragment": fragment,
                           "override_format": "NV16" if raw_format == "Y42B" else "NV12"},
                "inference": {"frames": FRAMES},
                "output": {"insight": {"host": "127.0.0.1", "video_port": video.port,
                                        "metadata_port": metadata_port}},
            })
            result = self.run_example(config_path, test_timeout_ms)
            assert result.returncode == 0, result.stdout + result.stderr
            received = listener.wait_for_messages(5)
            following = listener.wait_for_messages(5)
        video.assert_jpeg(0 if raw_format == "Y42B" else 1)
        assert received.success, received.error
        assert following.success, following.error
        assert following.messages[-1].timestamp_ms > received.messages[-1].timestamp_ms
        for message in received.messages + following.messages:
            assert message.timestamp_ms >= 0
            objects = json.loads(message.payload)["data"]["objects"]
            assert objects
            for obj in objects:
                assert obj["label"] and 0.3 <= obj["confidence"] <= 1
                x, y, w, h = obj["bbox"]
                assert 0 <= x < x + w <= 1920
                assert 0 <= y < y + h <= 1080
        match = re.search(r"processed=(\d+) detections=(\d+)", result.stdout)
        assert match and int(match.group(1)) == FRAMES and int(match.group(2)) > 0

    def test_startup_banner_reports_the_resolved_endpoints(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """The banner is the operator's only confirmation of where video went."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "inference": {"frames": 5},
                "output": {
                    "insight": {"host": "127.0.0.1", "video_port": 9200, "metadata_port": 9300}
                },
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode == 0, result.stderr
        assert "source=override" in result.stdout
        assert "stream=1920x1080@30" in result.stdout
        assert "insight=127.0.0.1" in result.stdout
        assert "video=9200" in result.stdout

    def test_profile_output_is_emitted_when_enabled(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """runtime.profile must produce windowed timing, not just a summary."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "inference": {"frames": 10},
                "runtime": {"profile": True, "profile_interval": 5},
                "output": {"insight": {"host": "127.0.0.1"}},
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode == 0, result.stderr
        assert "[profile] frames=5" in result.stdout
        assert "Backend:" in result.stdout, "profile mode must dump the generated backend"

    def test_invalid_source_fragment_fails_cleanly(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """A broken fragment must exit nonzero with a message, not hang or crash."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "source": {"override_fragment": "definitely-not-a-gst-element ! queue"},
                "inference": {"frames": 5},
                "output": {"insight": {"host": "127.0.0.1"}},
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode != 0
        assert "Error:" in result.stderr
