"""E2E tests for usb-camera-object-detector (Python).

There is no USB camera on the test target, so these drive the same graph from
the fixed-image NV12 source declared under `testing.e2e` in config.yaml. That
covers the NV12 branch, video sender, model, box decode, and metadata send.
USB capture and the encoded Input/JpegParse/SimaDecode route require camera
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
    """Collect H.264 RTP while the application runs; fail on loss or bad packets."""
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 19200))))
        self.socket.settimeout(0.1)
        self.port = self.socket.getsockname()[1]
        self.path = path
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.receive)

    def receive(self):
        sequence = None
        fragment = None
        try:
            with self.path.open("wb") as output:
                while not self.stop.is_set():
                    try:
                        packet = self.socket.recv(65536)
                    except socket.timeout:
                        continue
                    assert len(packet) > 12 and packet[0] == 0x80
                    assert packet[1] & 0x7f == 96
                    current = int.from_bytes(packet[2:4], "big")
                    if sequence is not None:
                        assert current == (sequence + 1) % 65536, "lost RTP packet"
                    sequence = current
                    payload = packet[12:]
                    kind = payload[0] & 31
                    if 1 <= kind <= 23:
                        assert fragment is None
                        output.write(b"\x00\x00\x00\x01" + payload)
                    elif kind == 28:
                        assert len(payload) > 2
                        if payload[1] & 0x80:
                            assert fragment is None
                            fragment = bytes([(payload[0] & 0xe0) | (payload[1] & 31)])
                        assert fragment is not None
                        fragment += payload[2:]
                        if payload[1] & 0x40:
                            output.write(b"\x00\x00\x00\x01" + fragment)
                            fragment = None
                    else:
                        raise AssertionError(f"unexpected H.264 RTP payload: {kind}")
        except BaseException as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.socket.close()

    def assert_decodable(self):
        assert self.error is None, str(self.error)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(self.path)],
            capture_output=True, text=True, timeout=20)
        assert probe.returncode == 0 and not probe.stderr, probe.stderr
        stream = json.loads(probe.stdout)["streams"][0]
        assert (stream["width"], stream["height"]) == (1920, 1080)
        assert int(stream["nb_read_frames"]) >= 3


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

    def test_full_pipeline(
        self,
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

        metadata_port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 19300))
        with MetadataJsonListener("127.0.0.1", metadata_port, num_ports=1,
                                  metadata_type="object-detection", data_array_key="objects",
                                  min_object_count=1) as listener, VideoReceiver(
                                      tmp_output_dir / "received.h264") as video:
            config_path = e2e_config_writer({
                "inference": {"frames": FRAMES},
                "output": {"insight": {"host": "127.0.0.1", "video_port": video.port,
                                        "metadata_port": metadata_port}},
            })
            result = self.run_example(config_path, test_timeout_ms)
            assert result.returncode == 0, result.stdout + result.stderr
            received = listener.wait_for_messages(5)
            following = listener.wait_for_messages(5)
        video.assert_decodable()
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
