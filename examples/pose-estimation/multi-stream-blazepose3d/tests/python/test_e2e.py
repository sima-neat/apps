"""Hardware E2E tests for the multi-stream BlazePose application."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DETECTOR_MODEL = "yolo26m-det-int8-b1.tar.gz"
POSE_MODEL = "blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz"
INSIGHT_HOST = "127.0.0.1"
MAX_STREAMS = 4


class RtpVideoListener:
    def __init__(self, host: str, base_port: int, num_ports: int, codec: str):
        self._stop = threading.Event()
        self._codec = codec
        self._counts = [0] * num_ports
        self._sockets = []
        self._threads = []
        for offset in range(num_ports):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            sock.bind((host, base_port + offset))
            self._sockets.append(sock)

    def _is_video_rtp(self, packet: bytes) -> bool:
        if len(packet) < 13 or packet[0] >> 6 != 2:
            return False
        header_size = 12 + 4 * (packet[0] & 0x0F)
        if header_size >= len(packet):
            return False
        if packet[0] & 0x10:
            if header_size + 4 > len(packet):
                return False
            extension_words = int.from_bytes(packet[header_size + 2 : header_size + 4])
            header_size += 4 + 4 * extension_words
            if header_size >= len(packet):
                return False
        if self._codec == "h264":
            return not packet[header_size] & 0x80 and 1 <= packet[header_size] & 0x1F <= 29
        if header_size + 2 > len(packet):
            return False
        nal_type = packet[header_size] >> 1 & 0x3F
        return (
            not packet[header_size] & 0x80
            and nal_type <= 49
            and packet[header_size + 1] & 0x07 != 0
        )

    def __enter__(self):
        for index, sock in enumerate(self._sockets):
            thread = threading.Thread(target=self._receive, args=(index, sock))
            thread.start()
            self._threads.append(thread)
        return self

    def __exit__(self, *_args):
        self._stop.set()
        for thread in self._threads:
            thread.join()
        for sock in self._sockets:
            sock.close()

    def _receive(self, index: int, sock: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                packet = sock.recv(65536)
            except TimeoutError:
                continue
            if self._is_video_rtp(packet):
                self._counts[index] += 1

    @property
    def received_all_ports(self) -> bool:
        return all(count > 0 for count in self._counts)


def test_h264_rtp_packet_validation():
    h264 = RtpVideoListener(INSIGHT_HOST, 0, 0, "h264")
    hevc = RtpVideoListener(INSIGHT_HOST, 0, 0, "h265")
    h264_packet = bytes([0x80, 96, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0x65])
    hevc_packet = bytes([0x80, 96, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 38, 1])
    assert h264._is_video_rtp(h264_packet)
    assert not hevc._is_video_rtp(h264_packet)
    assert hevc._is_video_rtp(hevc_packet)
    assert not h264._is_video_rtp(b"not-rtp")


def runtime_dependencies_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("cv2", "numpy", "pyneat")
    )


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@pytest.mark.e2e
class TestE2E:
    @pytest.mark.parametrize(
        ("codec", "urls_fixture"),
        [("h264", "rtsp_h264_urls"), ("h265", "rtsp_h265_urls")],
    )
    def test_multistream_pose_metadata(
        self,
        request,
        codec,
        urls_fixture,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        # The application accepts at most four streams; CI provides five per codec.
        urls = request.getfixturevalue(urls_fixture)[:MAX_STREAMS]
        detector_model = Path(
            os.environ.get(
                "SIMANEAT_APPS_TEST_DETECTOR_MODEL", models_dir / DETECTOR_MODEL
            )
        )
        pose_model = Path(
            os.environ.get(
                "SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL", models_dir / POSE_MODEL
            )
        )
        skip_unless_e2e_ready(
            runtime_dependencies_ready(), "pyneat runtime dependencies unavailable"
        )
        skip_unless_e2e_ready(
            detector_model.is_file(), f"missing YOLO26 package: {detector_model}"
        )
        skip_unless_e2e_ready(
            pose_model.is_file(), f"missing BlazePose package: {pose_model}"
        )
        metadata_port_base = env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
        video_port_base = env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
        streams = [
            {
                "id": f"camera{index}",
                "url": url,
                "codec": codec,
                "insight_channel": index,
            }
            for index, url in enumerate(urls)
        ]
        config = e2e_config_writer(
            {
                "models": {
                    "detector_path": str(detector_model),
                    "pose_path": str(pose_model),
                },
                "streams": streams,
                "pose": {
                    "max_people_per_frame": 2,
                    "presence_threshold": 0.0,
                    "job_timeout_ms": 10000,
                },
                "runtime": {"frames": 8},
                "output": {
                    "insight": {
                        "host": INSIGHT_HOST,
                        "video_port_base": video_port_base,
                        "metadata_port_base": metadata_port_base,
                    },
                    "video_enabled": True,
                },
            }
        )

        with (
            MetadataJsonListener(
                INSIGHT_HOST,
                metadata_port_base,
                num_ports=len(urls),
                require_all_ports=True,
                metadata_contracts={
                    "pose-estimation": "poses",
                    "auxiliary-visualization": "payload.poses",
                },
                min_object_count=1,
            ) as listener,
            RtpVideoListener(INSIGHT_HOST, video_port_base, len(urls), codec) as video,
        ):
            process = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config)],
                cwd=str(EXAMPLE_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=test_timeout_ms / 1000,
            )
            metadata = listener.wait_for_messages(10.0)

        assert process.returncode == 0, (
            f"main.py exited with {process.returncode}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        assert metadata.success, metadata.error
        assert video.received_all_ports, (
            f"valid {codec} RTP was not emitted on every configured video port"
        )
        pose_frames = set()
        world_pose_frames = set()
        for message in metadata.messages:
            parsed = json.loads(message.payload)
            frame = (message.port, message.timestamp_ms, message.frame_id)
            assert parsed["data"]["stream_id"] == (
                f"camera{message.port - metadata_port_base}"
            )
            if message.metadata_type == "pose-estimation":
                poses = parsed["data"]["poses"]
                assert all(len(pose.get("keypoints", [])) == 33 for pose in poses)
                assert all(len(pose.get("world_keypoints", [])) == 33 for pose in poses)
                assert all(
                    isinstance(pose.get("presence"), (int, float))
                    and 0.0 <= pose["presence"] <= 1.0
                    for pose in poses
                )
                assert all(
                    set(point) >= {"name", "x", "y", "z", "confidence"}
                    for pose in poses
                    for point in pose["world_keypoints"]
                )
                if poses:
                    pose_frames.add(frame)
                continue

            data = parsed["data"]
            assert data["schema_version"] == 1
            assert data["id"] == "world-pose"
            assert data["renderer"] == "blazepose-3d"
            poses = data["payload"]["poses"]
            for pose in poses:
                assert len(pose.get("keypoints", [])) == 33
                assert isinstance(pose.get("presence"), (int, float))
                assert 0.0 <= pose["presence"] <= 1.0
                assert all(
                    set(point) >= {"name", "x", "y", "z", "confidence"}
                    for point in pose["keypoints"]
                )
            if poses:
                world_pose_frames.add(frame)

        assert pose_frames, "no 2D BlazePose result was published"
        assert world_pose_frames, "no 3D BlazePose result was published"
        assert all(
            any(frame[0] == port and frame in world_pose_frames for frame in pose_frames)
            for port in metadata.ports_with_valid_json
        ), "2D and 3D metadata did not share a frame identity on every port"
