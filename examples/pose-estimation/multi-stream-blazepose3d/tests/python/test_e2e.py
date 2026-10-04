"""Hardware E2E tests for the multi-stream BlazePose application."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DETECTOR_MODEL = "yolo26m-det-int8-b1.tar.gz"
POSE_MODEL = "blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz"
INSIGHT_HOST = "127.0.0.1"
MAX_STREAMS = 4


class VideoListener:
    """Counts the UDP datagrams that arrive on each Insight video port."""

    def __init__(self, host: str, base_port: int, num_ports: int):
        self._stop = threading.Event()
        self._counts = [0] * num_ports
        self._sockets = []
        self._threads = []
        for offset in range(num_ports):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            sock.bind((host, base_port + offset))
            self._sockets.append(sock)

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
                sock.recv(65536)
            except TimeoutError:
                continue
            self._counts[index] += 1

    @property
    def received_all_ports(self) -> bool:
        return all(count > 0 for count in self._counts)


def runtime_dependencies_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("cv2", "numpy", "pyneat")
    )


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def metadata_problem(messages, metadata_port_base: int) -> str | None:
    """Every message carries its port's stream id and 33 world keypoints per pose,
    and at least one port publishes a non-empty 2D/3D pair for one frame."""
    pose_counts: dict[str, dict[tuple, int]] = {
        "pose-estimation": {},
        "auxiliary-visualization": {},
    }
    for message in messages:
        data = json.loads(message.payload)["data"]
        if data.get("stream_id") != f"camera{message.port - metadata_port_base}":
            return f"metadata on port {message.port} did not carry its stream id"
        if message.metadata_type == "pose-estimation":
            poses, points = data["poses"], "world_keypoints"
        elif (data.get("id"), data.get("renderer")) == ("world-pose", "blazepose-3d"):
            poses, points = data["payload"]["poses"], "keypoints"
        else:
            return "auxiliary metadata did not use the world-pose BlazePose 3D envelope"
        if not all(len(pose.get(points, [])) == 33 for pose in poses):
            return f"a {message.metadata_type} pose did not carry 33 world keypoints"
        frame = (message.port, message.timestamp_ms, message.frame_id)
        pose_counts[message.metadata_type][frame] = len(poses)
    world = pose_counts["auxiliary-visualization"]
    if not any(
        count > 0 and world.get(frame) == count
        for frame, count in pose_counts["pose-estimation"].items()
    ):
        return "no stream published a non-empty 2D/3D BlazePose pair for one frame"
    return None


def test_metadata_problem_requires_one_non_empty_pair_and_stream_ids():
    def message(port, metadata_type, frame_id, poses, stream_id=None):
        data = {"stream_id": stream_id or f"camera{port - 9100}"}
        if metadata_type == "pose-estimation":
            data["poses"] = [{"world_keypoints": [{}] * 33}] * poses
        else:
            data.update(id="world-pose", renderer="blazepose-3d")
            data["payload"] = {"poses": [{"keypoints": [{}] * 33}] * poses}
        return SimpleNamespace(
            port=port,
            metadata_type=metadata_type,
            timestamp_ms=1,
            frame_id=frame_id,
            payload=json.dumps({"data": data}),
        )

    pair = [
        message(9101, "pose-estimation", "1", 2),
        message(9101, "auxiliary-visualization", "1", 2),
    ]
    assert metadata_problem(pair, 9100) is None
    assert "non-empty" in metadata_problem(pair[:1], 9100)
    split = [pair[0], message(9101, "auxiliary-visualization", "2", 2)]
    assert "non-empty" in metadata_problem(split, 9100)
    assert "stream id" in metadata_problem([message(9100, "pose-estimation", "1", 1, "x")], 9100)


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
                "pose": {"max_people_per_frame": 2, "presence_threshold": 0.0},
                "runtime": {"frames": 30},
                "output": {
                    "insight": {
                        "host": INSIGHT_HOST,
                        "video_port_base": video_port_base,
                        "metadata_port_base": metadata_port_base,
                    },
                },
            }
        )

        with (
            MetadataJsonListener(
                INSIGHT_HOST,
                metadata_port_base,
                num_ports=len(urls),
                metadata_contracts={
                    "pose-estimation": "poses",
                    "auxiliary-visualization": "payload.poses",
                },
            ) as listener,
            VideoListener(INSIGHT_HOST, video_port_base, len(urls)) as video,
        ):
            process = subprocess.run(
                [sys.executable, str(MAIN_PY), "--config", str(config)],
                cwd=str(EXAMPLE_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=test_timeout_ms / 1000,
            )
            # The application has exited; drain the buffered metadata until one
            # port holds a non-empty 2D/3D pair or a message is invalid.
            messages = []
            problem = "no metadata was received"
            deadline = time.monotonic() + 10.0
            while problem is not None and time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                messages.extend(listener.wait_for_messages(min(1.0, remaining)).messages)
                problem = metadata_problem(messages, metadata_port_base)

        assert process.returncode == 0, (
            f"main.py exited with {process.returncode}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        assert video.received_all_ports, (
            f"{codec} video was not emitted on every configured video port"
        )
        assert problem is None, problem
