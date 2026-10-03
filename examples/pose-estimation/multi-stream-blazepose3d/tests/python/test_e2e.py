"""Hardware E2E tests for the multi-stream BlazePose application."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DETECTOR_MODEL = "yolo26m-det-int8-b1.tar.gz"
POSE_MODEL = "blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz"
INSIGHT_HOST = "127.0.0.1"


def run_case(command, urls, codec, models_dir, timeout_ms=120000):
    detector_model = Path(
        os.environ.get("SIMANEAT_APPS_TEST_DETECTOR_MODEL", models_dir / DETECTOR_MODEL)
    )
    pose_model = Path(
        os.environ.get("SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL", models_dir / POSE_MODEL)
    )
    assert detector_model.is_file() and pose_model.is_file(), (
        "YOLO26 and BlazePose packages required"
    )
    assert len(urls) == 4, "four RTSP streams are required for this E2E"
    metadata_port_base = int(
        os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", "9100")
    )
    video_port_base = int(
        os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000")
    )
    streams = [
        {
            "id": f"camera{index}",
            "url": url,
            "codec": codec,
            "insight_channel": index,
        }
        for index, url in enumerate(urls)
    ]
    settings = {
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
        },
    }

    with tempfile.TemporaryDirectory(prefix="blazepose-e2e-") as directory:
        config = Path(directory) / "config.yaml"
        config.write_text(json.dumps(settings))  # JSON is also valid YAML.
        with ExitStack() as stack:
            video = [
                stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                for _ in urls
            ]
            for offset, sock in enumerate(video):
                sock.bind((INSIGHT_HOST, video_port_base + offset))
                sock.settimeout(5.0)
            listener = stack.enter_context(
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
                )
            )
            process = subprocess.run(
                [*command, "--config", str(config)],
                cwd=str(EXAMPLE_DIR),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_ms / 1000,
            )
            metadata = listener.wait_for_messages(10.0)
            video_packets = [sock.recv(65536) for sock in video]

        assert process.returncode == 0, (
            f"application exited with {process.returncode}\n"
            f"stdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        assert metadata.success, metadata.error
        payload_type = 98 if codec == "h265" else 96
        assert all(
            len(packet) > 12
            and packet[0] >> 6 == 2
            and packet[1] & 0x7F == payload_type
            for packet in video_packets
        ), f"{codec} RTP was not emitted on every configured video port"
        frames = {"pose-estimation": set(), "auxiliary-visualization": set()}
        for message in metadata.messages:
            data = json.loads(message.payload)["data"]
            assert data["stream_id"] == f"camera{message.port - metadata_port_base}"
            if message.metadata_type == "pose-estimation":
                poses = data["poses"]
                assert all(len(pose["keypoints"]) == 33 for pose in poses)
                world_key = "world_keypoints"
            else:
                assert (data["schema_version"], data["id"], data["renderer"]) == (
                    1,
                    "world-pose",
                    "blazepose-3d",
                )
                poses = data["payload"]["poses"]
                world_key = "keypoints"
            for pose in poses:
                assert 0 <= pose["presence"] <= 1
                assert len(pose[world_key]) == 33
                assert all(
                    set(point) >= {"name", "x", "y", "z", "confidence"}
                    for point in pose[world_key]
                )
            if poses:
                frames[message.metadata_type].add(
                    (message.port, message.timestamp_ms, message.frame_id)
                )
        paired = frames["pose-estimation"] & frames["auxiliary-visualization"]
        assert {frame[0] for frame in paired} == set(
            range(metadata_port_base, metadata_port_base + len(urls))
        )
        print(
            f"[OK] {codec}: video and paired 33-landmark poses on {len(urls)} streams"
        )


if __name__ == "__main__":
    models_dir = Path(os.environ.get("SIMANEAT_APPS_TEST_MODELS_DIR", "models"))
    cases = [
        (codec, os.environ.get(f"SIMANEAT_TEST_RTSP_{codec.upper()}_URLS", ""))
        for codec in ("h264", "h265")
    ]
    assert any(urls for _, urls in cases), "RTSP URLs are required"
    for codec, urls in cases:
        if urls:
            run_case(
                [sys.argv[1]],
                urls.split(","),
                codec,
                models_dir,
                int(os.environ.get("SIMANEAT_APPS_TEST_TIMEOUT_MS", "120000")),
            )
else:
    import pytest

    @pytest.mark.e2e
    @pytest.mark.parametrize(
        "codec,urls_fixture", [("h264", "rtsp_h264_urls"), ("h265", "rtsp_h265_urls")]
    )
    def test_multistream_pose_metadata(
        request, codec, urls_fixture, models_dir, test_timeout_ms
    ):
        run_case(
            [sys.executable, str(MAIN_PY)],
            request.getfixturevalue(urls_fixture),
            codec,
            models_dir,
            test_timeout_ms,
        )
