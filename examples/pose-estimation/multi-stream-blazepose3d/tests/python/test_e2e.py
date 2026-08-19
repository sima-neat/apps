"""Hardware E2E tests for the multi-stream BlazePose application."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DETECTOR_MODEL = "yolo26m-det-int8-b1.tar.gz"
POSE_MODEL = "blazepose_heavy_3d_bf16_nopad_neat_mpk.tar.gz"
INSIGHT_HOST = "127.0.0.1"


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
        urls = request.getfixturevalue(urls_fixture)
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
                "pose": {"max_people_per_frame": 2, "job_timeout_ms": 10000},
                "runtime": {"frames": 8},
                "output": {
                    "insight": {
                        "host": INSIGHT_HOST,
                        "video_port_base": env_int(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port_base": metadata_port_base,
                    },
                    "video_enabled": True,
                },
            }
        )

        with MetadataJsonListener(
            INSIGHT_HOST,
            metadata_port_base,
            num_ports=len(urls),
            metadata_type="pose-estimation",
            data_array_key="poses",
            require_all_ports=True,
        ) as listener:
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
            f"main.py exited with {process.returncode}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        assert metadata.success, metadata.error
        poses = [
            pose
            for message in metadata.messages
            for pose in json.loads(message.payload)["data"]["poses"]
        ]
        assert poses, "no BlazePose result was published"
        assert all(len(pose.get("keypoints", [])) == 33 for pose in poses)
