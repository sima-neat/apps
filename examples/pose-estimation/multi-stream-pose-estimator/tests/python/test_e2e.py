"""E2E tests for multi-stream-pose-estimator (Python)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"


def _runtime_deps_ready() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("cv2", "numpy", "pyneat")
    )


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    @pytest.mark.parametrize(
        ("codec", "urls_fixture"),
        [("h264", "rtsp_h264_urls"), ("h265", "rtsp_h265_urls")],
    )
    def test_multi_stream_insight_and_save_pipeline(
        self,
        request,
        codec,
        urls_fixture,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        e2e_config_section,
        run_until_output_files,
    ):
        rtsp_urls = request.getfixturevalue(urls_fixture)
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        skip_unless_e2e_ready(
            len(rtsp_urls) >= 2,
            f"need at least two RTSP {codec.upper()} URLs for multistream e2e",
        )
        output_cfg = e2e_config_section(
            "multi-stream-pose-estimator", "testing.e2e.output"
        )
        total_saved_frames = int(output_cfg["total_saved_frames"])
        metadata_port_base = _env_int_or_default(
            "SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100
        )

        config_path = e2e_config_writer(
            {
                "streams": rtsp_urls[:2],
                "input": {"codec": codec},
                "output": {
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port_base": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port_base": metadata_port_base,
                    },
                    "debug_dir": str(tmp_output_dir),
                },
                "inference": {
                    "frames": 140,
                },
            }
        )

        cmd = [
            sys.executable,
            str(MAIN_PY),
            "--config",
            str(config_path),
        ]
        with MetadataJsonListener(
            E2E_INSIGHT_HOST,
            metadata_port_base,
            num_ports=2,
            metadata_type="pose-estimation",
            data_array_key="poses",
            require_all_ports=True,
        ) as metadata_listener:
            result = run_until_output_files(
                cmd,
                tmp_output_dir,
                total_saved_frames,
                test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            metadata = metadata_listener.wait_for_messages(5.0)

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert metadata.success, (
            "pose-estimation metadata was not received on all streams: "
            f"{metadata.error}"
        )
        for message in metadata.messages:
            poses = json.loads(message.payload)["data"]["poses"]
            assert all(len(pose.get("keypoints", [])) == 17 for pose in poses)

        files = [
            path
            for path in tmp_output_dir.rglob("*")
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(files) >= total_saved_frames, (
            f"Expected at least {total_saved_frames} sampled output files, got {len(files)}"
        )
        assert all(path.stat().st_size > 0 for path in files)
