"""E2E tests for yolo26-batch4-detector (Python)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"
# The model is compiled for batch 4, so the test needs all four lanes fed.
REQUIRED_STREAMS = 4


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    def test_batch4_insight_and_save_pipeline(
        self,
        e2e_model_path,
        tmp_output_dir,
        rtsp_h264_urls,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        e2e_config_section,
        run_until_output_files,
    ):
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        skip_unless_e2e_ready(
            len(rtsp_h264_urls) >= REQUIRED_STREAMS,
            "need four RTSP H.264 URLs to fill the batch-4 lanes",
        )
        output_cfg = e2e_config_section("yolo26-batch4-detector", "testing.e2e.output")
        save_every = int(output_cfg["save_every"])
        total_saved_frames = int(output_cfg["total_saved_frames"])
        metadata_port_base = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)

        config_path = e2e_config_writer(
            {
                "streams": rtsp_h264_urls[:REQUIRED_STREAMS],
                "output": {
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port_base": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port_base": metadata_port_base,
                    },
                    "debug_dir": str(tmp_output_dir),
                    "save_every": save_every,
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
            num_ports=REQUIRED_STREAMS,
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
            f"object-detection metadata was not received on all lanes: {metadata.error}"
        )
        assert all(
            isinstance(json.loads(message.payload).get("_insight", {}).get("rtp_timestamp"), int)
            for message in metadata.messages
        ), "metadata must carry the analysed frame's RTP timestamp"

        files = [
            path
            for path in tmp_output_dir.rglob("*")
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(files) >= total_saved_frames, (
            f"Expected at least {total_saved_frames} sampled output files, got {len(files)}"
        )
        assert all(path.stat().st_size > 0 for path in files)
