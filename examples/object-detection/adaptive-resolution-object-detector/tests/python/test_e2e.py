"""E2E tests for adaptive-resolution-object-detector (Python)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener
from tests.utils.output_assertions import assert_streamed_frames_are_usable
from tests.utils.process_assertions import assert_exited_cleanly


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"
STREAMS = 2


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    # group runs with one stream per group, so the two streams are detected by
    # two independent processes publishing on their own channels.
    @pytest.mark.parametrize("mode", ["live", "scale", "group"])
    def test_every_mode_publishes_detections_for_every_stream(
        self,
        mode,
        rtsp_h264_urls,
        e2e_model_path,
        tmp_output_dir,
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
            len(rtsp_h264_urls) >= STREAMS, f"need at least {STREAMS} RTSP H.264 URLs"
        )
        output_cfg = e2e_config_section("adaptive-resolution-object-detector", "testing.e2e.output")
        total_saved_frames = int(output_cfg["total_saved_frames"])
        metadata_port = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)

        config_path = e2e_config_writer(
            {
                "mode": mode,
                "streams": {
                    "max_streams": STREAMS,
                    "sources": [
                        {"id": f"cam-{index + 1}", "url": url}
                        for index, url in enumerate(rtsp_h264_urls[:STREAMS])
                    ],
                },
                "group": {"size": 1},
                "inference": {"frames": 140},
                "output": {
                    "save_dir": str(tmp_output_dir),
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "video_port": _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000),
                        "metadata_port": metadata_port,
                    },
                },
            }
        )

        cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path)]
        with MetadataJsonListener(
            E2E_INSIGHT_HOST,
            metadata_port,
            num_ports=STREAMS,
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

        assert_exited_cleanly(result)
        assert metadata.success, (
            f"{mode}: object-detection metadata was not received on all streams: {metadata.error}"
        )
        assert_streamed_frames_are_usable(tmp_output_dir, total_saved_frames, streams=STREAMS)
