"""E2E tests for multi-stream-multi-model (Python)."""

from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"

# One slot per stream: the task, the head layout of its package, the package file the scope
# downloads, and the Insight contract that stream must publish on.
STREAM_SLOTS = [
    ("detection", "yolov8", "yolo_11s_mpk.tar.gz", "object-detection", "objects"),
    ("segmentation", "yolov8", "yolo_11s_seg_mpk.tar.gz", "segmentation", "segments"),
    ("pose", "yolo26", "yolo26m-pose-int8-b1.tar.gz", "pose-estimation", "poses"),
    ("detection", "yolo26", "yolo26m-det-int8-b1.tar.gz", "object-detection", "objects"),
]


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    @pytest.mark.parametrize(
        ("codec", "urls_fixture"),
        [("h264", "rtsp_h264_urls"), ("h265", "rtsp_h265_urls")],
    )
    def test_every_stream_publishes_its_own_model_contract(
        self,
        request,
        codec,
        urls_fixture,
        models_dir,
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
            len(rtsp_urls) >= 2, f"need at least two RTSP {codec.upper()} URLs for multistream e2e"
        )

        model_paths = [models_dir / slot[2] for slot in STREAM_SLOTS]
        missing = [str(path) for path in model_paths if not path.is_file()]
        skip_unless_e2e_ready(
            not missing, f"configured models not found under {models_dir}: {', '.join(missing)}"
        )

        output_cfg = e2e_config_section("multi-stream-multi-model", "testing.e2e.output")
        total_saved_frames = int(output_cfg["total_saved_frames"])
        metadata_port_base = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)

        streams = [
            {
                "url": rtsp_urls[index % len(rtsp_urls)],
                "task": task,
                "decode": decode,
                "model": str(model_paths[index]),
            }
            for index, (task, decode, _, _, _) in enumerate(STREAM_SLOTS)
        ]

        config_path = e2e_config_writer(
            {
                "streams": streams,
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

        cmd = [sys.executable, str(MAIN_PY), "--config", str(config_path)]

        # One listener per stream: the four channels carry three different Insight contracts, so a
        # single shared listener could not tell a wrong-typed stream from a silent one.
        with contextlib.ExitStack() as stack:
            listeners = [
                (
                    index,
                    metadata_type,
                    stack.enter_context(
                        MetadataJsonListener(
                            E2E_INSIGHT_HOST,
                            metadata_port_base + index,
                            num_ports=1,
                            metadata_type=metadata_type,
                            data_array_key=data_array_key,
                            require_all_ports=True,
                        )
                    ),
                )
                for index, (_, _, _, metadata_type, data_array_key) in enumerate(STREAM_SLOTS)
            ]
            result = run_until_output_files(
                cmd,
                tmp_output_dir,
                total_saved_frames,
                test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            received = [
                (index, metadata_type, listener.wait_for_messages(5.0))
                for index, metadata_type, listener in listeners
            ]

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        for index, metadata_type, metadata in received:
            assert metadata.success, (
                f"stream {index} did not publish {metadata_type} metadata on port "
                f"{metadata_port_base + index}: {metadata.error}"
            )

        saved = [
            path
            for path in tmp_output_dir.rglob("*")
            if path.is_file() and path.name != "config.yaml"
        ]
        assert len(saved) >= total_saved_frames, (
            f"Expected at least {total_saved_frames} sampled output files, got {len(saved)}"
        )
        assert all(path.stat().st_size > 0 for path in saved)
        for index in range(len(STREAM_SLOTS)):
            assert any(path.name.startswith(f"stream_{index}_frame_") for path in saved), (
                f"stream {index} saved no annotated debug frame"
            )
