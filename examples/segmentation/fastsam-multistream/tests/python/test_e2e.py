"""E2E tests for fastsam-multistream (Python)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
E2E_INSIGHT_HOST = "127.0.0.1"

# model.path (FastSAM) is auto-wired from the config; the CLIP artifacts are not.
CLIP_IMAGE_ENCODER_FILE = "MobileCLIP2-S0_image_encoder_reparam_mpk.tar.gz"
CLIP_TEXT_ENCODER_FILE = "MobileCLIP2-S0_text_mpk.tar.gz"
CLIP_TEXT_CONSTS_FILE = "MobileCLIP2-S0_text_host_consts.npz"
FASTSAM_FILE = "FastSAM-x_quant_mpk.tar.gz"


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    def test_multi_stream_segmentation_metadata(
        self,
        tmp_output_dir,
        models_dir,
        rtsp_h264_urls,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        run_until_output_files,
    ):
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        skip_unless_e2e_ready(
            len(rtsp_h264_urls) >= 4, "need four RTSP H.264 URLs for multistream e2e"
        )

        clip_image = models_dir / CLIP_IMAGE_ENCODER_FILE
        clip_text = models_dir / CLIP_TEXT_ENCODER_FILE
        clip_consts = models_dir / CLIP_TEXT_CONSTS_FILE
        fastsam_pkg = models_dir / FASTSAM_FILE
        for artifact in (fastsam_pkg, clip_image, clip_text, clip_consts):
            skip_unless_e2e_ready(
                artifact.is_file(), f"required model artifact not found: {artifact}"
            )

        metadata_port_base = _env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
        streams = rtsp_h264_urls[:4]

        config_path = e2e_config_writer(
            {
                "source": {"rtsp_urls": streams},
                "clip": {
                    "image_encoder_path": str(clip_image),
                    "text_encoder_path": str(clip_text),
                    "text_host_consts": str(clip_consts),
                },
                "runtime": {"frames": 60},
                "output": {
                    "video_enabled": False,
                    "insight": {
                        "host": E2E_INSIGHT_HOST,
                        "metadata_port_base": metadata_port_base,
                    },
                },
            }
        )

        cmd = [sys.executable, str(MAIN_PY), str(config_path)]

        with MetadataJsonListener(
            E2E_INSIGHT_HOST,
            metadata_port_base,
            num_ports=len(streams),
            metadata_type="segmentation",
            data_array_key="segments",
            require_all_ports=True,
        ) as metadata_listener:
            result = run_until_output_files(
                cmd,
                tmp_output_dir,
                0,
                test_timeout_ms / 1000,
                cwd=str(EXAMPLE_DIR),
            )
            metadata = metadata_listener.wait_for_messages(10.0)

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert metadata.success, (
            f"segmentation metadata was not received on all streams: {metadata.error}"
        )
