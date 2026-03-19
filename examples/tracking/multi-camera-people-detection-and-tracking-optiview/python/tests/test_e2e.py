"""E2E tests for multi-camera-people-detection-and-tracking-optiview (Python)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
import textwrap

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


def _runtime_deps_ready() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("cv2", "numpy", "pyneat"))


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    for model in models_dir.iterdir():
        if pattern in model.name and "seg" not in model.name and model.name.endswith(".tar.gz"):
            return model
    return None


@pytest.mark.e2e
class TestE2E:
    def test_multi_stream_optiview_and_save_pipeline(
        self, models_dir, tmp_output_dir, rtsp_urls, test_timeout_ms, skip_unless_e2e_ready
    ):
        skip_unless_e2e_ready(
            _runtime_deps_ready(),
            "python runtime dependencies (cv2, numpy, pyneat) are not available",
        )
        model = _find_model(models_dir, "yolo_v8")
        skip_unless_e2e_ready(model is not None, "yolo detector model not found in models_dir")
        skip_unless_e2e_ready(len(rtsp_urls) >= 2, "need at least two RTSP URLs for multistream e2e")

        config_path = tmp_output_dir / "config.yaml"
        config_path.write_text(
            textwrap.dedent(
                f"""
                model: {model}

                input:
                  tcp: true
                  latency_ms: 200

                inference:
                  frames: 10
                  fps: 0
                  bitrate_kbps: 2500
                  profile: false
                  person_class_id: 0
                  detection_threshold: null
                  nms_iou_threshold: null
                  top_k: null

                tracking:
                  iou_threshold: 0.3
                  max_missing_frames: 15

                output:
                  optiview:
                    host: 127.0.0.1
                    video_port_base: 9000
                    json_port_base: 9100
                  debug_dir: {tmp_output_dir}
                  save_every: 2

                streams:
                """
            ).lstrip(),
            encoding="utf-8",
        )
        with config_path.open("a", encoding="utf-8") as handle:
            for url in rtsp_urls[:2]:
                handle.write(f"  - {url}\n")

        cmd = [
            sys.executable,
            str(MAIN_PY),
            "--config",
            str(config_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        files = [path for path in tmp_output_dir.rglob("*") if path.is_file()]
        assert files, "Expected sampled output files but output directory is empty"
