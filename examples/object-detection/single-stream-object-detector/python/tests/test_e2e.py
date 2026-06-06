"""E2E tests for single-stream-object-detector (Python)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    for f in models_dir.iterdir():
        if pattern in f.name and "seg" not in f.name and f.name.endswith(".tar.gz"):
            return f
    return None


def _env_int_or_default(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        models_dir,
        rtsp_url,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_section,
        e2e_config_writer,
    ):
        model = _find_model(models_dir, "yolo_v8s")
        skip_unless_e2e_ready(model is not None, "yolo (non-seg) model not found in models_dir")

        inference = e2e_config_section("single-stream-object-detector", "inference")
        config_path = e2e_config_writer(
            {
                "source": {"rtsp_url": rtsp_url, "latency_ms": 200, "tcp": True},
                "model": {"path": str(model)},
                "inference": {"frames": 10, **inference},
                "runtime": {"profile": False, "profile_interval": 100},
                "output": {
                    "save_dir": str(tmp_output_dir),
                    "insight": {
                        "host": "127.0.0.1",
                        "video_port": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000
                        ),
                        "metadata_port": _env_int_or_default(
                            "SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100
                        ),
                    }
                },
            }
        )
        cmd = [
            sys.executable, str(MAIN_PY),
            "--config", str(config_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        # Insight video is published through VideoSender over UDP, so verify the process exits
        # cleanly when the receiver path is configured.
        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        output_files = [path for path in tmp_output_dir.iterdir() if path.is_file()]
        assert output_files, "Expected saved output frames but output directory is empty"
        assert all(path.stat().st_size > 0 for path in output_files)
