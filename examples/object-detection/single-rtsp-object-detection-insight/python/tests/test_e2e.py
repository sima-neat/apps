"""E2E tests for single-rtsp-object-detection-insight (Python)."""

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


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(self, models_dir, rtsp_url, test_timeout_ms, skip_unless_e2e_ready):
        model = _find_model(models_dir, "yolo_v8s")
        skip_unless_e2e_ready(model is not None, "yolo (non-seg) model not found in models_dir")

        config_path = Path("/tmp") / "single_rtsp_python_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "source:",
                    f"  rtsp_url: {rtsp_url}",
                    "  latency_ms: 200",
                    "  udp: false",
                    "model:",
                    f"  path: {model}",
                    "runtime:",
                    "  frames: 10",
                    "  debug: false",
                    "insight:",
                    "  host: 127.0.0.1",
                    "  video_port: 9000",
                    "  metadata_port: 9100",
                    "",
                ]
            ),
            encoding="utf-8",
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
