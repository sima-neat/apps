"""E2E tests for usb-camera-object-detector (Python).

There is no USB camera on the test target, so these drive the same graph from
the synthetic NV12 source declared under `testing.e2e` in config.yaml. That
covers everything except the v4l2src element itself: the custom source node, the
branch, the video sender, the model, the box decode, and the metadata send.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CONFIG_YAML = EXAMPLE_DIR / "src" / "common" / "config.yaml"
FRAMES = 30


def shipped_test_override() -> str:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    return raw["testing"]["e2e"]["source"]["override_fragment"]


@pytest.mark.e2e
class TestE2E:
    def run_example(self, config_path, test_timeout_ms) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR.parents[2]),
        )

    def test_full_pipeline(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """The graph builds and publishes detections from the synthetic source."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "inference": {"frames": FRAMES},
                "output": {"insight": {"host": "127.0.0.1"}},
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode == 0, (
            f"main.py exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        match = re.search(r"processed=(\d+) detections=(\d+)", result.stdout)
        assert match, f"run did not report its totals\nstdout:\n{result.stdout}"
        assert int(match.group(1)) == FRAMES, "every requested frame must be published"

    def test_startup_banner_reports_the_resolved_endpoints(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """The banner is the operator's only confirmation of where video went."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "inference": {"frames": 5},
                "output": {
                    "insight": {"host": "127.0.0.1", "video_port": 9200, "metadata_port": 9300}
                },
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode == 0, result.stderr
        assert "source=override" in result.stdout
        assert "stream=1920x1080@30" in result.stdout
        assert "insight=127.0.0.1" in result.stdout
        assert "video=9200" in result.stdout

    def test_profile_output_is_emitted_when_enabled(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """runtime.profile must produce windowed timing, not just a summary."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "inference": {"frames": 10},
                "runtime": {"profile": True, "profile_interval": 5},
                "output": {"insight": {"host": "127.0.0.1"}},
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode == 0, result.stderr
        assert "[profile] frames=5" in result.stdout
        assert "Backend:" in result.stdout, "profile mode must dump the generated backend"

    def test_invalid_source_fragment_fails_cleanly(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        """A broken fragment must exit nonzero with a message, not hang or crash."""
        skip_unless_e2e_ready(
            bool(shipped_test_override()),
            "config.yaml declares no testing.e2e source override",
        )

        config_path = e2e_config_writer(
            {
                "source": {"override_fragment": "definitely-not-a-gst-element ! queue"},
                "inference": {"frames": 5},
                "output": {"insight": {"host": "127.0.0.1"}},
            }
        )
        result = self.run_example(config_path, test_timeout_ms)

        assert result.returncode != 0
        assert "Error:" in result.stderr
