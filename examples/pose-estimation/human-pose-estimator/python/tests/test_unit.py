"""Unit tests for human-pose-estimator (Python)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Point directly to main.py in the workspace
MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the pose estimation overlay pipeline."""

    def test_help_runs(self):
        """Help should exit cleanly and mention config usage."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_missing_config_file_fails_cleanly(self):
        """A nonexistent config should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "does-not-exist.yaml"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "config file not found" in r.stderr.lower()

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_help_runs_without_sidecar_config_module(self):
        """The packaged entrypoint should still show help with utils/config.py."""
        source = MAIN_PY.read_text(encoding="utf-8")
        config_source = (
            MAIN_PY.resolve().parent / "utils" / "config.py"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_main = Path(temp_dir) / "main.py"
            temp_utils = Path(temp_dir) / "utils"
            temp_utils.mkdir()
            (temp_utils / "config.py").write_text(config_source, encoding="utf-8")
            temp_main.write_text(source, encoding="utf-8")
            r = subprocess.run(
                [sys.executable, str(temp_main), "--help"],
                capture_output=True, text=True, timeout=10,
            )
        assert r.returncode == 0
        assert "--config" in r.stdout
