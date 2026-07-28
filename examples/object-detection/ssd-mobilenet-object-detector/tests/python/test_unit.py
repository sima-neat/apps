"""Unit tests for ssd-mobilenet-object-detector (Python)."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def _run(args, cwd=EXAMPLE_DIR):
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(cwd),
    )


def _write_config(tmp_path: Path, section: str, key: str, value: str) -> Path:
    config = tmp_path / f"{section}_{key}.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: /nonexistent/ssd_model.tar.gz
            io:
              input_dir: assets/datasets/coco
              output_dir: sandbox/ssd-mobilenet-object-detector
            {section}:
              {key}: {value}
            """
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the SSD example."""

    def test_help(self):
        r = _run(["--help"])
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        r = _run(["--config", "/nonexistent/ssd-config.yaml"])
        assert r.returncode == 2
        assert "failed to open config file" in r.stderr

    def test_unknown_flag(self):
        r = _run(["--bogus"])
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()


@pytest.mark.unit
class TestDecodeConfigValidation:
    """The decode section drives the model-managed BoxDecode stage, so it must be validated."""

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("score_threshold", "1.5", "decode.score_threshold must be in [0.0, 1.0]"),
            ("nms_iou", "-0.1", "decode.nms_iou must be in [0.0, 1.0]"),
            ("max_detections", "0", "decode.max_detections must be >= 1"),
        ],
    )
    def test_rejects_out_of_range_values(self, tmp_path, key, value, message):
        config = _write_config(tmp_path, "decode", key, value)
        r = _run(["--config", str(config)])
        assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert message in r.stderr


@pytest.mark.unit
class TestRuntimeConfigValidation:
    """Runtime limits must be rejected as config errors, not surface as load/profile failures."""

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("timeout_ms", "0", "runtime.timeout_ms must be > 0"),
            ("num_runs", "0", "runtime.num_runs must be >= 1"),
        ],
    )
    def test_rejects_out_of_range_values(self, tmp_path, key, value, message):
        config = _write_config(tmp_path, "runtime", key, value)
        r = _run(["--config", str(config)])
        assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert message in r.stderr
