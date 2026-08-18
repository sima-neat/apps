"""Focused unit tests for ssd-mobilenet-object-detector."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

SPEC = importlib.util.spec_from_file_location("ssd_mobilenet_main", MAIN_PY)
ssd_mobilenet_main = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ssd_mobilenet_main)


@pytest.mark.parametrize(
    ("profile", "mean", "stddev"),
    [
        ("tensorflow_ssd", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        (
            "torchvision_ssdlite",
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ],
)
def test_normalization_profiles(profile, mean, stddev):
    assert ssd_mobilenet_main.normalization_for_profile(profile) == (mean, stddev)


def test_unknown_normalization_profile_is_rejected():
    with pytest.raises(ValueError, match="preprocessing_profile"):
        ssd_mobilenet_main.normalization_for_profile("unknown")


@pytest.mark.unit
class TestArgParsing:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_bad_config_path(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/ssd-config.yaml"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
            check=False,
        )
        assert result.returncode == 2
        assert "failed to open config" in result.stderr

    @pytest.mark.parametrize(
        "contents", ["io:\n  input_dir: /tmp\n", "model:\n  path: '   '\n"]
    )
    def test_model_path_is_required(self, tmp_path, contents):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(contents, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
            check=False,
        )
        assert result.returncode == 2
        assert "model.path must be a nonempty path" in result.stderr

    def test_unknown_flag(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
            check=False,
        )
        assert result.returncode == 2
        assert "error" in result.stderr.lower()
