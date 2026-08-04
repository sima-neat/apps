"""CLI tests for the Python SuperPoint example."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def load_example():
    spec = importlib.util.spec_from_file_location("superpoint_example", MAIN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draw_points_marks_the_requested_coordinate():
    class FakeCv2:
        LINE_AA = 16
        FONT_HERSHEY_SIMPLEX = 0

        def __init__(self):
            self.circles = []

        def circle(self, _frame, center, *_args):
            self.circles.append(center)

        @staticmethod
        def putText(*_args):
            pass

    example = load_example()
    cv2 = FakeCv2()

    example.draw_points(object(), [(40.0, 50.0)], cv2)

    assert cv2.circles == [(40, 50)]


@pytest.mark.unit
class TestCli:
    def run(self, *args: str):
        return subprocess.run(
            [sys.executable, str(MAIN_PY), *args],
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_help(self):
        result = self.run("--help")
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_unknown_argument_is_rejected(self):
        result = self.run("--bogus")
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr

    def test_missing_config_value_is_rejected(self):
        result = self.run("--config")
        assert result.returncode == 2
        assert "expected one argument" in result.stderr

    def test_missing_config_is_reported(self):
        result = self.run("--config", "/does/not/exist.yaml")
        assert result.returncode == 2
        assert "Error:" in result.stderr
