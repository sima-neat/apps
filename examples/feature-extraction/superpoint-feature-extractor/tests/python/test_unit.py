"""CLI tests for the Python SuperPoint example."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_remap_points_scales_model_coordinates_to_source_frame():
    example = load_example()

    actual = example.remap_points(
        np.asarray([[320.0, 240.0]], dtype=np.float32), 1280, 720, np
    )

    np.testing.assert_allclose(actual, [[640.0, 360.0]])


def test_model_options_delegate_image_geometry_to_core_preproc():
    class FakePyneat:
        AutoFlag = SimpleNamespace(On="on")
        InputKind = SimpleNamespace(Image="image")
        ResizeMode = SimpleNamespace(Stretch="stretch")
        PreprocessColorFormat = SimpleNamespace(BGR="bgr", GRAY8="gray8")
        BoxDecodeType = SimpleNamespace(SuperPoint="superpoint")
        SuperPointProfile = SimpleNamespace(A65V1="a65-v1")
        SuperPointOutputFormat = SimpleNamespace(FeaturePointsV1="feature-points-v1")
        TensorDType = SimpleNamespace(Float32="fp32")

        @staticmethod
        def ModelOptions():
            return SimpleNamespace(
                preprocess=SimpleNamespace(
                    kind=None,
                    enable=None,
                    input_max_width=0,
                    input_max_height=0,
                    input_max_depth=0,
                    resize=SimpleNamespace(enable=None, mode=None),
                    color_convert=SimpleNamespace(
                        enable=None, input_format=None, output_format=None
                    ),
                    normalize=SimpleNamespace(
                        enable=None,
                        mean=None,
                        stddev=None,
                        has_explicit_stats=False,
                    ),
                ),
                superpoint=SimpleNamespace(
                    profile=None, output_format=None, descriptor_output_dtype=None
                ),
                processcvu=SimpleNamespace(post_run_target=None),
            )

    example = load_example()
    options = example.model_options(FakePyneat, 1280, 720)

    assert options.preprocess.kind == "image"
    assert options.preprocess.enable == "on"
    assert (
        options.preprocess.input_max_width,
        options.preprocess.input_max_height,
    ) == (
        1280,
        720,
    )
    assert options.preprocess.resize.mode == "stretch"
    assert options.preprocess.color_convert.input_format == "bgr"
    assert options.preprocess.color_convert.output_format == "gray8"
    assert options.preprocess.normalize.mean == (0.0, 0.0, 0.0)
    assert options.preprocess.normalize.stddev == (1.0, 1.0, 1.0)
    assert options.preprocess.normalize.has_explicit_stats is True
    assert not hasattr(options, "boxdecode_original_width")
    assert not hasattr(options, "boxdecode_original_height")
    assert not hasattr(options, "boxdecode_resize_mode")


def test_validate_frame_accepts_any_stable_bgr_resolution():
    example = load_example()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    example.validate_frame(frame, np)
    example.validate_frame(frame, np, frame.shape)

    with pytest.raises(RuntimeError, match="resolution changed"):
        example.validate_frame(np.zeros((480, 640, 3), dtype=np.uint8), np, frame.shape)


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
