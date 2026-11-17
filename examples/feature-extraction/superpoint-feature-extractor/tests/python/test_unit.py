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


@pytest.mark.unit
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


def fake_pyneat(model_pixel_points):
    """A pyneat whose SuperPoint decoder returns the given keypoints in model pixels."""
    points = np.asarray(model_pixel_points, dtype=np.float32)
    features = SimpleNamespace(
        keypoints=SimpleNamespace(dtype="fp32", to_numpy=lambda copy: points.copy()),
        scores=SimpleNamespace(shape=(len(points),), dtype="fp32"),
        descriptors=SimpleNamespace(shape=(len(points), 256), dtype="fp32"),
    )
    return SimpleNamespace(
        TensorDType=SimpleNamespace(Float32="fp32"),
        decode_superpoint=lambda _output: [features],
    )


@pytest.mark.unit
def test_model_input_size_reads_the_resolved_mla_contract():
    example = load_example()
    contract = SimpleNamespace(width=640, height=480)
    plan = SimpleNamespace(mla_contract=contract)
    model = SimpleNamespace(resolved_preprocess_plan=lambda: plan)

    assert example.model_input_size(model) == example.FrameSize(640, 480)

    contract.width = 0
    with pytest.raises(RuntimeError, match="unresolved"):
        example.model_input_size(model)


@pytest.mark.unit
def test_feature_points_scale_model_pixels_onto_the_source_frame():
    example = load_example()
    model_size = example.FrameSize(640, 480)
    source_size = example.FrameSize(1280, 720)

    actual = example.feature_points(
        [object()], model_size, source_size, np, fake_pyneat([[600.0, 400.0]])
    )

    np.testing.assert_array_equal(actual, [[1200.0, 600.0]])


@pytest.mark.unit
def test_feature_points_keep_model_sized_frames_unchanged():
    example = load_example()
    model_size = example.FrameSize(640, 480)

    actual = example.feature_points(
        [object()], model_size, model_size, np, fake_pyneat([[10.0, 20.0]])
    )

    np.testing.assert_array_equal(actual, [[10.0, 20.0]])


@pytest.mark.unit
def test_feature_points_discards_coordinates_outside_the_source_frame():
    example = load_example()
    model_size = example.FrameSize(640, 480)
    source_size = example.FrameSize(1280, 720)
    pyneat = fake_pyneat(
        [[600.0, 400.0], [640.0, 200.0], [20.0, 480.0], [np.nan, 10.0]]
    )

    actual = example.feature_points([object()], model_size, source_size, np, pyneat)

    np.testing.assert_array_equal(actual, [[1200.0, 600.0]])


@pytest.mark.unit
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
                    resize=SimpleNamespace(enable=None, mode=None, width=0, height=0),
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
    assert (options.preprocess.resize.width, options.preprocess.resize.height) == (0, 0)
    assert options.preprocess.color_convert.input_format == "bgr"
    assert options.preprocess.color_convert.output_format == "gray8"
    assert options.preprocess.normalize.mean == (0.0, 0.0, 0.0)
    assert options.preprocess.normalize.stddev == (1.0, 1.0, 1.0)
    assert options.preprocess.normalize.has_explicit_stats is True


@pytest.mark.unit
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
            check=False,
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
