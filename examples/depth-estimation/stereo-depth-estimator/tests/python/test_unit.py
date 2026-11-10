"""Unit tests for the Python stereo depth estimator. They run without a board or pyneat."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"
FAKE_PYNEAT = SimpleNamespace(
    PixelFormat=SimpleNamespace(NV12="nv12", NV16="nv16"),
    PlaneRole=SimpleNamespace(Y="y", UV="uv"),
    SampleKind=SimpleNamespace(Tensor="tensor"),
)


def load_example():
    spec = importlib.util.spec_from_file_location("stereo_depth_estimator", MAIN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses resolve their module's annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.cv2, module.np, module.pyneat = cv2, np, FAKE_PYNEAT
    return module


def merge(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        base[key] = merge(base[key], value) if isinstance(value, dict) else value
    return base


def write_config(tmp_path, **sections):
    config = merge(yaml.safe_load(COMMON_CONFIG.read_text(encoding="utf-8")), sections)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.unit
class TestConfig:
    def test_common_config_reads_two_rtsp_cameras(self):
        cfg = load_example().load_app_config(COMMON_CONFIG)

        assert (cfg.source_type, cfg.codec) == ("rtsp", "h264")
        assert cfg.cameras == ("<left-rtsp-url>", "<right-rtsp-url>")
        assert cfg.model_path.endswith("lightstereo_s_w16a16_mpk.tar.gz")
        assert cfg.max_disparity == pytest.approx(96.0)

    @pytest.mark.parametrize("source_type", ["usb", "mipi"])
    def test_camera_sources_read_only_their_own_section(self, tmp_path, source_type):
        cameras = {"left_camera": "cam-left", "right_camera": "cam-right"}
        path = write_config(
            tmp_path,
            source={"type": source_type, source_type: cameras, "rtsp": {"left_url": ""}},
        )

        cfg = load_example().load_app_config(path)

        assert cfg.cameras == ("cam-left", "cam-right")
        assert min(cfg.width, cfg.height, cfg.fps) > 0

    @pytest.mark.parametrize(
        ("sections", "message"),
        [
            ({"model": {"path": ""}}, "model.path must be set"),
            ({"source": {"type": "file"}}, "source.type"),
            ({"source": {"rtsp": {"codec": "vp9"}}}, "source.rtsp.codec"),
            ({"source": {"rtsp": {"right_url": ""}}}, "source.rtsp.right_url must be set"),
            (
                {"source": {"rtsp": {"right_url": "<left-rtsp-url>"}}},
                "source.rtsp.right_url must differ",
            ),
            ({"source": {"type": "usb"}}, "source.usb.right_camera must be set"),
            ({"source": {"rtsp": {"tcp": "yes"}}}, "tcp must be of type bool"),
            ({"inference": {"frames": -1}}, "inference.frames"),
            ({"output": {"max_disparity": 0}}, "output.max_disparity"),
            ({"output": {"insight": {"video_port": 0}}}, "video_port"),
        ],
    )
    def test_invalid_settings_are_reported(self, tmp_path, sections, message):
        with pytest.raises(ValueError, match=message):
            load_example().load_app_config(write_config(tmp_path, **sections))


def decoded_frame(planes: dict, payload: np.ndarray, width: int, height: int, nv16: bool):
    """A pyneat tensor's frame accessors, over a padded payload."""
    return SimpleNamespace(
        width=lambda: width,
        height=lambda: height,
        copy_payload_bytes=lambda: payload.tobytes(),
        planes=[
            SimpleNamespace(role=role, byte_offset=offset, strides_bytes=(stride, 1))
            for role, (offset, stride) in planes.items()
        ],
        image_format=lambda: "nv16" if nv16 else "nv12",
        is_nv12=lambda: not nv16,
    )


def color_blocks(width: int, height: int) -> np.ndarray:
    """Four flat BGR quadrants, so that chroma subsampling leaves them unchanged."""
    bgr = np.empty((height, width, 3), np.uint8)
    half_h, half_w = height // 2, width // 2
    bgr[:half_h, :half_w] = (40, 80, 200)
    bgr[:half_h, half_w:] = (200, 60, 30)
    bgr[half_h:, :half_w] = (30, 190, 70)
    bgr[half_h:, half_w:] = (128, 128, 128)
    return bgr


def nv12_frame(bgr: np.ndarray, stride: int, padded_rows: int):
    """The BGR pixels as a decoder would hand them over: NV12 with padded rows and planes."""
    height, width = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y, u, v = i420[:height], i420[height : height * 5 // 4], i420[height * 5 // 4 :]
    payload = np.zeros(stride * padded_rows * 3 // 2, np.uint8)
    payload[: stride * height].reshape(height, stride)[:, :width] = y
    uv = payload[stride * padded_rows : stride * (padded_rows + height // 2)]
    uv = uv.reshape(height // 2, stride)
    uv[:, 0:width:2] = u.reshape(height // 2, width // 2)
    uv[:, 1:width:2] = v.reshape(height // 2, width // 2)
    planes = {"y": (0, stride), "uv": (stride * padded_rows, stride)}
    return decoded_frame(planes, payload, width, height, False)


def round_trip(bgr: np.ndarray) -> np.ndarray:
    """The pixels after BGR to 4:2:0 and back, which is what a decoded frame holds."""
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420), cv2.COLOR_YUV2BGR_I420)


@pytest.mark.unit
def test_nv12_frames_are_read_through_padded_strides_and_planes():
    example = load_example()
    bgr = color_blocks(64, 48)

    frame = example.frame_bgr(nv12_frame(bgr, stride=96, padded_rows=64))

    np.testing.assert_array_equal(frame, round_trip(bgr))


@pytest.mark.unit
def test_nv16_frames_keep_every_other_chroma_row():
    example = load_example()
    width, height, stride = 64, 48, 80
    bgr = color_blocks(width, height)
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    # 4:2:2 keeps every chroma row; the quadrants are flat, so 4:2:0 rows repeat exactly.
    u = np.repeat(i420[height : height * 5 // 4].reshape(height // 2, width // 2), 2, axis=0)
    v = np.repeat(i420[height * 5 // 4 :].reshape(height // 2, width // 2), 2, axis=0)
    payload = np.zeros(stride * height * 2, np.uint8)
    payload[: stride * height].reshape(height, stride)[:, :width] = i420[:height]
    uv = payload[stride * height :].reshape(height, stride)
    uv[:, 0:width:2] = u
    uv[:, 1:width:2] = v
    planes = {"y": (0, stride), "uv": (stride * height, stride)}

    frame = example.frame_bgr(decoded_frame(planes, payload, width, height, True))

    np.testing.assert_array_equal(frame, round_trip(bgr))


def source_run(frames: dict):
    """A source run whose pull returns one decoded frame per view, or None for a missing view."""

    def pull(view, _timeout_ms):
        if view not in frames:
            return None
        return SimpleNamespace(fields=[], kind="tensor", tensor=frames[view])

    return SimpleNamespace(pull=pull)


@pytest.mark.unit
def test_pull_pair_returns_the_left_then_the_right_view():
    example = load_example()
    left, right = color_blocks(64, 48), color_blocks(64, 48)[::-1].copy()
    run = source_run({"left": nv12_frame(left, 64, 48), "right": nv12_frame(right, 64, 48)})

    pulled_left, pulled_right = example.pull_pair(run)

    np.testing.assert_array_equal(pulled_left, round_trip(left))
    np.testing.assert_array_equal(pulled_right, round_trip(right))


@pytest.mark.unit
def test_pull_pair_names_the_camera_that_sent_no_frame():
    example = load_example()
    run = source_run({"left": nv12_frame(color_blocks(64, 48), 64, 48)})

    with pytest.raises(RuntimeError, match="from the right camera"):
        example.pull_pair(run)


@pytest.mark.unit
def test_pull_pair_rejects_cameras_with_different_frame_sizes():
    example = load_example()
    run = source_run(
        {
            "left": nv12_frame(color_blocks(64, 48), 64, 48),
            "right": nv12_frame(color_blocks(32, 24), 32, 24),
        }
    )

    with pytest.raises(RuntimeError, match="left is 64x48, right is 32x24"):
        example.pull_pair(run)


@pytest.mark.unit
def test_heatmap_runs_from_the_darkest_color_at_zero_to_the_brightest_at_the_maximum():
    example = load_example()
    palette = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], cv2.COLORMAP_INFERNO)[0]
    disparity = np.float32([[0.0, 48.0, 96.0, 188.0]])

    heatmap = example.disparity_heatmap(disparity, 96.0)

    np.testing.assert_array_equal(heatmap[0], palette[[0, 128, 255, 255]])


@pytest.mark.unit
def test_canvas_holds_the_resized_left_view_beside_the_heatmap():
    example = load_example()
    left = np.full((96, 128, 3), (10, 20, 30), np.uint8)
    heatmap = np.full((48, 64, 3), (200, 100, 50), np.uint8)

    canvas = example.compose_canvas(left, heatmap)

    assert canvas.shape == (48, 128, 3)
    assert (canvas[:, :64] == (10, 20, 30)).all()
    np.testing.assert_array_equal(canvas[:, 64:], heatmap)


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

    def test_missing_config_is_reported(self):
        result = self.run("--config", "/does/not/exist.yaml")
        assert result.returncode == 2
        assert "Error:" in result.stderr

    def test_missing_model_is_reported(self, tmp_path):
        path = write_config(tmp_path, model={"path": str(tmp_path / "missing.tar.gz")})
        result = self.run("--config", str(path))
        assert result.returncode == 2
        assert "model does not exist" in result.stderr
