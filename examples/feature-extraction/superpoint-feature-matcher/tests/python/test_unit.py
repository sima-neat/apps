"""Unit tests for the Python SuperPoint feature matcher. They run without a board or pyneat."""

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
)


def load_example():
    spec = importlib.util.spec_from_file_location("superpoint_feature_matcher", MAIN_PY)
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
    def test_common_config_selects_lightglue_on_rtsp(self):
        cfg = load_example().load_app_config(COMMON_CONFIG)

        assert (cfg.matcher_type, cfg.source_type, cfg.source_codec) == (
            "lightglue",
            "rtsp",
            "h264",
        )
        assert cfg.matcher_path.endswith("lightglue_superpoint_w8a8_mpk.tar.gz")
        assert cfg.match_threshold == pytest.approx(0.1)

    def test_superglue_reads_only_its_own_section(self, tmp_path):
        path = write_config(tmp_path, matcher={"type": "superglue", "lightglue": {"path": ""}})

        cfg = load_example().load_app_config(path)

        assert cfg.matcher_path.endswith("superglue_outdoor_w16a16_mpk.tar.gz")
        assert (cfg.match_threshold, cfg.sinkhorn_iterations) == (pytest.approx(0.2), 20)

    def test_camera_sources_take_the_fps_from_their_mode(self, tmp_path):
        path = write_config(tmp_path, source={"type": "usb", "usb": {"camera": "cam0"}})

        cfg = load_example().load_app_config(path)

        assert (cfg.source_key, cfg.source_codec, cfg.source_fps) == (
            "source.usb.camera",
            "mjpeg",
            30,
        )

    @pytest.mark.parametrize(
        ("sections", "message"),
        [
            ({"matcher": {"type": "orb"}}, "matcher.type"),
            ({"source": {"type": "file"}}, "source.type"),
            ({"source": {"type": "usb"}}, "source.usb.camera must be set"),
            ({"matcher": {"lightglue": {"match_threshold": 1.0}}}, "match_threshold"),
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


@pytest.mark.unit
def test_nv12_frames_are_read_through_padded_strides_and_planes():
    example = load_example()
    width, height, stride, padded_rows = 64, 48, 96, 64
    bgr = color_blocks(width, height)
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y, u, v = i420[:height], i420[height : height * 5 // 4], i420[height * 5 // 4 :]
    payload = np.zeros(stride * padded_rows * 3 // 2, np.uint8)
    payload[: stride * height].reshape(height, stride)[:, :width] = y
    uv = payload[stride * padded_rows : stride * (padded_rows + height // 2)]
    uv = uv.reshape(height // 2, stride)
    uv[:, 0:width:2] = u.reshape(height // 2, width // 2)
    uv[:, 1:width:2] = v.reshape(height // 2, width // 2)
    frame = decoded_frame(
        {"y": (0, stride), "uv": (stride * padded_rows, stride)}, payload, width, height, False
    )

    luma, actual = example.frame_luma_and_bgr(frame)

    np.testing.assert_array_equal(luma[..., 0], y)
    assert luma.flags["C_CONTIGUOUS"]
    assert np.abs(actual.astype(int) - bgr).max() <= 3


@pytest.mark.unit
def test_nv16_frames_keep_full_height_chroma():
    example = load_example()
    width, height, stride = 64, 48, 80
    bgr = color_blocks(width, height)
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = i420[:height]
    # 4:2:2 keeps every chroma row; the quadrants are flat, so 4:2:0 rows repeat exactly.
    u = np.repeat(i420[height : height * 5 // 4].reshape(height // 2, width // 2), 2, axis=0)
    v = np.repeat(i420[height * 5 // 4 :].reshape(height // 2, width // 2), 2, axis=0)
    payload = np.zeros(stride * height * 2, np.uint8)
    payload[: stride * height].reshape(height, stride)[:, :width] = y
    uv = payload[stride * height :].reshape(height, stride)
    uv[:, 0:width:2] = u
    uv[:, 1:width:2] = v
    frame = decoded_frame(
        {"y": (0, stride), "uv": (stride * height, stride)}, payload, width, height, True
    )

    luma, actual = example.frame_luma_and_bgr(frame)

    np.testing.assert_array_equal(luma[..., 0], y)
    assert np.abs(actual.astype(int) - bgr).max() <= 3


@pytest.mark.unit
def test_lightglue_encoding_matches_the_rotary_formula():
    example = load_example()
    keypoints = np.array([[320.0, 240.0], [12.5, 470.0], [639.0, 3.0]], np.float32)

    encoding = example.lightglue_encoding(keypoints)

    assert encoding.shape == (3, 128) and encoding.dtype == np.float32
    np.testing.assert_allclose(encoding[0, :64], 1.0)
    np.testing.assert_allclose(encoding[0, 64:], 0.0)
    projection = np.asarray(example.LIGHTGLUE_PROJECTION)
    x, y = (keypoints[2].astype(float) - (320.0, 240.0)) / 320.0
    for index, (wx, wy) in enumerate(projection):
        phase = wx * x + wy * y
        assert (
            encoding[2, 2 * index]
            == encoding[2, 2 * index + 1]
            == pytest.approx(np.cos(phase), abs=1e-6)
        )
        assert encoding[2, 64 + 2 * index] == pytest.approx(np.sin(phase), abs=1e-6)


@pytest.mark.unit
def test_mutual_matches_keep_mutual_best_pairs_above_the_threshold():
    example = load_example()
    scores = np.array(
        [
            [0.9, 0.05, 0.0],  # mutual with live 0
            [0.6, 0.3, 0.0],  # its best, live 0, prefers reference 0
            [0.0, 0.0, 0.08],  # mutual, but below the threshold
        ],
        np.float32,
    )

    matches = example.mutual_matches(scores, np.ascontiguousarray(scores.T), 0.1)

    assert matches.reference.tolist() == [0]
    assert matches.live.tolist() == [0]
    np.testing.assert_allclose(matches.scores, [0.9])


def log_sinkhorn_matches(couplings, iterations, threshold):
    """SuperGlue's log_optimal_transport and mutual matching, written independently."""
    rows = cols = couplings.shape[0] - 1
    norm = -np.log(rows + cols)
    log_mu = np.r_[np.full(rows, norm), np.log(cols) + norm]
    log_nu = np.r_[np.full(cols, norm), np.log(rows) + norm]
    u, v = np.zeros(rows + 1), np.zeros(cols + 1)
    cost = couplings.astype(np.float64)
    for _ in range(iterations):
        u = log_mu - np.logaddexp.reduce(cost + v[None, :], axis=1)
        v = log_nu - np.logaddexp.reduce(cost + u[:, None], axis=0)
    transport = np.exp(cost + u[:, None] + v[None, :] - norm)[:-1, :-1]
    best_live, best_reference = transport.argmax(1), transport.argmax(0)
    reference = np.arange(rows)
    scores = transport[reference, best_live]
    keep = (best_reference[best_live] == reference) & (scores > threshold)
    return reference[keep], best_live[keep], scores[keep]


@pytest.mark.unit
def test_superglue_matches_agree_with_log_domain_sinkhorn():
    example = load_example()
    rng = np.random.default_rng(0)
    size = example.KEYPOINTS + 1
    couplings = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    true_pairs = rng.permutation(example.KEYPOINTS)[:200]
    couplings[np.arange(200), true_pairs] += 12.0

    matches = example.superglue_matches(couplings.reshape(1, 1, size, size), 0.2, 20)
    reference, live, scores = log_sinkhorn_matches(couplings, 20, 0.2)

    assert len(matches.reference) >= 150
    np.testing.assert_array_equal(matches.reference, reference)
    np.testing.assert_array_equal(matches.live, live)
    np.testing.assert_allclose(matches.scores, scores, rtol=1e-3)


@pytest.mark.unit
def test_match_lines_scale_model_pixels_and_draw_the_weakest_first():
    example = load_example()
    reference = example.Features(np.array([[10.0, 20.0], [100.0, 200.0]], np.float32), None, None)
    live = example.Features(np.array([[30.0, 40.0], [300.0, 400.0]], np.float32), None, None)
    matches = example.Matches(np.array([0, 1]), np.array([1, 0]), np.array([0.9, 0.3]))

    lines, scores = example.match_lines(matches, reference, live, np.array([2.0, 1.5]), 1280)

    assert lines.tolist() == [[200, 300, 1280 + 60, 60], [20, 30, 1280 + 600, 600]]
    np.testing.assert_allclose(scores, [0.3, 0.9])


@pytest.mark.unit
def test_match_colors_run_from_red_at_the_threshold_to_green():
    example = load_example()

    colors = example.match_colors(np.array([0.1, 1.0, 0.0]), 0.1)

    assert colors.tolist() == [[0, 0, 255], [0, 255, 0], [0, 0, 255]]


@pytest.mark.unit
def test_compose_frame_keeps_both_frames_side_by_side():
    example = load_example()
    reference = np.full((120, 160, 3), 50, np.uint8)
    live = np.full((120, 160, 3), 200, np.uint8)

    canvas = example.compose_frame(
        reference, live, np.empty((0, 4), int), np.empty((0, 3), int), ""
    )

    assert canvas.shape == (120, 320, 3)
    assert canvas[60, 80].tolist() == [50, 50, 50]
    assert canvas[60, 240].tolist() == [200, 200, 200]


@pytest.mark.unit
def test_match_stats_summarize_matched_and_skipped_frames(monkeypatch):
    example = load_example()
    clock = iter([10.0, 10.5, 11.0])
    monkeypatch.setattr(example.time, "perf_counter", lambda: next(clock))
    stats = example.MatchStats()

    stats.record(example.Rendered(None, 100, 0.8))
    stats.record(example.Rendered(None, None, None))
    stats.record(example.Rendered(None, 50, 0.6))

    assert stats.summary() == (
        "frames=3 matched=2 skipped=1 average_matches=75.0 inlier_ratio=0.70 output_fps=2.0"
    )
    assert stats.recent_fps() == pytest.approx(2.0)


def stage_links(example, captured_frames: int):
    import queue
    import threading

    links = example.StageLinks(queue.Queue(2), queue.Queue(2), threading.Event(), threading.Event())
    for _ in range(captured_frames):
        links.captured.put(object())
    links.captured_all.set()
    return links


class EmptyRunner:
    """A runner whose pull returns nothing, as on a timeout or after the last output."""

    def pull(self, _timeout_ms):
        return SimpleNamespace(tensors=[])

    def close_input(self):
        pass


@pytest.mark.unit
def test_match_stage_ends_once_every_captured_frame_is_matched():
    example = load_example()
    links = stage_links(example, captured_frames=0)

    example.match_frames(EmptyRunner(), None, EmptyRunner(), [], links)

    assert links.matched.get_nowait() is None


@pytest.mark.unit
def test_match_stage_reports_a_superpoint_stall_while_frames_wait():
    example = load_example()
    links = stage_links(example, captured_frames=1)

    with pytest.raises(RuntimeError, match="SuperPoint returned no output"):
        example.match_frames(EmptyRunner(), None, EmptyRunner(), [], links)


@pytest.mark.unit
@pytest.mark.parametrize(("width", "height", "level"), [(2560, 720, "4.0"), (3840, 1080, "5.1")])
def test_encoder_level_fits_the_composed_frame(width, height, level):
    assert load_example().encoder_level(width, height) == level


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
        path = write_config(tmp_path, superpoint={"path": str(tmp_path / "missing.tar.gz")})
        result = self.run("--config", str(path))
        assert result.returncode == 2
        assert "model does not exist" in result.stderr
