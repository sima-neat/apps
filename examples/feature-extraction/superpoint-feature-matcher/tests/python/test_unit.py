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


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """One NV12 array, luma rows then interleaved chroma rows, from BGR pixels."""
    height, width = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    luma, quarter = height * width, height * width // 4
    nv12 = np.empty((height * 3 // 2, width), np.uint8)
    nv12[:height] = i420[:luma].reshape(height, width)
    nv12[height:, 0::2] = i420[luma : luma + quarter].reshape(height // 2, width // 2)
    nv12[height:, 1::2] = i420[luma + quarter :].reshape(height // 2, width // 2)
    return nv12


def nv12_to_bgr(nv12: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


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

    nv12 = example.frame_nv12(frame)

    np.testing.assert_array_equal(nv12, bgr_to_nv12(bgr))
    assert example.luma_plane(nv12).flags["C_CONTIGUOUS"]


@pytest.mark.unit
def test_nv16_frames_keep_every_other_chroma_row():
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

    nv12 = example.frame_nv12(frame)

    np.testing.assert_array_equal(nv12, bgr_to_nv12(bgr))


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
def test_color_steps_run_from_red_at_the_threshold_to_green():
    example = load_example()

    steps = example.color_steps(np.array([0.1, 1.0, 0.0, 0.55]), 0.1)

    assert steps.tolist() == [0, example.COLOR_STEPS - 1, 0, 8]
    assert example.step_color(0) == (0, 0, 255)
    assert example.step_color(example.COLOR_STEPS - 1) == (0, 255, 0)


@pytest.mark.unit
def test_compose_frame_draws_each_line_in_its_step_color():
    example = load_example()
    frame = bgr_to_nv12(np.full((120, 160, 3), 128, np.uint8))
    lines = np.array([[10, 60, 170, 60], [10, 100, 170, 100]])

    canvas = example.compose_frame(frame, frame, lines, np.array([0, example.COLOR_STEPS - 1]), "")

    view = nv12_to_bgr(canvas).astype(int)
    assert np.abs(view[60, 100] - [0, 0, 255]).max() <= 6
    assert np.abs(view[100, 100] - [0, 255, 0]).max() <= 6


@pytest.mark.unit
def test_compose_frame_keeps_both_frames_side_by_side():
    example = load_example()
    reference = bgr_to_nv12(np.full((120, 160, 3), 50, np.uint8))
    live = bgr_to_nv12(np.full((120, 160, 3), 200, np.uint8))

    canvas = example.compose_frame(reference, live, np.empty((0, 4), int), np.empty(0, int), "")

    assert canvas.shape == (180, 320)
    view = nv12_to_bgr(canvas).astype(int)
    assert np.abs(view[60, 80] - 50).max() <= 2
    assert np.abs(view[60, 240] - 200).max() <= 2


@pytest.mark.unit
def test_compose_frame_marks_both_ends_of_each_line_with_a_dot():
    example = load_example()
    frame = bgr_to_nv12(np.full((120, 160, 3), 128, np.uint8))
    lines = np.array([[40, 60, 200, 60]])

    canvas = example.compose_frame(frame, frame, lines, np.array([example.COLOR_STEPS - 1]), "")

    view = nv12_to_bgr(canvas).astype(int)
    for x in (40, 200):
        assert np.abs(view[58, x] - [0, 255, 0]).max() <= 40
    assert np.abs(view[58, 120] - [0, 255, 0]).max() > 40


@pytest.mark.unit
def test_encoder_tensor_pads_to_the_encoder_layout(monkeypatch):
    example = load_example()
    made = SimpleNamespace(cvu=lambda: made)
    monkeypatch.setattr(
        example,
        "pyneat",
        SimpleNamespace(
            **vars(FAKE_PYNEAT),
            Plane=SimpleNamespace,
            TensorMemory=SimpleNamespace(CPU="cpu"),
            Tensor=SimpleNamespace(
                from_numpy=lambda array, **_: setattr(made, "array", array) or made
            ),
        ),
    )
    nv12 = bgr_to_nv12(np.full((54, 100, 3), 90, np.uint8))

    tensor = example.encoder_tensor(nv12)

    # 100 columns pad to a 128-byte pitch, and 54 rows to 56 storage rows.
    assert tensor.array.shape == (84, 128)
    np.testing.assert_array_equal(tensor.array[:54, :100], nv12[:54])
    np.testing.assert_array_equal(tensor.array[56:83, :100], nv12[54:])
    assert (tensor.shape, tensor.strides_bytes) == ([54, 100], [128, 1])
    assert [(p.role, p.byte_offset, p.shape) for p in tensor.planes] == [
        ("y", 0, [54, 100]),
        ("uv", 128 * 56, [27, 100]),
    ]


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
def test_match_stage_reports_a_rejected_matcher_frame():
    example = load_example()
    links = stage_links(example, captured_frames=1)
    features = example.Features(np.zeros((512, 2), np.float32), None, None)

    class OneOutputRunner(EmptyRunner):
        def pull(self, _timeout_ms):
            return SimpleNamespace(tensors=[object()])

    class RejectingRunner(EmptyRunner):
        def push(self, _inputs):
            return False

    example.superpoint_features = lambda _outputs: features
    matcher = example.Matcher("LightGlue", lambda _features: [], None)

    with pytest.raises(RuntimeError, match="LightGlue rejected a frame"):
        example.match_frames(OneOutputRunner(), matcher, RejectingRunner(), [], links)


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
