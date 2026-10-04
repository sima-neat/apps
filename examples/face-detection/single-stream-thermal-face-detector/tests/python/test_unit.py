"""Unit tests for single-stream-thermal-face-detector (Python)."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.utils.config_cases import config_writer, load_example_main
from tests.utils.fake_run import FakeRun

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"

# Pyramid grid sizes for the 800x800 canvas the model was compiled for.
LEVEL_SIZES = (100, 50, 25)


def load_example():
    module = load_example_main(EXAMPLE_DIR, "thermal_face_example")
    module.np = np  # main.py binds numpy lazily at runtime
    return module


def build_nhwc_heads(example):
    """Six [1,H,W,C] split heads with one planted high-confidence face per level."""
    rng = np.random.default_rng(0)
    heads = []
    for size in LEVEL_SIZES:
        for channels in (example.BOX_CHANNELS, example.LM_CHANNELS):
            head = rng.standard_normal((1, size, size, channels)).astype(np.float32) * 0.5
            if channels == example.BOX_CHANNELS:
                head[0, size // 3, size // 4, 4] = 4.0  # objectness logit
                head[0, size // 3, size // 4, 5] = 4.0  # class logit
            heads.append(head)
    return heads


@pytest.mark.unit
class TestSplitHeadLayouts:
    """The decoder must accept every layout a raw split head can arrive in.

    NEAT delivers raw heads as [1,H,C,W] (the RetinaFace example documents the
    same), so a decoder that only probed the last and first axes rejected the
    real runtime tensors and published no metadata at all.
    """

    def decode(self, example, heads):
        return example.decode_yolov5face_split(
            heads, conf_threshold=0.25, iou_threshold=0.45, max_detections=50)

    def test_nhwc_layout_decodes(self):
        example = load_example()
        boxes, scores, landmarks = self.decode(example, build_nhwc_heads(example))
        assert len(boxes) > 0
        assert scores.shape == (len(boxes),)
        assert landmarks.shape == (len(boxes), example.NUM_LANDMARKS, 2)

    @pytest.mark.parametrize("name,axes", [
        ("neat_hcw", (0, 1, 3, 2)),  # [1,H,W,C] -> [1,H,C,W], NEAT's raw layout
        ("nchw", (0, 3, 1, 2)),      # [1,H,W,C] -> [1,C,H,W], ONNX-native
    ])
    def test_permuted_layouts_match_nhwc(self, name, axes):
        """A permuted head must decode to exactly the NHWC result."""
        example = load_example()
        nhwc = build_nhwc_heads(example)
        permuted = [np.ascontiguousarray(h.transpose(*axes)) for h in nhwc]

        expected = self.decode(example, nhwc)
        actual = self.decode(example, permuted)

        for got, want, field in zip(actual, expected, ("boxes", "scores", "landmarks")):
            assert np.allclose(got, want, atol=1e-5), f"{name}: {field} differ"

    def test_unrecognized_shape_is_rejected(self):
        example = load_example()
        bogus = [np.zeros((1, 7, 7, 7), dtype=np.float32)]
        with pytest.raises(ValueError, match="Unrecognized split output shape"):
            self.decode(example, bogus)


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the single-stream yolov5s-face pipeline."""

    def test_help(self):
        """--help should describe the config-driven CLI."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        """A missing config file should produce a nonzero exit."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/single-stream-thermal-face-detector-config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_validate_config_only(self):
        """--validate-config-only should parse the shipped config without touching hardware."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(COMMON_CONFIG), "--validate-config-only"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "validated" in r.stdout.lower()

    def test_validate_config_rejects_empty_labels_path(self, tmp_path):
        """An explicit empty path must not be normalized to the current directory."""
        config = COMMON_CONFIG.read_text(encoding="utf-8")
        config = config.replace(
            "labels: examples/face-detection/single-stream-thermal-face-detector/src/common/face_label.txt",
            'labels: ""',
        )
        config_path = tmp_path / "empty-labels.yaml"
        config_path.write_text(config, encoding="utf-8")

        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode != 0
        assert "model.labels must be set" in r.stderr


@pytest.mark.unit
def test_single_frame_profile_includes_frame_elapsed_time(monkeypatch, capsys):
    example = load_example()
    timestamps = iter((100.0, 110.0))
    monkeypatch.setattr(example, "time_ms", lambda: next(timestamps))

    profile = example.ProfileWindow(enabled=True, interval=1)
    profile.start_frame()
    profile.add(pull_ms=6.0, decode_ms=3.0, metadata_ms=1.0, face_count=1)

    assert "output_fps=100.0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Configuration handling and option validation (Refs #526).
# Each test starts from VALID_CONFIG and breaks exactly one thing, so a failure
# names the rule that fired rather than "config rejected". The empty labels
# value is owned by TestArgParsing above through --validate-config-only.
# ---------------------------------------------------------------------------

main_module = load_example()

VALID_CONFIG = {
    "model": {"path": "model.tar.gz", "labels": "face_label.txt"},
    "source": {"rtsp_url": "rtsp://127.0.0.1:8554/src1", "tcp": True, "latency_ms": 100},
    "inference": {"frames": 0, "min_score": 0.25, "nms_iou": 0.45, "max_detections": 50},
    "runtime": {"profile": False, "profile_interval": 100},
    "output": {"insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100}},
}


# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_full_config = config_writer(VALID_CONFIG)


@pytest.mark.unit
class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main_module.load_app_config(write_full_config(tmp_path))

        assert cfg.model_path == "model.tar.gz"
        assert cfg.rtsp_url == "rtsp://127.0.0.1:8554/src1"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": "model.tar.gz"},
            "source": {"rtsp_url": "rtsp://127.0.0.1:8554/src1"},
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main_module.load_app_config(write_full_config(tmp_path, root=raw))

        assert cfg.latency_ms == 200
        assert cfg.tcp == True
        assert cfg.frames == 0
        assert cfg.min_score == pytest.approx(0.25)
        assert cfg.nms_iou == pytest.approx(0.45)
        assert cfg.max_detections == 50
        assert cfg.profile == False
        assert cfg.profile_interval == 100
        assert cfg.video_port == 9000
        assert cfg.metadata_port == 9100
        assert cfg.labels_path.name == "face_label.txt"


REJECTED = [
    pytest.param(('source', 'rtsp_url'), '', 'source.rtsp_url must be set', id='rtsp-url-empty'),
    pytest.param(('model', 'path'), '', 'model.path must be set', id='model-path-empty'),
    pytest.param(('output', 'insight', 'host'), '', 'output.insight.host must be set', id='insight-host-empty'),
    pytest.param(('source', 'latency_ms'), -1, 'source.latency_ms must be >= 0', id='latency-negative'),
    pytest.param(('inference', 'frames'), -1, 'inference.frames must be >= 0', id='frames-negative'),
    pytest.param(('inference', 'min_score'), -0.01, 'inference.min_score must be between 0 and 1', id='min-score-below'),
    pytest.param(('inference', 'min_score'), 1.01, 'inference.min_score must be between 0 and 1', id='min-score-above'),
    pytest.param(('inference', 'nms_iou'), -0.01, 'inference.nms_iou must be between 0 and 1', id='nms-below'),
    pytest.param(('inference', 'nms_iou'), 1.01, 'inference.nms_iou must be between 0 and 1', id='nms-above'),
    pytest.param(('inference', 'max_detections'), 0, 'inference.max_detections must be > 0', id='max-detections-zero'),
    pytest.param(('runtime', 'profile_interval'), 0, 'runtime.profile_interval must be > 0', id='profile-interval-zero'),
    pytest.param(('output', 'insight', 'video_port'), 0, 'output.insight.video_port must be > 0', id='video-port-zero'),
    pytest.param(('output', 'insight', 'metadata_port'), 0, 'output.insight.metadata_port must be > 0', id='metadata-port-zero'),
]


@pytest.mark.unit
class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        with pytest.raises(ValueError) as excinfo:
            main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))

        assert message in str(excinfo.value)


@pytest.mark.unit
class TestBoundariesAreAccepted:
    """An off-by-one that rejects a legal value is the failure nobody writes a test for."""

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (('inference', 'min_score'), 0.0),
            (('inference', 'min_score'), 1.0),
            (('inference', 'nms_iou'), 0.0),
            (('inference', 'nms_iou'), 1.0),
            (('source', 'latency_ms'), 0),
            (('inference', 'frames'), 0),
            (('inference', 'max_detections'), 1),
            (('runtime', 'profile_interval'), 1),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main_module.load_app_config(write_full_config(tmp_path, [(path, value)]))


@pytest.mark.unit
class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        """Raised as TypeError here, unlike the other applications; main() reports
        both the same way, so the message is what matters."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises((TypeError, ValueError), match="config root must be a mapping"):
            main_module.load_app_config(config_path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="source.rtsp_url must be set"):
            main_module.load_app_config(config_path)


@pytest.mark.unit
class TestPullOutcomes:
    """The pull loop, driven by a run that yields no sample.

    A timeout is a warning and another pull, and a runtime error raised by the pull ends
    the run without a summary. pyneat's pull returns None for a closed output as for a
    timeout, so Python cannot yet tell a source that ended from a slow one.
    """

    def test_timeout_is_not_a_sample(self):
        example = load_example()
        run = FakeRun("timeout")
        run.pull("detections", 20000)

        assert example.pull_result_has_sample(run, None, "detections") is False

    def test_run_pipeline_warns_on_timeout_and_stops_on_runtime_error(self, capsys):
        example = load_example()
        run = FakeRun("timeout", ("error", "queue torn down"))
        runtime = SimpleNamespace(run=run)
        cfg = SimpleNamespace(frames=0, profile=False, profile_interval=1)

        with pytest.raises(RuntimeError, match="queue torn down"):
            example.run_pipeline(runtime, cfg)

        captured = capsys.readouterr()
        assert captured.err.count("[warn] timed out waiting for detections") == 1
        assert "processed=" not in captured.out
        assert run.pulls == [("detections", 20000)] * 2
