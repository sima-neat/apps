"""Unit tests for fastflow-anomaly-detector (Python)."""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.utils.fake_run import FakeRun

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"

_SPEC = importlib.util.spec_from_file_location("fastflow_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)

RUNNABLE_OVERRIDES = {
    "model": {"path": "models/fastflow_demo_mpk.tar.gz"},
    "source": {"rtsp_url": "rtsp://camera/live"},
    "output": {"insight": {"host": "127.0.0.1"}},
}


def deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    """Write the packaged config with the placeholders filled in, plus overrides."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    deep_update(config, copy.deepcopy(RUNNABLE_OVERRIDES))
    deep_update(config, overrides or {})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.unit
class TestConfig:
    def test_packaged_config_loads_once_placeholders_are_set(self, tmp_path):
        cfg = main.load_config(write_config(tmp_path))
        assert cfg.model_path == "models/fastflow_demo_mpk.tar.gz"
        assert cfg.rtsp_url == "rtsp://camera/live"
        assert (cfg.tcp, cfg.latency_ms, cfg.frames) == (True, 100, 0)
        assert (cfg.threshold, cfg.min_region_px) == (0.5, 300)
        assert (cfg.insight_host, cfg.video_port) == ("127.0.0.1", 9000)
        assert (cfg.heat_max, cfg.alpha) == (0.7, 0.55)
        assert (cfg.save_dir, cfg.save_every) == ("", 0)
        # The packaged config carries the fastflow_demo normalisation.
        assert cfg.mean == pytest.approx([0.3893437, 0.35421189, 0.36142577])
        assert cfg.stddev == [1.0, 1.0, 1.0]

    def test_normalize_is_read_from_config(self, tmp_path):
        overrides = {"model": {"normalize": {"mean": [0, 0, 0], "stddev": [1, 1, 1]}}}
        cfg = main.load_config(write_config(tmp_path, overrides))
        assert cfg.mean == [0.0, 0.0, 0.0]
        assert cfg.stddev == [1.0, 1.0, 1.0]

    def test_normalize_defaults_when_omitted(self, tmp_path):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        deep_update(config, copy.deepcopy(RUNNABLE_OVERRIDES))
        del config["model"]["normalize"]
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        cfg = main.load_config(path)
        assert cfg.mean == pytest.approx([0.3893437, 0.35421189, 0.36142577])
        assert cfg.stddev == [1.0, 1.0, 1.0]

    def test_validate_config_only_accepts_common_config(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
        )
        assert r.returncode == 0, r.stderr

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"source": {"rtsp_url": ""}}, "source.rtsp_url"),
            ({"output": {"insight": {"host": ""}}}, "output.insight.host"),
            ({"model": {"path": ""}}, "model.path"),
            ({"model": {"normalize": {"mean": [0.5, 0.5]}}}, "three numbers"),
            ({"model": {"normalize": {"mean": "a, b, c"}}}, "three numbers"),
            ({"model": {"normalize": {"stddev": [1.0, 0.0, 1.0]}}}, "model.normalize.stddev"),
            ({"model": {"normalize": {"stddev": [-1.0, 1.0, 1.0]}}}, "model.normalize.stddev"),
            ({"source": {"latency_ms": -1}}, "source.latency_ms"),
            ({"inference": {"frames": -1}}, "inference.frames"),
            ({"inference": {"threshold": 1.5}}, "inference.threshold"),
            ({"inference": {"min_region_px": 0}}, "inference.min_region_px"),
            ({"runtime": {"profile_interval": 0}}, "runtime.profile_interval"),
            ({"output": {"insight": {"video_port": 0}}}, "output.insight.video_port"),
            ({"output": {"save_every": -1}}, "output.save_every"),
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, overrides, message):
        with pytest.raises(ValueError, match=message):
            main.load_config(write_config(tmp_path, overrides))


# ---------------------------------------------------------------------------
# Configuration handling and option validation (Refs #526).
# TestConfig above owns the required keys, the normalisation triplet and one
# rejection per rule. These add the rules it does not reach, both ends of each
# range, the documented defaults, and malformed files.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRejectedValues:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            pytest.param({"output": {"alpha": 1.01}}, "output.alpha must be between 0 and 1", id="alpha-above"),
            pytest.param({"output": {"alpha": -0.01}}, "output.alpha must be between 0 and 1", id="alpha-below"),
            pytest.param(
                {"output": {"heat_max": 0.5}},
                "output.heat_max must be greater than inference.threshold",
                id="heat-max-equals-threshold",
            ),
            pytest.param(
                {"inference": {"threshold": 0.8}, "output": {"heat_max": 0.7}},
                "output.heat_max must be greater than inference.threshold",
                id="heat-max-below-threshold",
            ),
            pytest.param({"inference": {"threshold": -0.01}}, "inference.threshold must be between 0 and 1", id="threshold-below"),
            pytest.param({"output": {"insight": {"video_port": 65536}}}, "output.insight.video_port must be in [1, 65535]", id="video-port-above"),
            pytest.param({"inference": {"min_region_px": -5}}, "inference.min_region_px must be > 0", id="min-region-negative"),
        ],
    )
    def test_invalid_value_is_rejected_with_an_actionable_message(self, tmp_path, overrides, message):
        with pytest.raises(ValueError) as excinfo:
            main.load_config(write_config(tmp_path, overrides))

        assert message in str(excinfo.value)


@pytest.mark.unit
class TestBoundariesAreAccepted:
    """An off-by-one that rejects a legal value is the failure nobody writes a test for."""

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param({"inference": {"threshold": 0.0}}, id="threshold-zero"),
            pytest.param({"inference": {"threshold": 1.0}, "output": {"heat_max": 1.5}}, id="threshold-one"),
            pytest.param({"output": {"heat_max": 0.5001}}, id="heat-max-just-above-threshold"),
            pytest.param({"output": {"alpha": 0.0}}, id="alpha-zero"),
            pytest.param({"output": {"alpha": 1.0}}, id="alpha-one"),
            pytest.param({"source": {"latency_ms": 0}}, id="latency-zero"),
            pytest.param({"inference": {"frames": 0}}, id="frames-zero"),
            pytest.param({"inference": {"min_region_px": 1}}, id="min-region-one"),
            pytest.param({"runtime": {"profile_interval": 1}}, id="profile-interval-one"),
            pytest.param({"output": {"insight": {"video_port": 1}}}, id="video-port-one"),
            pytest.param({"output": {"insight": {"video_port": 65535}}}, id="video-port-max"),
            pytest.param({"output": {"save_every": 0}}, id="save-every-zero"),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, overrides):
        main.load_config(write_config(tmp_path, overrides))


@pytest.mark.unit
class TestDocumentedDefaults:
    def test_omitted_sections_fall_back_to_the_loader_defaults(self, tmp_path):
        raw = {
            "model": {"path": "models/fastflow_demo_mpk.tar.gz"},
            "source": {"rtsp_url": "rtsp://camera/live"},
            "output": {"insight": {"host": "127.0.0.1"}},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        cfg = main.load_config(path)

        assert cfg.mean == pytest.approx(main.DEFAULT_MEAN)
        assert cfg.stddev == pytest.approx(main.DEFAULT_STDDEV)
        assert (cfg.tcp, cfg.latency_ms, cfg.frames) == (True, 100, 0)
        assert (cfg.threshold, cfg.min_region_px) == (0.5, 300)
        assert (cfg.profile, cfg.profile_interval) == (False, 100)
        assert cfg.video_port == 9000
        assert (cfg.heat_max, cfg.alpha) == (0.7, 0.55)
        assert (cfg.save_dir, cfg.save_every) == ("", 0)


@pytest.mark.unit
class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        """A list at the root used to escape as an AttributeError from `raw.get`."""
        path = tmp_path / "config.yaml"
        path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main.load_config(path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="model.path must be set"):
            main.load_config(path)

    def test_a_non_integer_value_is_rejected(self, tmp_path):
        """Raised as TypeError here, unlike the other applications' ValueError, and
        without the section prefix; main() reports both the same way, so the
        message is what matters."""
        with pytest.raises((TypeError, ValueError), match="latency_ms must be an integer"):
            main.load_config(write_config(tmp_path, {"source": {"latency_ms": "fast"}}))


@pytest.mark.unit
def test_regions_from_map_keeps_large_regions_only():
    np = pytest.importorskip("numpy")
    main.cv2 = pytest.importorskip("cv2")
    main.np = np

    anomaly_map = np.zeros((256, 256), dtype=np.float32)
    anomaly_map[10:50, 10:50] = 0.9      # 1600 px: a defect
    anomaly_map[100:104, 100:104] = 0.8  # 16 px: speckle

    regions, mask = main.regions_from_map(anomaly_map, threshold=0.5, min_region_px=300)

    assert len(regions) == 1
    assert regions[0]["bbox"] == (10, 10, 40, 40)
    assert regions[0]["score"] == pytest.approx(0.9)
    assert mask.sum() == 1600  # only the defect's pixels carry the heatmap


@pytest.mark.unit
def test_render_draws_the_heatmap_only_over_the_regions():
    np = pytest.importorskip("numpy")
    main.cv2 = pytest.importorskip("cv2")
    main.np = np

    anomaly_map = np.zeros((256, 256), dtype=np.float32)
    anomaly_map[10:50, 10:50] = 0.9
    regions, mask = main.regions_from_map(anomaly_map, threshold=0.5, min_region_px=300)
    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    config = main.Config(
        model_path="m", mean=[0, 0, 0], stddev=[1, 1, 1], rtsp_url="rtsp://x", tcp=True,
        latency_ms=100, frames=0, threshold=0.5, min_region_px=300, profile=False,
        profile_interval=100, insight_host="127.0.0.1", video_port=9000, heat_max=0.7,
        alpha=0.55, save_dir="", save_every=0,
    )
    main.render(frame, anomaly_map, regions, mask, config, 30.0)

    # The map's 10..50 square covers 20..100 of the 512 px frame.
    assert frame[60, 60].any()                # the heatmap is painted inside the region
    assert not frame[300:500, 300:500].any()  # a good area keeps the original frame
    assert frame[18, 20].any()                # the verdict banner is drawn on every frame


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the single RTSP Insight anomaly pipeline."""

    def test_help(self):
        """--help should describe the config-driven CLI."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        """Missing config should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/fastflow-config.yaml"],
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

    def test_validate_config_only(self, tmp_path):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(write_config(tmp_path)), "--validate-config-only"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "Config validated" in r.stdout


@pytest.mark.unit
class TestPullOutcomes:
    """The frame loop, driven by a run that yields no sample.

    A timeout is a warning and another pull; a closed output ends the run with a message
    instead of waiting forever on a dead stream.
    """

    def test_timeout_is_not_a_sample(self):
        run = FakeRun("timeout")
        run.pull("frame", main.PULL_TIMEOUT_MS)

        assert main.pull_result_has_sample(run, None, "frame") is False

    def test_closed_output_ends_the_run_with_the_reason(self):
        run = FakeRun(("closed", "source reached EOS"))
        run.pull("frame", main.PULL_TIMEOUT_MS)

        with pytest.raises(
            RuntimeError, match="frame output closed unexpectedly: source reached EOS"
        ):
            main.pull_result_has_sample(run, None, "frame")

    def test_runtime_error_ends_the_run(self):
        run = FakeRun(("error", "queue torn down"))
        run.pull("frame", main.PULL_TIMEOUT_MS)

        with pytest.raises(RuntimeError, match="runtime error: queue torn down"):
            main.pull_result_has_sample(run, None, "frame")

    def test_run_warns_on_timeout_and_stops_on_closed_source(self, monkeypatch, capsys):
        run = FakeRun("timeout", ("closed", "source reached EOS"))
        video = SimpleNamespace(port=9000, closed=False)
        video.close = lambda: setattr(video, "closed", True)
        monkeypatch.setattr(main, "probe_stream", lambda url, tcp: (640, 640, 30))
        monkeypatch.setattr(main, "build_model", lambda cfg, width, height: (object(), 32))
        monkeypatch.setattr(main, "InsightVideo", lambda cfg, width, height, fps: video)
        monkeypatch.setattr(main, "build_source", lambda cfg, width, height, fps: (object(), run))
        cfg = SimpleNamespace(
            rtsp_url="rtsp://camera/live", tcp=True, frames=0, threshold=0.5, min_region_px=300,
            insight_host="127.0.0.1", save_dir="", save_every=0, profile=False, profile_interval=1,
        )

        with pytest.raises(
            RuntimeError, match="frame output closed unexpectedly: source reached EOS"
        ):
            main.run(cfg)

        captured = capsys.readouterr()
        assert captured.err.count("[warn] timed out waiting for a frame") == 1
        assert run.pulls == [("frame", main.PULL_TIMEOUT_MS)] * 2
        # The run and the video sender are still released on the way out.
        assert run.closed_by_app and video.closed
