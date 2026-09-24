"""Unit tests for fastflow-anomaly-detector (Python)."""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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
            ({"inference": {"threshold": 1.5}}, "inference.threshold"),
            ({"inference": {"min_region_px": 0}}, "inference.min_region_px"),
            ({"runtime": {"profile_interval": 0}}, "runtime.profile_interval"),
        ],
    )
    def test_invalid_values_are_rejected(self, tmp_path, overrides, message):
        with pytest.raises(ValueError, match=message):
            main.load_config(write_config(tmp_path, overrides))


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
