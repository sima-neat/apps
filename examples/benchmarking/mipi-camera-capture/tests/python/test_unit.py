"""Unit tests for the Python MIPI camera capture example."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DEFAULT_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("mipi_camera_capture", MAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_help_runs_without_pyneat() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


@pytest.mark.unit
def test_default_config_validates_without_pyneat() -> None:
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(DEFAULT_CONFIG), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "config valid" in result.stdout


@pytest.mark.unit
def test_rejects_sample_time_after_duration(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["capture"]["duration_seconds"] = 2
    raw["capture"]["sample_times_seconds"] = [1, 3]
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "sample times must be finite, at least zero" in result.stderr


@pytest.mark.unit
def test_rejects_empty_output_directory(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["output"]["directory"] = ""
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "output.directory must not be empty" in result.stderr


@pytest.mark.unit
@pytest.mark.parametrize("duration", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_duration(tmp_path: Path, duration: float) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["capture"]["duration_seconds"] = duration
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "capture.duration_seconds must be finite and positive" in result.stderr


@pytest.mark.unit
def test_rejects_snapshot_without_one_frame_of_deadline_margin(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["capture"]["duration_seconds"] = 2
    raw["capture"]["sample_times_seconds"] = [1.999]
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "sample times must leave at least one requested frame period" in result.stderr


@pytest.mark.unit
def test_rejects_snapshot_schedule_above_memory_budget(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["capture"]["duration_seconds"] = 60
    raw["capture"]["sample_times_seconds"] = list(range(1, 51))
    config = tmp_path / "invalid.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "snapshot schedule may retain" in result.stderr
    assert "maximum is 134217728" in result.stderr


@pytest.mark.unit
def test_pull_timeout_is_bounded_by_remaining_duration() -> None:
    module = load_module()
    assert module.bounded_pull(2000, 10.0) == (2000, False)
    assert module.bounded_pull(2000, 0.1251) == (126, True)
    assert module.bounded_pull(2000, 0.0001) == (1, True)
    assert module.bounded_pull(2000, 1.9999) == (2000, True)


@pytest.mark.unit
def test_interarrival_stats_use_consecutive_pts() -> None:
    module = load_module()
    mean_ms, max_ms = module.pts_interarrival_stats_ms(
        [1_000_000_000, 1_033_333_333, 1_100_000_000]
    )
    assert mean_ms == pytest.approx(50.0)
    assert max_ms == pytest.approx(66.666667)


@pytest.mark.unit
def test_one_frame_consumes_every_reached_snapshot_target() -> None:
    module = load_module()
    targets = (0.01, 0.02, 0.10)
    assert module.reached_target_indices(targets, 0, 0.035) == range(0, 2)
    assert module.reached_target_indices(targets, 2, 0.035) == range(2, 2)


@pytest.mark.unit
def test_invalid_output_path_is_rejected_before_graph_build(tmp_path: Path) -> None:
    module = load_module()
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("occupied", encoding="utf-8")
    built = False

    class Graph:
        def __init__(self, _name: str) -> None:
            pass

        def add(self, _node) -> None:
            pass

        def build(self, _options):
            nonlocal built
            built = True
            raise AssertionError("graph must not be built")

    pyneat = SimpleNamespace(
        CameraInputOptions=SimpleNamespace,
        Graph=Graph,
        nodes=SimpleNamespace(
            camera_input=lambda *_args, **_kwargs: None,
            output=lambda *_args, **_kwargs: None,
        ),
        OutputOptions=SimpleNamespace(latest=lambda: None),
        RunOptions=lambda: SimpleNamespace(advanced=SimpleNamespace()),
        RunPreset=SimpleNamespace(Realtime="realtime"),
        OverflowPolicy=SimpleNamespace(KeepLatest="keep-latest"),
        OutputMemory=SimpleNamespace(ZeroCopy="zero-copy"),
    )
    config = module.AppConfig(
        camera=module.CameraConfig("", 2, 2, 30, 1, "NV12", 8, True, 2),
        capture=module.CaptureConfig(1.0, (), 100),
        output_directory=output_file,
    )

    with pytest.raises(FileExistsError):
        module.capture_frames(config, pyneat)
    assert not built


@pytest.mark.unit
def test_nv12_copy_removes_plane_stride_padding() -> None:
    module = load_module()
    y_role = "Y"
    uv_role = "UV"
    raw = bytes(
        [1, 2, 99, 99, 3, 4, 99, 99, 5, 6, 99, 99]
    )

    class Tensor:
        planes = [
            SimpleNamespace(role=y_role, byte_offset=0, strides_bytes=[4]),
            SimpleNamespace(role=uv_role, byte_offset=8, strides_bytes=[4]),
        ]

        @staticmethod
        def is_nv12() -> bool:
            return True

        @staticmethod
        def width() -> int:
            return 2

        @staticmethod
        def height() -> int:
            return 2

        @staticmethod
        def copy_payload_bytes() -> bytes:
            return raw

    assert module.nv12_contiguous_payload(Tensor(), 2, 2, y_role, uv_role) == bytes(
        [1, 2, 3, 4, 5, 6]
    )


@pytest.mark.unit
def test_frame_stats_reports_luma_distribution() -> None:
    module = load_module()
    payload = bytes([0, 10, 20, 255, 128, 128])
    stats = module.frame_stats(payload, 2, 2)
    assert stats == {
        "y_mean": 71.25,
        "y_min": 0,
        "y_max": 255,
        "y_p01": 0,
        "y_p50": 10,
        "y_p99": 255,
        "y_low_clip_pct": 25.0,
        "y_high_clip_pct": 25.0,
    }
