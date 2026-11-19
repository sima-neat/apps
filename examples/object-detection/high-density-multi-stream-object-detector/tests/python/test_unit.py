"""Unit tests for the high-density multi-stream object detector."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(
    EXAMPLE_DIR, "high_density_multi_stream_object_detector_main"
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("filename", "streams", "fps"),
    [
        ("config.yaml", 20, 0),
        ("config-30x720p20fps.yaml", 30, 20),
        ("config-60x720p10fps.yaml", 60, 10),
    ],
)
def test_shipped_profiles_load(filename, streams, fps):
    cfg = main_module.load_app_config(COMMON_DIR / filename)
    assert len(cfg.sources) == streams
    assert cfg.fps == fps
    assert (cfg.width, cfg.height) == (0, 0), "RTSP streams report their own frame size"
