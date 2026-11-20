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
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / filename)
    assert len(config.string_list("input.source")) == streams
    assert config.integer("input.fps") == fps
    assert (config.integer("input.width"), config.integer("input.height")) == (0, 0), (
        "RTSP streams report their own frame size"
    )
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert (EXAMPLE_DIR.parents[2] / config.string("model.labels")).is_file()
    for key in ("input.rtsp.latency_ms", "inference.max_detections",
                "inference.max_inflight_per_stream", "inference.max_inflight_total",
                "app.decoder_buffers", "app.queue_depth", "app.internal_queue_depth"):
        assert config.integer(key) > 0, key
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    assert config.string("output.insight.host")
