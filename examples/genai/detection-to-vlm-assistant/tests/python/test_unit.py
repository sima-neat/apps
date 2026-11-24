"""Unit tests for the detection-to-VLM example: the checks the application leaves to its config."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "detection_to_vlm_assistant_main")

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_input_model_and_labels():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert config.string("input.type") in main_module.INPUT_TYPES
    # A camera is opened in the mode the config names, so every value has to be set.
    for key in ("input.width", "input.height", "input.fps"):
        assert config.integer(key) > 0, key
    assert config.string("input.source")

    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolo26m_w16a16_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")
    # YOLO26's class IDs index the 80 COCO labels.
    labels = (EXAMPLE_DIR.parents[2] / config.string("model.labels")).read_text().splitlines()
    assert len(labels) == 80

    assert config.integer("inference.frames") >= 0
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    assert config.string("output.insight.host")


def test_unsupported_input_type_is_rejected_by_name():
    """main() names the key and the supported values, rather than failing later in the graph."""
    with pytest.raises(ValueError, match="input.type must be one of"):
        main_module.main(["--config", str(_config_with_input_type("rtsp"))])


def _config_with_input_type(input_type: str) -> Path:
    import tempfile

    text = (COMMON_DIR / "config.yaml").read_text().replace(
        "  type: mipi", f"  type: {input_type}", 1
    )
    path = Path(tempfile.mkdtemp()) / "config.yaml"
    path.write_text(text)
    return path


def test_strict_zero_copy_is_required_of_the_camera_stage():
    """A camera that copies frames instead of forwarding DMA-BUFs stops the run with a cause."""
    main_module.require_strict_zero_copy("backend: zero_copy=strict")
    with pytest.raises(RuntimeError, match="not forwarding DMA-BUFs"):
        main_module.require_strict_zero_copy("backend: zero_copy=relaxed")
