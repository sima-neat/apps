"""Unit tests for the RF-DETR detection and segmentation example."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "rfdetr_detection_segmentation_main")
postprocess = main_module.postprocess

pytestmark = pytest.mark.unit

LABELS = ["unused", "person"] + ["unused"] * 89


def test_shipped_config_names_a_supported_model_stream_and_labels():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert config.string("model.id") == "det-small"
    model_path = COMMON_DIR / config.string("model.path")
    # The decoder package sits next to the backbone that model.path names.
    assert model_path.name == "rf_detr_small_w8a8_mixed_w16a16_backbone_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    # The model's class IDs index the labels directly.
    labels = (EXAMPLE_DIR.parents[2] / config.string("model.labels")).read_text().splitlines()
    assert len(labels) == postprocess.NUM_CLASSES
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames", "inference.max_detections", "app.mask_grid_size",
                "output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert config.integer(key) >= 0, key
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("app.mask_threshold") <= 1
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def test_top_proposals_are_stable_and_best_first():
    scores = np.zeros((1, 305), dtype=np.float32)
    scores[0, 3] = scores[0, 4] = 2.0
    proposals = np.zeros((1, 305, 4), dtype=np.float32)
    proposals[0, :, 0] = np.arange(305)
    best = postprocess.top_proposals(scores, proposals, 300)
    assert best.shape == (300, 4)
    assert list(best[:3, 0]) == [3.0, 4.0, 0.0]


def test_detection_skips_unused_classes_and_scales_to_the_frame():
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 0] = boxes[0, 1] = [0.5, 0.5, 0.5, 0.25]
    logits = np.full((1, 300, 91), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0  # person
    logits[0, 1, 2] = 12.0  # an unused class, ranked first
    options = postprocess.PostprocessOptions(1920, 1080, 0.5, 10, 0.08, 640)
    objects = json.loads(postprocess.detection_json(boxes, logits, LABELS, options))[
        "objects"
    ]
    assert len(objects) == 1
    assert objects[0]["id"] == "obj_1" and objects[0]["label"] == "person"
    assert objects[0]["bbox"] == pytest.approx([480.0, 405.0, 960.0, 270.0])


@pytest.mark.parametrize(("queries", "mask_size"), [(200, 108), (100, 78)])
@pytest.mark.parametrize("mask_grid_size", [None, 432, 640])
def test_segmentation_traces_a_polygon_inside_the_box(
    queries, mask_size, mask_grid_size
):
    boxes = np.zeros((1, queries, 4), dtype=np.float32)
    boxes[0, 0] = boxes[0, 1] = [0.5, 0.5, 0.5, 0.5]
    logits = np.full((1, queries, 91), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 11.0  # person on query 0
    logits[0, 1, 1] = 10.0  # person on query 1, whose mask is empty
    masks = np.full((1, mask_size, mask_size, queries), -20.0, dtype=np.float32)
    lo, hi = mask_size * 3 // 8, mask_size * 5 // 8
    masks[0, lo:hi, lo:hi, 0] = 10.0
    options = postprocess.PostprocessOptions(
        1280, 720, 0.3, 24, 0.08, mask_grid_size or mask_size
    )
    payload = postprocess.segmentation_json(
        boxes, logits, masks, mask_size, LABELS, options
    )
    segments = json.loads(payload)["segments"]
    assert len(segments) == 1
    segment = segments[0]
    assert segment["label"] == "person" and segment["mask_format"] == "polygon"
    assert segment["bbox"] == [320, 180, 640, 360]
    assert len(segment["mask"]) >= 3
    assert all(320 <= x < 960 and 180 <= y < 540 for x, y in segment["mask"])
    assert len(payload) <= postprocess.SEGMENTS_BYTE_BUDGET
