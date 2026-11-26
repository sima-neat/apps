"""Unit tests for the EfficientSAM3 promptable segmenter: the checks the application leaves to its
config, the CLIP tokenizer in utils/clip_tokenizer.py, and the host work in
utils/efficientsam3.py. tests/cpp/test_unit.cpp runs the same checks in C++.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]

main_module = load_example_main(EXAMPLE_DIR, "efficientsam3_promptable_segmenter_main")
efficientsam3 = main_module.efficientsam3
clip_tokenizer = main_module.clip_tokenizer

pytestmark = pytest.mark.unit

# The C++ unit test asserts the same token ids and the same metadata.
TOKENS = {
    "person": [49406, 2533, 49407],
    "red car": [49406, 736, 1615, 49407],
    "it's a dog!": [49406, 585, 568, 320, 1929, 256, 49407],
    "café 2 cups": [49406, 15304, 273, 11463, 49407],
    "Traffic   Light": [49406, 3399, 1395, 49407],
    "a person wearing a yellow safety vest near the road": [
        49406,
        320,
        2533,
        3309,
        320,
        4481,
        3406,
        12473,
        2252,
        518,
        1759,
        49407,
    ],
}
SEGMENTS_JSON = (
    '{"segments":[{"id":"seg_1","label":"red car","confidence":0.9123,"bbox":[390,276,420,180],'
    '"mask_format":"polygon","mask":[[400,284],[403,449],[799,447],[795,281]]},'
    '{"id":"seg_3","label":"red car","confidence":0.4001,"bbox":[1809,1071,111,9],'
    '"mask_format":"polygon","mask":[[1800,1063],[1800,1079],[1919,1079],[1919,1063]]}]}'
)


def frame_result():
    """Detections and mask logits of one 1920x1080 frame, as the model returns them."""
    detections = np.zeros((200, 6), np.float32)
    masks = np.full((192, 192, 200), -10.0, np.float32)
    masks[50:80, 40:80, 7] = 10.0
    detections[7] = [205.0, 258.0, 425.0, 425.0, 0.91234, 0]
    masks[10:20, 10:20, 3] = 10.0
    detections[3] = [10, 10, 100, 100, 0.25, 0]  # below min_score
    detections[12] = [500, 500, 600, 600, 0.5, 0]  # empty mask: skipped, still numbered
    masks[180:192, 180:192, 30] = 10.0
    detections[30] = [950.0, 1000.0, 1010.0, 1015.0, 0.40005, 0]  # box past the frame edge
    return detections, masks


def options(**overrides) -> efficientsam3.SegmentOptions:
    values = {
        "width": 1920,
        "height": 1080,
        "label": "red car",
        "min_score": 0.3,
        "max_detections": 20,
        "mask_threshold": 0.5,
    }
    return efficientsam3.SegmentOptions(**{**values, **overrides})


def nv12_frame(width: int, height: int, left_y: int, right_y: int) -> np.ndarray:
    """A gray NV12 frame whose left half has luma `left_y` and right half `right_y`."""
    frame = np.full((height * 3 // 2, width), 128, np.uint8)
    frame[:height, : width // 2] = left_y
    frame[:height, width // 2 :] = right_y
    return frame


def test_shipped_config_names_the_model_and_usable_values():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "efficientsam3_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models, where the README downloads both packages.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.source").startswith("rtsp://")
    # The shipped prompt fits the text encoder.
    tokenizer = clip_tokenizer.ClipTokenizer()
    clip_tokenizer.prompt_tokens(tokenizer, config.string("app.prompt"), main_module.TEXT_TOKENS)
    for key in (
        "input.width",
        "input.height",
        "input.fps",
        "input.rtsp.latency_ms",
        "inference.frames",
    ):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    # The threshold becomes a logit, so 0 and 1 are out of range.
    assert 0 < config.number("inference.mask_threshold") < 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


@pytest.mark.parametrize("prompt", TOKENS)
def test_tokenizer_matches_clip(prompt):
    expected = TOKENS[prompt]
    tokens = clip_tokenizer.ClipTokenizer().encode(prompt, 16).tolist()
    assert tokens == expected + [0] * (16 - len(expected))


def test_tokenizer_truncates_and_keeps_the_end_token():
    tokens = clip_tokenizer.ClipTokenizer().encode(" ".join(["car"] * 20), 16)
    assert tokens[0] == 49406 and tokens[15] == 49407 and len(tokens) == 16


@pytest.mark.parametrize("prompt", ["", "   "])
def test_prompt_must_name_something(prompt):
    with pytest.raises(ValueError, match="app.prompt must name what to segment"):
        clip_tokenizer.prompt_tokens(clip_tokenizer.ClipTokenizer(), prompt, 16)


def test_prompt_longer_than_the_text_encoder_is_rejected():
    tokenizer = clip_tokenizer.ClipTokenizer()
    fits = " ".join(["car"] * 14)
    assert clip_tokenizer.prompt_tokens(tokenizer, fits, 16).tolist()[15] == 49407
    with pytest.raises(
        ValueError, match="app.prompt is 15 tokens; the text encoder reads at most 14"
    ):
        clip_tokenizer.prompt_tokens(tokenizer, fits + " bus", 16)


@pytest.mark.parametrize(("width", "height"), [(1920, 1080), (640, 480)], ids=["shrink", "grow"])
def test_preprocess_makes_rgb_from_minus_one_to_one(width, height):
    """Black on the left (luma 16) and white on the right (luma 235) become -1 and 1, at the
    model's size, whether the frame shrinks or grows to it."""
    out = np.empty((efficientsam3.MODEL_SIZE, efficientsam3.MODEL_SIZE, 3), np.float32)
    efficientsam3.preprocess(nv12_frame(width, height, 16, 235), out)
    half = efficientsam3.MODEL_SIZE // 2
    np.testing.assert_allclose(out[:, : half - 8], -1.0, atol=1e-6)
    np.testing.assert_allclose(out[:, half + 8 :], 1.0, atol=1e-6)


def test_segments_and_metadata_of_one_frame():
    detections, masks = frame_result()
    segments = efficientsam3.segments_of(detections, masks, options())
    assert efficientsam3.segments_json(segments) == SEGMENTS_JSON


def test_segments_accept_outputs_with_a_batch_axis():
    detections, masks = frame_result()
    segments = efficientsam3.segments_of(detections[None], masks[None], options())
    assert efficientsam3.segments_json(segments) == SEGMENTS_JSON


def test_metadata_keeps_the_best_segments_within_the_byte_budget():
    detections, masks = frame_result()
    segments = efficientsam3.segments_of(detections, masks, options(label="x" * 20000))
    payload = efficientsam3.segments_json(segments)
    assert len(payload) <= efficientsam3.METADATA_BYTE_BUDGET
    assert [s["id"] for s in json.loads(payload)["segments"]] == ["seg_1"]


def test_max_detections_keeps_the_best_scores():
    detections, masks = frame_result()
    segments = efficientsam3.segments_of(detections, masks, options(max_detections=1))
    assert [s["id"] for s in segments] == ["seg_1"]


def test_min_score_drops_lower_scores():
    detections, masks = frame_result()
    segments = efficientsam3.segments_of(detections, masks, options(min_score=0.45))
    assert [s["id"] for s in segments] == ["seg_1"]


def test_duplicate_instances_are_dropped():
    detections, masks = frame_result()
    detections[40] = [210.0, 262.0, 420.0, 400.0, 0.45, 0]  # most of instance 7's box, lower
    masks[50:75, 40:80, 40] = 10.0
    segments = efficientsam3.segments_of(detections, masks, options())
    assert [s["id"] for s in segments] == ["seg_1", "seg_4"]
    # The empty and duplicate instances do not use up max_detections.
    segments = efficientsam3.segments_of(detections, masks, options(max_detections=2))
    assert [s["id"] for s in segments] == ["seg_1", "seg_4"]


def test_empty_mask_has_no_outline():
    empty = np.full((192, 192), -10.0, np.float32)
    assert efficientsam3.mask_outline(empty, (100, 100, 300, 300), 1920, 1080, 0.0) == []


def test_overlay_clock_sends_every_frame_the_nearest_result():
    clock = efficientsam3.OverlayClock()
    for pts_ms, frame_id in [(0, "1"), (33, "2"), (66, "3"), (100, "4")]:
        clock.add_frame(pts_ms, frame_id)
    assert clock.add_result(33, "A") == [("A", 0, "1"), ("A", 33, "2")]
    for pts_ms, frame_id in [(133, "5"), (166, "6")]:
        clock.add_frame(pts_ms, frame_id)
    # Frame 66 is nearer the result at 33, frame 100 nearer the one at 133; frame 166 waits.
    assert clock.add_result(133, "B") == [("A", 66, "3"), ("B", 100, "4"), ("B", 133, "5")]
