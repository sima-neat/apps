"""Unit tests for efficientsam3-promptable-segmenter (Python)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"

sys.path.insert(0, str(MAIN_PY.parent))
_SPEC = importlib.util.spec_from_file_location("efficientsam3_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)

from clip_tokenizer import ClipTokenizer  # noqa: E402

# The C++ unit test asserts the same token ids and the same metadata.
TOKENS = {
    "person": [49406, 2533, 49407],
    "red car": [49406, 736, 1615, 49407],
    "it's a dog!": [49406, 585, 568, 320, 1929, 256, 49407],
    "café 2 cups": [49406, 15304, 273, 11463, 49407],
    "Traffic   Light": [49406, 3399, 1395, 49407],
    "a person wearing a yellow safety vest near the road":
        [49406, 320, 2533, 3309, 320, 4481, 3406, 12473, 2252, 518, 1759, 49407],
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


def config(**overrides):
    cfg = main.load_config(CONFIG)
    return main.Config(**{**cfg.__dict__, "prompt": "red car", **overrides})


@pytest.mark.unit
def test_packaged_config_loads():
    cfg = main.load_config(CONFIG)
    assert cfg.prompt == "person"
    assert (cfg.tcp, cfg.latency_ms, cfg.frames) == (True, 100, 0)
    assert (cfg.min_score, cfg.max_detections, cfg.mask_threshold) == (0.3, 20, 0.5)
    assert (cfg.video_port, cfg.metadata_port) == (9000, 9100)
    assert (cfg.save_dir, cfg.save_every) == ("", 0)


@pytest.mark.unit
@pytest.mark.parametrize("prompt", TOKENS)
def test_tokenizer_matches_clip(prompt):
    expected = TOKENS[prompt]
    assert ClipTokenizer().encode(prompt, 16).tolist() == expected + [0] * (16 - len(expected))


@pytest.mark.unit
def test_tokenizer_truncates_and_keeps_the_end_token():
    tokens = ClipTokenizer().encode(" ".join(["car"] * 20), 16)
    assert tokens[0] == 49406 and tokens[15] == 49407 and len(tokens) == 16


@pytest.mark.unit
def test_segments_match_the_cpp_implementation():
    detections, masks = frame_result()
    segments = main.segments_of(detections, masks, config(), 1920, 1080)
    assert json.dumps({"segments": segments}, separators=(",", ":")) == SEGMENTS_JSON


@pytest.mark.unit
def test_max_detections_keeps_the_best_scores():
    detections, masks = frame_result()
    segments = main.segments_of(detections, masks, config(max_detections=1), 1920, 1080)
    assert [s["id"] for s in segments] == ["seg_1"]


@pytest.mark.unit
def test_empty_mask_has_no_outline():
    empty = np.full((192, 192), -10.0, np.float32)
    assert main.mask_outline(empty, (100, 100, 300, 300), 1920, 1080, 0.0) == []


@pytest.mark.unit
def test_overlay_clock_sends_every_frame_the_nearest_result():
    clock = main.OverlayClock()
    for pts_ms, frame_id in [(0, 1), (33, 2), (66, 3), (100, 4)]:
        clock.add_frame(SimpleNamespace(pts_ns=pts_ms * 1_000_000, frame_id=frame_id))
    assert clock.add_result(33, "A") == [("A", 0, "1"), ("A", 33, "2")]
    for pts_ms, frame_id in [(133, 5), (166, 6)]:
        clock.add_frame(SimpleNamespace(pts_ns=pts_ms * 1_000_000, frame_id=frame_id))
    # Frame 66 is nearer the result at 33, frame 100 nearer the one at 133; frame 166 waits.
    assert clock.add_result(133, "B") == [("A", 66, "3"), ("B", 100, "4"), ("B", 133, "5")]


@pytest.mark.unit
def test_cli_rejects_unknown_flags():
    r = subprocess.run([sys.executable, str(MAIN_PY), "--bogus"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "unrecognized arguments" in r.stderr
