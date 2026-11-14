"""Unit tests for edgetam-video-object-segmenter (Python): the same tracking-step and config checks as
tests/cpp/test_unit.cpp."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"

sys.path.insert(0, str(MAIN_PY.parent))
_SPEC = importlib.util.spec_from_file_location("edgetam_video_object_segmenter_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)

from utils import tracker  # noqa: E402

pytestmark = pytest.mark.unit


def decoder_outputs(iou_scores: list[float], object_score: float) -> list[np.ndarray]:
    """Mask decoder outputs whose mask k is k + 1 everywhere and whose pointer k is 10 * (k + 1) everywhere."""
    masks = np.broadcast_to(np.arange(1, tracker.MASKS + 1, dtype=np.float32),
                            (1, tracker.MASK_SIZE, tracker.MASK_SIZE, tracker.MASKS))
    pointers = np.repeat(10 * np.arange(1, tracker.MASKS + 1, dtype=np.float32), tracker.POINTER_DIM)
    return [masks, np.asarray([iou_scores], np.float32), np.asarray([[object_score]], np.float32),
            pointers.reshape(1, tracker.MASKS, tracker.POINTER_DIM)]


def write_config(tmp_path: Path, section: str, key: str, value) -> Path:
    """The shipped config with its four model paths pointing at files that exist and one value replaced."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for name in raw["model"]:
        raw["model"][name] = str(tmp_path / f"{name}.tar.gz")
        Path(raw["model"][name]).touch()
    raw[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_prompt_point_is_a_foreground_pixel_centre():
    coords, types = tracker.prompt_arrays((0.25, 0.75))
    assert coords.shape == (1, 1, 3, 2) and types.shape == (1, 1, 3, 5)
    # (0.25 * 1024 + 0.5) / 512 - 1 and (0.75 * 1024 + 0.5) / 512 - 1.
    np.testing.assert_allclose(coords[0, 0, 0], [-0.4990234375, 0.5009765625])
    assert types[0, 0].tolist() == [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    assert not coords[0, 0, 1:].any()


def test_tracked_frame_prompt_is_all_padding():
    coords, types = tracker.prompt_arrays(None)
    assert not coords.any()
    assert types[0, 0].tolist() == [[1, 0, 0, 0, 0]] * 3


def test_choose_mask_keeps_the_best_of_masks_1_to_3():
    # Mask 0 has the highest score and must still lose: the model's authors use it only for a box.
    mask, pointer = tracker.choose_mask(*decoder_outputs([0.9, 0.2, 0.7, 0.4], object_score=1.0))
    assert mask.shape == (256, 256) and (mask == 3).all()
    assert pointer.shape == (256,) and (pointer == 30).all()


def test_choose_mask_reports_no_object_when_the_object_score_is_not_positive():
    mask, pointer = tracker.choose_mask(*decoder_outputs([0.1, 0.8, 0.3, 0.2], object_score=0.0))
    assert (mask == tracker.NO_OBJECT_LOGIT).all()
    assert (pointer == 20).all()


def test_memory_mask_is_thresholded_on_the_prompted_frame_and_a_probability_after():
    mask = np.full((256, 256), -2.0, np.float32)
    mask[:, 128:] = 2.0
    prompted = tracker.memory_mask(mask, prompted=True)
    assert prompted.shape == (1, 1024, 1024, 1)
    assert (prompted[0, :, :500] == 0).all() and (prompted[0, :, 524:] == 1).all()
    tracked = tracker.memory_mask(mask, prompted=False)
    np.testing.assert_allclose(tracked[0, 0, 0, 0], 1 / (1 + np.exp(2)), rtol=1e-5)
    np.testing.assert_allclose(tracked[0, 0, -1, 0], 1 / (1 + np.exp(-2)), rtol=1e-5)
    absent = tracker.memory_mask(np.full((256, 256), tracker.NO_OBJECT_LOGIT, np.float32), prompted=False)
    assert (absent == 0).all()


def test_locate_object_gives_the_area_and_centre_in_frame_fractions():
    mask = np.full((256, 256), -1.0, np.float32)
    mask[32:96, 128:256] = 1.0
    assert tracker.locate_object(mask) == tracker.ObjectLocation(True, 0.125, 0.75, 0.25)
    assert tracker.locate_object(np.full((256, 256), -1.0, np.float32)) == tracker.ObjectLocation(
        False, 0.0, None, None)


def test_memory_bank_keeps_the_prompted_frame_then_the_latest_newest_first():
    bank = tracker.MemoryBank()
    for frame in range(20):
        bank.add(np.full((1, 512, 64), frame, np.float32), np.full(256, frame, np.float32))
        memories, used_memories, pointers, used_pointers = bank.arrays()
        assert memories.shape == (1, 3584, 64) and pointers.shape == (1, 16, 256)
        assert used_memories.sum() == min(frame + 1, 7) and used_pointers.sum() == min(frame + 1, 16)
    # Frame 0 is the prompted frame; 19 is the newest. Each row holds its frame number.
    assert memories[0, ::512, 0].tolist() == [0, 19, 18, 17, 16, 15, 14]
    assert pointers[0, :, 0].tolist() == [0, *range(19, 4, -1)]


def test_memory_bank_leaves_unused_slots_zero():
    bank = tracker.MemoryBank()
    bank.add(np.ones((512, 64), np.float32), np.ones(256, np.float32))
    memories, used_memories, pointers, used_pointers = bank.arrays()
    assert used_memories.tolist() == [[1, 0, 0, 0, 0, 0, 0]] and used_pointers[0, 1:].sum() == 0
    assert memories[0, :512].all() and not memories[0, 512:].any()
    assert pointers[0, 0].all() and not pointers[0, 1:].any()


def test_shipped_config_loads(tmp_path):
    cfg = main.load_config(write_config(tmp_path, "inference", "frames", 150))
    assert (cfg.prompt_x, cfg.prompt_y, cfg.frames, cfg.save_every) == (0.87, 0.38, 150, 10)
    assert cfg.video_path == "assets/datasets/tum-rgbd/freiburg1-desk.mp4"
    assert cfg.output_dir == Path("sandbox/edgetam-video-object-segmenter")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("prompt", "x", 1.2, "prompt.x 1.2 and prompt.y 0.38 must be fractions of the frame, from 0 to 1"),
        ("prompt", "y", -0.1, "prompt.x 0.87 and prompt.y -0.1 must be fractions of the frame, from 0 to 1"),
        ("inference", "frames", -1, "inference.frames must be 0 or more"),
        ("output", "save_every", -1, "output.save_every must be 0 or more"),
        ("model", "mask_decoder", "missing.tar.gz", "model.mask_decoder missing.tar.gz does not exist"),
    ],
)
def test_config_errors_name_the_setting(tmp_path, section, key, value, message):
    with pytest.raises(ValueError, match=message):
        main.load_config(write_config(tmp_path, section, key, value))


def test_unreadable_video_is_reported(tmp_path):
    with pytest.raises(ValueError, match="could not open source.video_path"):
        main.open_video(str(tmp_path / "missing.mp4"))


def test_help_describes_the_config_option():
    result = subprocess.run([sys.executable, str(MAIN_PY), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--config" in result.stdout
