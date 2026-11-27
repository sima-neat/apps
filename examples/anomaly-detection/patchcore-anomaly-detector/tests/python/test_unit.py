"""Unit tests for the PatchCore anomaly detector: the checks the application leaves to its config,
and the host-side memory bank on hand-made patch features. tests/cpp/test_unit.cpp runs the same
checks in C++.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
APPS_ROOT = EXAMPLE_DIR.parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "patchcore_anomaly_detector_main")
memory_bank = main_module.memory_bank

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_model_and_input():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert config.string("input.type") in main_module.INPUT_TYPES
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "patchcore_wide_resnet50_2_bf16_mla.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(APPS_ROOT / "models")
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames"):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def test_shipped_calibration_settings_are_in_range():
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert 0.0 < config.number("app.coreset_ratio") <= 1.0
    assert 0.0 <= config.number("app.threshold_percentile") <= 100.0
    assert config.number("app.threshold_margin") >= 1.0
    nominal = APPS_ROOT / config.string("app.nominal_images_dir")
    assert len(main_module.image_paths(nominal)) >= main_module.THRESHOLD_EVERY
    # Outlined images written among the nominal images would be calibrated on next time.
    bank_dir = Path(config.string("app.memory_bank_dir")).resolve()
    output_dir = Path(config.string("output.dir")).resolve()
    assert nominal.resolve() not in (bank_dir, output_dir)


def test_anomaly_map_covers_the_frame_and_keeps_the_peak_in_place():
    scores = np.zeros((28, 28), np.float32)
    scores[0, 27] = 10.0
    result = memory_bank.anomaly_map(scores, 224, 4.0)
    assert result.shape == (224, 224) and result.dtype == np.float32
    row, col = np.unravel_index(int(np.argmax(result)), result.shape)
    assert row < 16 and col > 208


def _one_hot_patches() -> np.ndarray:
    """A 2x2 grid of 4-channel patches, each a different unit vector."""
    return np.eye(4, dtype=np.float32).reshape(2, 2, 4)


def test_identical_patches_score_zero_and_a_far_one_scores_high():
    normal = _one_hot_patches()
    bank = memory_bank.MemoryBank.build([normal], 1.0, 0)
    assert bank.size == 4, "a ratio of 1 keeps every patch"
    assert bank.score(normal, 1).image_score < 1e-5
    defect = normal.copy()
    defect[0, 0, 0] = 5.0
    far = bank.score(defect, 1)
    assert far.image_score == pytest.approx(4.0, abs=1e-4)
    assert far.score_map[0, 0] == far.image_score


def test_reweighting_never_raises_the_plain_max_score():
    normal = _one_hot_patches()
    bank = memory_bank.MemoryBank.build([normal], 1.0, 0)
    defect = normal.copy()
    defect[0, 0, 0] = 5.0
    reweighted = bank.score(defect, 4).image_score
    assert 0.0 < reweighted < bank.score(defect, 1).image_score


def test_coreset_keeps_the_ratio_starting_at_the_seed_patch():
    """Ten points on a line: the coreset starts at patch seed % n, then takes the farthest."""
    line = np.arange(10, dtype=np.float32).reshape(10, 1)
    assert memory_bank.greedy_coreset_indices(line, 0.3, 13).tolist() == [3, 9, 0]
    assert sorted(memory_bank.greedy_coreset_indices(line, 1.0, 0).tolist()) == list(range(10))


def test_percentile_threshold_interpolates_like_numpy():
    scores = [float(value) for value in range(1, 101)]
    assert memory_bank.percentile_threshold(scores, 99.0) == pytest.approx(99.01, abs=1e-4)


def test_extract_hwc_transposes_channel_first_output():
    chw = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    hwc = memory_bank.extract_hwc(chw, embed_dim=3)
    assert hwc.shape == (2, 2, 3)
    assert hwc[0, 0].tolist() == [0.0, 4.0, 8.0] and hwc[0, 1].tolist() == [1.0, 5.0, 9.0]


def test_bank_meta_round_trips(tmp_path):
    files = {}
    for name in ("model.tar.gz", "memory_bank.npy"):
        files[name] = tmp_path / name
        files[name].write_bytes(name.encode())
    meta = memory_bank.build_bank_meta(
        model_path=files["model.tar.gz"],
        bank_path=files["memory_bank.npy"],
        backbone="wide_resnet50_2",
        torchvision_weights="IMAGENET1K_V1",
        embed_dim=1536,
        patch_grid=(28, 28),
        coreset_ratio=0.01,
        seed=0,
        num_nominal_images=10,
        bank_size=42,
        num_neighbors=9,
        gaussian_sigma=4.0,
        threshold=12.5,
        threshold_percentile=99.0,
        threshold_num_images=10,
        patch_threshold=8.25,
    )
    memory_bank.save_bank_meta(tmp_path / "bank_meta.json", meta)
    loaded = memory_bank.load_bank_meta(tmp_path / "bank_meta.json")
    assert loaded["model_sha256"] == memory_bank.sha256_file(files["model.tar.gz"])
    assert loaded["threshold"]["value"] == 12.5
    assert loaded["patch_threshold"]["value"] == 8.25


def test_a_bank_for_another_model_or_bank_file_is_rejected(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.write_bytes(b"model")
    bank = tmp_path / "memory_bank.npy"
    bank.write_bytes(b"bank")
    meta = {"model_sha256": memory_bank.sha256_file(model), "bank_sha256": memory_bank.sha256_file(bank)}
    memory_bank.verify_bank_matches_model(meta, model)
    memory_bank.verify_bank_hash(meta, bank)
    with pytest.raises(RuntimeError, match="different model package"):
        memory_bank.verify_bank_matches_model(meta, bank)
    with pytest.raises(RuntimeError, match="does not match"):
        memory_bank.verify_bank_hash(meta, model)
    # A bank_meta.json without bank_sha256 skips the bank check.
    memory_bank.verify_bank_hash({}, tmp_path / "missing.npy")
