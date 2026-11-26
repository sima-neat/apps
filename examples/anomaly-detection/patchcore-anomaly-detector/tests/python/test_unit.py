"""Unit tests for the PatchCore anomaly detector: the checks the application leaves to its config,
and the host-side memory bank on hand-made patch features. tests/cpp/test_unit.cpp runs the same
config checks in C++.
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


@pytest.mark.unit
class TestExtractHwc:
    def test_hwc_passthrough(self):
        arr = np.random.default_rng(0).random((1, 4, 4, 8)).astype(np.float32)
        out = memory_bank.extract_hwc(arr, embed_dim=8)
        assert out.shape == (4, 4, 8)
        np.testing.assert_allclose(out, arr[0])

    def test_chw_transposed(self):
        rng = np.random.default_rng(1)
        chw = rng.random((1, 8, 4, 4)).astype(np.float32)
        out = memory_bank.extract_hwc(chw, embed_dim=8)
        assert out.shape == (4, 4, 8)
        np.testing.assert_allclose(out, np.transpose(chw[0], (1, 2, 0)))

    def test_unrecognized_channel_axis_raises(self):
        arr = np.zeros((1, 4, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            memory_bank.extract_hwc(arr, embed_dim=8)

    def test_wrong_ndim_raises(self):
        arr = np.zeros((1, 4, 8), dtype=np.float32)  # squeezes to 2D
        with pytest.raises(ValueError):
            memory_bank.extract_hwc(arr, embed_dim=8)


@pytest.mark.unit
class TestGreedyCoreset:
    def test_selects_requested_fraction(self):
        vectors = np.random.default_rng(2).random((200, 6)).astype(np.float32)
        idx = memory_bank.greedy_coreset_indices(vectors, ratio=0.1, seed=0)
        assert idx.shape[0] == 20
        assert len(set(idx.tolist())) == 20  # no duplicates

    def test_ratio_at_or_above_one_keeps_everything(self):
        vectors = np.random.default_rng(3).random((10, 3)).astype(np.float32)
        idx = memory_bank.greedy_coreset_indices(vectors, ratio=1.0, seed=0)
        assert sorted(idx.tolist()) == list(range(10))

    def test_deterministic_given_seed(self):
        vectors = np.random.default_rng(4).random((50, 5)).astype(np.float32)
        idx_a = memory_bank.greedy_coreset_indices(vectors, ratio=0.2, seed=42)
        idx_b = memory_bank.greedy_coreset_indices(vectors, ratio=0.2, seed=42)
        np.testing.assert_array_equal(idx_a, idx_b)

    def test_first_pick_is_seed_mod_n(self):
        """The shared start rule both languages use, so they build the same bank."""
        vectors = np.random.default_rng(5).random((50, 5)).astype(np.float32)
        for seed in (0, 7, 123):
            assert memory_bank.greedy_coreset_indices(vectors, ratio=0.2, seed=seed)[0] == seed % 50


@pytest.mark.unit
class TestMemoryBankScore:
    def _bank_and_patches(self):
        # A 2D toy embedding space: a memory bank clustered near the origin, and one
        # test patch placed far away so its nearest-neighbor distance is unambiguous.
        bank_vectors = np.array(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1], [0.05, 0.05]],
            dtype=np.float32,
        )
        bank = memory_bank.MemoryBank(bank_vectors)
        # One normal-looking patch (near the bank) and one clearly anomalous patch (far away).
        patches = np.array(
            [[[0.02, -0.01], [10.0, 10.0]]],
            dtype=np.float32,
        ).reshape(1, 2, 2)
        return bank, patches

    def test_score_map_matches_manual_nearest_neighbor(self):
        bank, patches = self._bank_and_patches()
        scored = bank.score(patches, num_neighbors=1)
        expected_near = float(np.min(np.linalg.norm(bank.vectors - patches[0, 0], axis=1)))
        expected_far = float(np.min(np.linalg.norm(bank.vectors - patches[0, 1], axis=1)))
        np.testing.assert_allclose(scored.score_map[0, 0], expected_near, atol=1e-5)
        np.testing.assert_allclose(scored.score_map[0, 1], expected_far, atol=1e-5)

    def test_num_neighbors_one_is_plain_max(self):
        bank, patches = self._bank_and_patches()
        scored = bank.score(patches, num_neighbors=1)
        assert scored.image_score == pytest.approx(float(scored.score_map.max()))

    def test_reweighted_score_is_between_zero_and_plain_max(self):
        bank, patches = self._bank_and_patches()
        plain = bank.score(patches, num_neighbors=1).image_score
        reweighted = bank.score(patches, num_neighbors=4).image_score
        # weight = 1 - softmax(...)[0] is in [0, 1), so the reweighted score can only
        # shrink the plain max-distance score, never exceed or invert its sign.
        assert 0.0 <= reweighted <= plain + 1e-6

    def test_num_neighbors_clamped_to_bank_size(self):
        bank, patches = self._bank_and_patches()
        # Requesting more neighbors than the bank holds should not raise.
        scored = bank.score(patches, num_neighbors=1000)
        assert scored.image_score >= 0.0

    def test_dimension_mismatch_raises(self):
        bank, _ = self._bank_and_patches()
        bad_patches = np.zeros((1, 2, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            bank.score(bad_patches, num_neighbors=1)

    def test_identical_bank_and_query_score_near_zero(self):
        """A patch identical to its one bank vector must score ~0, at the
        real embedding dimension (1536) where cancellation error can hide."""
        rng = np.random.default_rng(0)
        vector = rng.standard_normal(1536).astype(np.float32)
        bank = memory_bank.MemoryBank(vector[None, :])
        patch = vector.reshape(1, 1, 1536)
        scored = bank.score(patch, num_neighbors=1)
        assert scored.score_map[0, 0] < 1e-3


@pytest.mark.unit
class TestThresholdAndMeta:
    def test_percentile_threshold(self):
        scores = [float(x) for x in range(1, 101)]  # 1..100
        assert memory_bank.percentile_threshold(scores, 99.0) == pytest.approx(99.01, abs=0.5)

    def test_percentile_threshold_empty_raises(self):
        with pytest.raises(ValueError):
            memory_bank.percentile_threshold([], 99.0)

    def test_bank_meta_roundtrip(self, tmp_path):
        meta = memory_bank.build_bank_meta(
            model_path=__file__,  # any real, stable file to hash
            bank_path=__file__,  # any real, stable file to hash
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
            threshold=12.3,
            threshold_percentile=99.0,
            threshold_num_images=10,
            patch_threshold=8.1,
        )
        meta_path = tmp_path / "bank_meta.json"
        memory_bank.save_bank_meta(meta_path, meta)
        loaded = memory_bank.load_bank_meta(meta_path)
        assert loaded["model_sha256"] == memory_bank.sha256_file(__file__)
        assert loaded["bank_sha256"] == memory_bank.sha256_file(__file__)
        assert loaded["threshold"]["value"] == pytest.approx(12.3)
        assert loaded["patch_threshold"]["value"] == pytest.approx(8.1)

    def test_verify_bank_matches_model_ok(self, tmp_path):
        meta = {"model_sha256": memory_bank.sha256_file(__file__)}
        memory_bank.verify_bank_matches_model(meta, __file__)  # should not raise

    def test_verify_bank_matches_model_mismatch_raises(self):
        meta = {"model_sha256": "0" * 64}
        with pytest.raises(RuntimeError):
            memory_bank.verify_bank_matches_model(meta, __file__)

    def test_verify_bank_hash_ok(self):
        meta = {"bank_sha256": memory_bank.sha256_file(__file__)}
        memory_bank.verify_bank_hash(meta, __file__)  # should not raise

    def test_verify_bank_hash_mismatch_raises(self):
        meta = {"bank_sha256": "0" * 64}
        with pytest.raises(RuntimeError):
            memory_bank.verify_bank_hash(meta, __file__)

    def test_verify_bank_hash_skips_when_absent(self):
        memory_bank.verify_bank_hash({}, "/nonexistent/path.npy")  # should not raise
