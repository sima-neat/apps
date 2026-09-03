"""Unit tests for the PatchCore host-side scoring stage (patchcore_scoring.py).

These exercise the memory-bank math directly against fixed, hand-constructed
embeddings -- no model, no hardware, no Neat runtime -- per the example's
acceptance criteria that the scoring stage has coverage independent of the
hardware-gated end-to-end path in test_e2e.py.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_SRC = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_SRC / "main.py"

sys.path.insert(0, str(PYTHON_SRC))

import patchcore_scoring as pcs  # noqa: E402


@pytest.mark.unit
class TestArgParsing:
    def test_help(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"], capture_output=True, text=True, timeout=20
        )
        assert r.returncode == 0
        assert "--config" in r.stdout
        assert "--calibrate" in r.stdout

    def test_bad_config_path(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/patchcore-config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"], capture_output=True, text=True, timeout=20
        )
        assert r.returncode == 2


@pytest.mark.unit
class TestExtractHwc:
    def test_hwc_passthrough(self):
        arr = np.random.default_rng(0).random((1, 4, 4, 8)).astype(np.float32)
        out = pcs.extract_hwc(arr, embed_dim=8)
        assert out.shape == (4, 4, 8)
        np.testing.assert_allclose(out, arr[0])

    def test_chw_transposed(self):
        rng = np.random.default_rng(1)
        chw = rng.random((1, 8, 4, 4)).astype(np.float32)
        out = pcs.extract_hwc(chw, embed_dim=8)
        assert out.shape == (4, 4, 8)
        np.testing.assert_allclose(out, np.transpose(chw[0], (1, 2, 0)))

    def test_unrecognized_channel_axis_raises(self):
        arr = np.zeros((1, 4, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            pcs.extract_hwc(arr, embed_dim=8)

    def test_wrong_ndim_raises(self):
        arr = np.zeros((1, 4, 8), dtype=np.float32)  # squeezes to 2D
        with pytest.raises(ValueError):
            pcs.extract_hwc(arr, embed_dim=8)


@pytest.mark.unit
class TestGreedyCoreset:
    def test_selects_requested_fraction(self):
        vectors = np.random.default_rng(2).random((200, 6)).astype(np.float32)
        idx = pcs.greedy_coreset_indices(vectors, ratio=0.1, seed=0)
        assert idx.shape[0] == 20
        assert len(set(idx.tolist())) == 20  # no duplicates

    def test_ratio_at_or_above_one_keeps_everything(self):
        vectors = np.random.default_rng(3).random((10, 3)).astype(np.float32)
        idx = pcs.greedy_coreset_indices(vectors, ratio=1.0, seed=0)
        assert sorted(idx.tolist()) == list(range(10))

    def test_deterministic_given_seed(self):
        vectors = np.random.default_rng(4).random((50, 5)).astype(np.float32)
        idx_a = pcs.greedy_coreset_indices(vectors, ratio=0.2, seed=42)
        idx_b = pcs.greedy_coreset_indices(vectors, ratio=0.2, seed=42)
        np.testing.assert_array_equal(idx_a, idx_b)


@pytest.mark.unit
class TestMemoryBankScore:
    def _bank_and_patches(self):
        # A 2D toy embedding space: a memory bank clustered near the origin, and one
        # test patch placed far away so its nearest-neighbor distance is unambiguous.
        bank_vectors = np.array(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1], [0.05, 0.05]],
            dtype=np.float32,
        )
        bank = pcs.MemoryBank(bank_vectors)
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


@pytest.mark.unit
class TestThresholdAndMeta:
    def test_percentile_threshold(self):
        scores = [float(x) for x in range(1, 101)]  # 1..100
        assert pcs.percentile_threshold(scores, 99.0) == pytest.approx(99.01, abs=0.5)

    def test_percentile_threshold_empty_raises(self):
        with pytest.raises(ValueError):
            pcs.percentile_threshold([], 99.0)

    def test_bank_meta_roundtrip(self, tmp_path):
        meta = pcs.build_bank_meta(
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
        )
        meta_path = tmp_path / "bank_meta.json"
        pcs.save_bank_meta(meta_path, meta)
        loaded = pcs.load_bank_meta(meta_path)
        assert loaded["model_sha256"] == pcs.sha256_file(__file__)
        assert loaded["bank_sha256"] == pcs.sha256_file(__file__)
        assert loaded["threshold"]["value"] == pytest.approx(12.3)

    def test_verify_bank_matches_model_ok(self, tmp_path):
        meta = {"model_sha256": pcs.sha256_file(__file__)}
        pcs.verify_bank_matches_model(meta, __file__)  # should not raise

    def test_verify_bank_matches_model_mismatch_raises(self):
        meta = {"model_sha256": "0" * 64}
        with pytest.raises(RuntimeError):
            pcs.verify_bank_matches_model(meta, __file__)

    def test_verify_bank_hash_ok(self):
        meta = {"bank_sha256": pcs.sha256_file(__file__)}
        pcs.verify_bank_hash(meta, __file__)  # should not raise

    def test_verify_bank_hash_mismatch_raises(self):
        meta = {"bank_sha256": "0" * 64}
        with pytest.raises(RuntimeError):
            pcs.verify_bank_hash(meta, __file__)

    def test_verify_bank_hash_skips_when_absent(self):
        pcs.verify_bank_hash({}, "/nonexistent/path.npy")  # should not raise
