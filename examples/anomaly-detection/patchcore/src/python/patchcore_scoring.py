"""Host-side PatchCore memory bank: coreset build, calibration, and anomaly scoring.

The compiled MLA graph only extracts per-patch feature embeddings (see the
`wide_resnet50_2` layer2+layer3 tap in main.py's model options). The anomaly
decision itself -- nearest-neighbor lookup against a coreset of "normal"
reference patches, the image-level score, and the pass/fail threshold -- is
non-parametric and has no place in a compiled graph, so it lives here as plain
host-side numpy, following Roth et al., "Towards Total Recall in Industrial
Anomaly Detection" (CVPR 2022).

Two on-disk artifacts travel together as a versioned pair:
  - `memory_bank.npy`: the coreset, a float32 (N, embed_dim) array.
  - `bank_meta.json`: the model package hash the bank was built against, the
    coreset ratio and nominal-image count used, and the decision threshold
    with the percentile and image count it was derived from.

`bank_meta.json` pinning the model hash means a bank built for one compiled
model package fails loudly at load time if pointed at a different package,
instead of silently producing meaningless scores.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# WideResNet-50 layer2 (512ch) + upsampled layer3 (1024ch) concatenated patch embedding.
EMBED_DIM = 1536


def sha256_file(path: str | Path) -> str:
    """Hex sha256 digest of a file's contents, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_hwc(embedding: np.ndarray, embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Normalizes a raw model output array to (H, W, C).

    The MLA compiles tensors NHWC-native, but the channel axis is detected at
    runtime instead of assumed, so this also tolerates an NCHW-shaped output.
    """
    arr = np.squeeze(embedding)
    if arr.ndim != 3:
        raise ValueError(
            f"expected a 3D patch-embedding array after squeezing the batch dim, got shape {arr.shape}"
        )
    if arr.shape[-1] == embed_dim:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[0] == embed_dim:
        return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)), dtype=np.float32)
    raise ValueError(f"could not find the {embed_dim}-channel axis in output shape {arr.shape}")


def _pairwise_l2(query: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray | None = None) -> np.ndarray:
    """query: (N, C), bank: (M, C) -> (N, M) L2 distances, via the expand-and-dot
    identity (no scipy/faiss dependency). `bank_sq` is `sum(bank*bank, axis=1)`;
    pass it precomputed when scoring many queries against the same bank, since
    it never changes between calls."""
    q_sq = np.sum(query * query, axis=1, keepdims=True)  # (N, 1)
    if bank_sq is None:
        bank_sq = np.sum(bank * bank, axis=1)
    cross = query @ bank.T  # (N, M)
    sq_dist = np.clip(q_sq + bank_sq[None, :] - 2.0 * cross, a_min=0.0, a_max=None)
    return np.sqrt(sq_dist)


def greedy_coreset_indices(vectors: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Greedy k-center (farthest-point) coreset selection, matching the PatchCore
    paper's subsampling strategy: pick a random start, then repeatedly add whichever
    remaining point is farthest (by L2) from every point already selected.

    This runs directly in the full `embed_dim`-dimensional space rather than the
    paper's random low-dimensional (Johnson-Lindenstrauss) projection, trading
    build-time speed for simplicity -- a deliberate simplification worth knowing
    about if you port this toward the paper's reported build times on very large
    nominal sets. It is O(k * n) distance computations, which is fine for the
    nominal-set sizes a single inspection category calibration run collects, but
    is the dominant cost of `--calibrate` for large sets.

    Returns the selected row indices into `vectors`, not the vectors themselves.
    """
    n = vectors.shape[0]
    k = max(1, int(round(n * ratio)))
    if k >= n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(rng.integers(0, n))
    min_dist = np.linalg.norm(vectors - vectors[selected[0]], axis=1)
    for i in range(1, k):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        new_dist = np.linalg.norm(vectors - vectors[next_idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return selected


@dataclass
class ScoredImage:
    score_map: np.ndarray  # (H, W) float32, patch-grid resolution nearest-neighbor distances
    image_score: float  # PatchCore-reweighted image-level anomaly score


class MemoryBank:
    """Coreset of "normal" patch-feature vectors, plus nearest-neighbor anomaly scoring."""

    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2:
            raise ValueError(f"memory bank must be a 2D (N, embed_dim) array, got shape {vectors.shape}")
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # Precomputed once per bank rather than on every score() call.
        self._vectors_sq = np.sum(self.vectors * self.vectors, axis=1)

    @property
    def size(self) -> int:
        return self.vectors.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.vectors.shape[1]

    @classmethod
    def build(cls, per_image_embeddings: list[np.ndarray], coreset_ratio: float, seed: int) -> "MemoryBank":
        """Pools every patch from every reference image, then greedily subsamples
        the pool down to `coreset_ratio` of its size."""
        if not per_image_embeddings:
            raise ValueError("no reference ('normal') images given to build the memory bank")
        pooled = np.concatenate([img.reshape(-1, img.shape[-1]) for img in per_image_embeddings], axis=0)
        pooled = pooled.astype(np.float32)
        idx = greedy_coreset_indices(pooled, coreset_ratio, seed)
        return cls(pooled[idx])

    def save(self, path: str | Path) -> None:
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryBank":
        return cls(np.load(path))

    def score(self, patch_embeddings: np.ndarray, num_neighbors: int) -> ScoredImage:
        """patch_embeddings: (H, W, C). Returns the per-patch nearest-neighbor score
        map and the PatchCore-reweighted image-level score.

        Reweighting follows the paper's Eq. 7-8 (and anomalib's reference
        implementation): downweight the image score when the memory-bank
        neighborhood around the best-matching reference patch is itself diverse
        (a well-covered, typical normal region), and leave it near full weight
        when that match is comparatively isolated in the bank.
        """
        h, w, c = patch_embeddings.shape
        if c != self.embed_dim:
            raise ValueError(f"patch embedding dim {c} does not match memory bank dim {self.embed_dim}")
        flat = patch_embeddings.reshape(-1, c)

        dists = _pairwise_l2(flat, self.vectors, self._vectors_sq)  # (num_patches, bank_size)
        locations = np.argmin(dists, axis=1)  # nearest bank index per patch, m^* index
        patch_scores = dists[np.arange(dists.shape[0]), locations]  # s per patch
        score_map = patch_scores.reshape(h, w)

        k = min(num_neighbors, self.size)
        if k <= 1:
            return ScoredImage(score_map=score_map, image_score=float(patch_scores.max()))

        max_patch = int(np.argmax(patch_scores))
        q_star = flat[max_patch]  # m^test,*
        s_star = float(patch_scores[max_patch])
        nn_index = int(locations[max_patch])  # index of m^*
        m_star = self.vectors[nn_index]

        # N_b(m^*): the k nearest bank neighbors of m^* itself (support_idx[0] == nn_index,
        # since m^* is trivially its own closest neighbor at distance 0).
        dist_to_m_star = _pairwise_l2(m_star[None, :], self.vectors, self._vectors_sq)[0]
        support_idx = np.argpartition(dist_to_m_star, k - 1)[:k]
        support_idx = support_idx[np.argsort(dist_to_m_star[support_idx])]

        # Distance from the test patch (not m^*) to each support sample.
        support_dists = np.linalg.norm(self.vectors[support_idx] - q_star[None, :], axis=1)
        exp = np.exp(support_dists - support_dists.max())
        softmax = exp / exp.sum()
        weight = 1.0 - float(softmax[0])

        return ScoredImage(score_map=score_map, image_score=weight * s_star)


def upsample_and_smooth(score_map: np.ndarray, out_size: tuple[int, int], sigma: float) -> np.ndarray:
    """Upsamples a patch-grid score map to `out_size` (width, height) pixels and
    applies Gaussian smoothing, matching the PatchCore anomaly-map post-process."""
    upsampled = cv2.resize(score_map, out_size, interpolation=cv2.INTER_LINEAR)
    if sigma <= 0:
        return upsampled
    return cv2.GaussianBlur(upsampled, ksize=(0, 0), sigmaX=sigma)


def percentile_threshold(scores: list[float], percentile: float) -> float:
    if not scores:
        raise ValueError("no scores given to derive a threshold from")
    return float(np.percentile(np.asarray(scores, dtype=np.float64), percentile))


def build_bank_meta(
    *,
    model_path: str | Path,
    bank_path: str | Path,
    backbone: str,
    torchvision_weights: str,
    embed_dim: int,
    patch_grid: tuple[int, int],
    coreset_ratio: float,
    seed: int,
    num_nominal_images: int,
    bank_size: int,
    num_neighbors: int,
    gaussian_sigma: float,
    threshold: float,
    threshold_percentile: float,
    threshold_num_images: int,
) -> dict:
    return {
        "model_sha256": sha256_file(model_path),
        "model_filename": Path(model_path).name,
        # Pins bank_meta.json to the exact memory_bank.npy it was derived from --
        # the threshold above is only valid for the score distribution that
        # specific bank produces, so a bank swapped in from a different
        # calibration run (same model, different coreset) must not be scored
        # against this threshold silently.
        "bank_sha256": sha256_file(bank_path),
        "backbone": backbone,
        "torchvision_weights": torchvision_weights,
        "embed_dim": embed_dim,
        "patch_grid": list(patch_grid),
        "coreset_ratio": coreset_ratio,
        "seed": seed,
        "num_nominal_images": num_nominal_images,
        "bank_size": bank_size,
        "num_neighbors": num_neighbors,
        "gaussian_sigma": gaussian_sigma,
        "threshold": {
            "value": threshold,
            "percentile": threshold_percentile,
            "num_images": threshold_num_images,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def load_bank_meta(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_bank_meta(path: str | Path, meta: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")


def verify_bank_matches_model(meta: dict, model_path: str | Path) -> None:
    """Raises RuntimeError if `bank_meta.json`'s pinned model hash does not match
    the model package this run is configured to use -- a mismatched bank silently
    produces meaningless scores instead of failing, which this check turns into a
    load-time error instead of a scoring-time surprise."""
    actual = sha256_file(model_path)
    expected = meta.get("model_sha256")
    if expected != actual:
        raise RuntimeError(
            "memory bank was built against a different model package than the one configured "
            f"now (bank_meta.json model_sha256={expected}, configured model sha256={actual}); "
            "rebuild the bank with --calibrate against the current model.path"
        )


def verify_bank_hash(meta: dict, bank_path: str | Path) -> None:
    """Raises RuntimeError if `memory_bank.npy`'s contents don't match the hash
    `bank_meta.json` was saved with. The model-hash check above only proves the
    *model* is consistent; this proves the *bank* and the threshold derived
    from it are the ones actually paired -- an interrupted calibration or a
    bank file swapped in from a different run would otherwise still pass
    verify_bank_matches_model but score against the wrong threshold.

    `bank_sha256` is absent from bank_meta.json files written before this
    check existed; skip rather than fail so those banks keep working."""
    expected = meta.get("bank_sha256")
    if expected is None:
        return
    actual = sha256_file(bank_path)
    if expected != actual:
        raise RuntimeError(
            "memory_bank.npy does not match the bank bank_meta.json was saved with "
            f"(bank_meta.json bank_sha256={expected}, actual={actual}); rebuild both "
            "together with --calibrate"
        )
