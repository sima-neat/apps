"""Host-side PatchCore memory bank: coreset build, calibration, and anomaly
scoring -- non-parametric, so it runs as plain numpy rather than in the
compiled graph, following Roth et al., "Towards Total Recall in Industrial
Anomaly Detection" (CVPR 2022).

Two on-disk artifacts travel together: `memory_bank.npy` (the coreset, a
float32 (N, embed_dim) array) and `bank_meta.json` (model hash, calibration
params, and decision threshold). The pinned model hash makes a bank pointed
at the wrong compiled package fail loudly at load time instead of silently
producing meaningless scores.
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


def nearest_neighbors(query: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """query: (N, C), bank: (M, C), both float32 -> (distance, bank index) of each
    query row's nearest bank row. OpenCV's batchDistance sums squared differences
    directly (no BLAS matmul), the same computation as the C++ implementation."""
    dist, idx = cv2.batchDistance(query, bank, cv2.CV_32F, normType=cv2.NORM_L2, K=1)
    return dist[:, 0], idx[:, 0]


def greedy_coreset_indices(vectors: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    """Greedy k-center (farthest-point) coreset selection, matching the
    PatchCore paper's subsampling strategy, but in the full `embed_dim`-
    dimensional space rather than a random low-dimensional projection
    (simpler, at the cost of build time on very large nominal sets). Returns
    row indices into `vectors`, not the vectors themselves."""
    n = vectors.shape[0]
    k = max(1, int(round(n * ratio)))
    if k >= n:
        return np.arange(n)

    selected = np.empty(k, dtype=np.int64)
    # Start at patch `seed % n`, the same rule as the C++ implementation, so both
    # languages build the same bank from the same images and seed.
    selected[0] = seed % n
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
        flat = np.ascontiguousarray(patch_embeddings.reshape(-1, c), dtype=np.float32)

        # s per patch, and the index of its nearest bank vector (m^* for each patch).
        patch_scores, locations = nearest_neighbors(flat, self.vectors)
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
        dist_to_m_star = np.linalg.norm(self.vectors - m_star[None, :], axis=1)
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
    patch_threshold: float,
    patch_scale_min: float,
    patch_scale_max: float,
) -> dict:
    return {
        "model_sha256": sha256_file(model_path),
        "model_filename": Path(model_path).name,
        # Pins this metadata to the memory_bank.npy its threshold was derived from.
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
        # Overlay highlight: pixels above scale_max are highlighted, colored over the span
        # scale_max - scale_min. Separate from the image-level threshold.
        "patch_threshold": {
            "value": patch_threshold,
            "scale_min": patch_scale_min,
            "scale_max": patch_scale_max,
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
    """Raises RuntimeError if the bank's pinned model hash doesn't match the
    configured model -- a mismatched bank otherwise silently scores wrong."""
    actual = sha256_file(model_path)
    expected = meta.get("model_sha256")
    if expected != actual:
        raise RuntimeError(
            "memory bank was built against a different model package than the one configured "
            f"now (bank_meta.json model_sha256={expected}, configured model sha256={actual}); "
            "rebuild the bank with --calibrate against the current model.path"
        )


def verify_bank_hash(meta: dict, bank_path: str | Path) -> None:
    """Raises RuntimeError if the bank file's contents don't match the hash
    bank_meta.json was saved with. Skipped when `bank_sha256` is absent."""
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
