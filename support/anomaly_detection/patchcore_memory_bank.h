#pragma once

/// PatchCore memory-bank storage, coreset construction, and nearest-neighbor
/// anomaly scoring -- the host-side half of the two-stage split described in
/// examples/anomaly-detection/patchcore/README.md. The compiled MLA graph only
/// extracts per-patch feature embeddings; the anomaly decision itself is a
/// non-parametric lookup against a coreset of "normal" reference patches, so it
/// has no place in a compiled graph and lives here instead, shared by every
/// language variant of this example.
///
/// Two on-disk artifacts travel together as a versioned pair:
///   - `memory_bank.npy`: the coreset, a float32 (N, embed_dim) array (the exact
///     layout `numpy.save` writes for such an array; this reader/writer only
///     supports that one shape).
///   - `bank_meta.json`: the model package hash the bank was built against, the
///     coreset ratio and nominal-image count used, and the decision threshold
///     with the percentile and image count it was derived from. See
///     `patchcore_scoring.py` for the byte-identical Python implementation this
///     mirrors.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace patchcore {

/// A dense (H, W, C) tensor of patch-feature embeddings, row-major with C fastest.
struct PatchEmbeddings {
  int height = 0;
  int width = 0;
  int channels = 0;
  std::vector<float> values; // size == height * width * channels

  [[nodiscard]] std::size_t patch_count() const {
    return static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  }
};

/// Normalizes a raw model output buffer to (H, W, C). The MLA compiles tensors
/// NHWC-native, but the channel axis is detected at runtime instead of assumed:
/// `shape` is the tensor's shape (including any size-1 batch dim), and `flat` is
/// its dense row-major payload, already converted to float32.
PatchEmbeddings extract_hwc(const std::vector<int64_t>& shape, const std::vector<float>& flat,
                            int embed_dim);

/// Nearest-neighbor distance from every patch in an image to the memory bank.
struct AnomalyResult {
  std::vector<float> score_map; // size == height * width, row-major (H, W)
  float image_score = 0.0f;     // PatchCore-reweighted image-level anomaly score
};

/// Greedy k-center (farthest-point) coreset selection, matching the PatchCore
/// paper's subsampling strategy: pick a random start, then repeatedly add
/// whichever remaining point is farthest (by L2) from every point already
/// selected. Runs directly in the full `embed_dim`-dimensional space rather than
/// the paper's random low-dimensional (Johnson-Lindenstrauss) projection --
/// simpler, at the cost of build-time speed on very large nominal sets.
///
/// `vectors` is a row-major (n, embed_dim) pool; returns the selected row
/// indices, sized `max(1, round(n * ratio))` (or all of `vectors` if that would
/// exceed `n`).
std::vector<std::size_t> greedy_coreset_indices(const std::vector<float>& vectors,
                                                std::size_t embed_dim, double ratio,
                                                std::uint64_t seed);

/// Coreset of "normal" patch-feature vectors, plus nearest-neighbor anomaly scoring.
class MemoryBank {
public:
  MemoryBank() = default;

  static MemoryBank load(const std::filesystem::path& path);
  void save(const std::filesystem::path& path) const;

  /// Pools every patch from every reference image, then greedily subsamples the
  /// pool down to `coreset_ratio` of its size (see `greedy_coreset_indices`).
  static MemoryBank build(const std::vector<PatchEmbeddings>& per_image_embeddings,
                          double coreset_ratio, std::uint64_t seed);

  /// Per-patch nearest-neighbor L2 distance to the bank (`score_map`), and the
  /// image-level score with the PatchCore paper's neighborhood-reweighting term
  /// (Eq. 7-8 of Roth et al., CVPR 2022; matches anomalib's reference
  /// implementation). `num_neighbors` is the reweighting support-set size;
  /// values <= 1 (or a bank too small to support it) fall back to the plain
  /// max-of-score-map image score.
  [[nodiscard]] AnomalyResult score(const PatchEmbeddings& embeddings, int num_neighbors) const;

  [[nodiscard]] std::size_t size() const { return num_vectors_; }
  [[nodiscard]] std::size_t embed_dim() const { return embed_dim_; }
  [[nodiscard]] bool empty() const { return num_vectors_ == 0; }

private:
  void compute_squared_norms();

  std::vector<float> vectors_; // row-major (num_vectors_, embed_dim_)
  std::size_t num_vectors_ = 0;
  std::size_t embed_dim_ = 0;
  // sum(vectors_[i] * vectors_[i]) per bank row, precomputed once (by load()/build(),
  // via compute_squared_norms()) instead of every score() call -- the bank never
  // changes mid-run, so recomputing this per video/RTSP frame was a wasted
  // O(bank_size * embed_dim) cost on the hot path.
  std::vector<float> vectors_sq_;
};

/// `p`-th percentile (linear interpolation, matching numpy.percentile's default)
/// of a list of per-image anomaly scores. Throws if `scores` is empty.
float percentile_threshold(std::vector<float> scores, double percentile);

/// The `bank_meta.json` sidecar: everything a scoring run needs to trust and
/// interpret `memory_bank.npy` without hard-coding it in application source.
struct BankMeta {
  std::string model_sha256;
  // Pins this metadata to the exact memory_bank.npy it was derived from -- the
  // threshold below is only valid for that bank's score distribution. Empty
  // for bank_meta.json files written before this field existed; verify_bank_hash
  // skips the check in that case rather than failing.
  std::string bank_sha256;
  std::string model_filename;
  std::string backbone;
  std::string torchvision_weights;
  int embed_dim = 0;
  int patch_grid_h = 0;
  int patch_grid_w = 0;
  double coreset_ratio = 0.0;
  std::uint64_t seed = 0;
  int num_nominal_images = 0;
  int bank_size = 0;
  int num_neighbors = 0;
  double gaussian_sigma = 0.0;
  double threshold_value = 0.0;
  double threshold_percentile = 0.0;
  int threshold_num_images = 0;
  std::string created_at;
};

BankMeta load_bank_meta(const std::filesystem::path& path);
void save_bank_meta(const std::filesystem::path& path, const BankMeta& meta);

/// Current UTC time as "%Y-%m-%dT%H:%M:%SZ", for `BankMeta::created_at`.
std::string current_utc_timestamp();

/// Throws `std::runtime_error` if `meta.model_sha256` does not match the sha256
/// of `model_path` -- a mismatched bank silently produces meaningless scores
/// instead of failing, which this check turns into a load-time error.
void verify_bank_matches_model(const BankMeta& meta, const std::filesystem::path& model_path);

/// Throws `std::runtime_error` if `meta.bank_sha256` is set and does not match
/// the sha256 of `bank_path` -- proves the bank and the threshold derived from
/// it are the ones actually paired, which verify_bank_matches_model alone
/// cannot: an interrupted calibration or a bank swapped in from a different
/// run still has the right model hash but the wrong score distribution.
void verify_bank_hash(const BankMeta& meta, const std::filesystem::path& bank_path);

} // namespace patchcore
