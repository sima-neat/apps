#include "patchcore_memory_bank.h"
#include "sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace patchcore {

// ---------------------------------------------------------------------------
// extract_hwc
// ---------------------------------------------------------------------------

PatchEmbeddings extract_hwc(const std::vector<int64_t>& shape, const std::vector<float>& flat,
                            int embed_dim) {
  std::vector<int64_t> squeezed;
  for (const auto dim : shape) {
    if (dim > 1) {
      squeezed.push_back(dim);
    }
  }
  if (squeezed.size() != 3) {
    throw std::runtime_error("expected a 3D patch-embedding tensor after squeezing the batch "
                             "dim, got " +
                             std::to_string(squeezed.size()) + " non-unit dims");
  }

  PatchEmbeddings out;
  if (squeezed[2] == embed_dim) {
    out.height = static_cast<int>(squeezed[0]);
    out.width = static_cast<int>(squeezed[1]);
    out.channels = embed_dim;
    out.values = flat;
    return out;
  }
  if (squeezed[0] == embed_dim) {
    out.height = static_cast<int>(squeezed[1]);
    out.width = static_cast<int>(squeezed[2]);
    out.channels = embed_dim;
    out.values.resize(flat.size());
    const int h = out.height;
    const int w = out.width;
    for (int c = 0; c < embed_dim; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          out.values[(static_cast<std::size_t>(y) * w + x) * embed_dim + c] =
              flat[(static_cast<std::size_t>(c) * h + y) * w + x];
        }
      }
    }
    return out;
  }
  throw std::runtime_error("could not find the " + std::to_string(embed_dim) +
                           "-channel axis in the patch-embedding tensor");
}

// ---------------------------------------------------------------------------
// NPY read/write -- minimal v1.0 float32 2D reader/writer (see header).
// ---------------------------------------------------------------------------

namespace {

constexpr char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

struct NpyHeader {
  std::size_t rows = 0;
  std::size_t cols = 0;
};

std::string read_exact(std::ifstream& in, std::size_t n) {
  std::string buf(n, '\0');
  in.read(buf.data(), static_cast<std::streamsize>(n));
  if (static_cast<std::size_t>(in.gcount()) != n) {
    throw std::runtime_error("unexpected end of file while reading .npy header");
  }
  return buf;
}

NpyHeader parse_header_dict(const std::string& dict) {
  if (dict.find("'<f4'") == std::string::npos && dict.find("\"<f4\"") == std::string::npos) {
    throw std::runtime_error(".npy dtype must be little-endian float32 ('<f4')");
  }
  if (dict.find("fortran_order': True") != std::string::npos ||
      dict.find("fortran_order\": true") != std::string::npos) {
    throw std::runtime_error(".npy must be C-order, not Fortran-order");
  }

  const auto shape_key = dict.find("'shape'");
  const auto shape_pos = shape_key == std::string::npos ? dict.find("\"shape\"") : shape_key;
  if (shape_pos == std::string::npos) {
    throw std::runtime_error(".npy header missing 'shape'");
  }
  const auto paren_open = dict.find('(', shape_pos);
  const auto paren_close = dict.find(')', paren_open);
  if (paren_open == std::string::npos || paren_close == std::string::npos) {
    throw std::runtime_error(".npy header has a malformed shape tuple");
  }
  const std::string tuple = dict.substr(paren_open + 1, paren_close - paren_open - 1);

  std::vector<std::size_t> dims;
  std::stringstream ss(tuple);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const auto first = item.find_first_not_of(" \t");
    const auto last = item.find_last_not_of(" \t");
    if (first == std::string::npos) {
      continue;
    }
    dims.push_back(static_cast<std::size_t>(std::stoull(item.substr(first, last - first + 1))));
  }
  if (dims.size() != 2) {
    throw std::runtime_error("memory bank .npy must be a 2D array, got " +
                             std::to_string(dims.size()) + " dims");
  }
  return NpyHeader{dims[0], dims[1]};
}

} // namespace

void MemoryBank::compute_squared_norms() {
  vectors_sq_.resize(num_vectors_);
  for (std::size_t b = 0; b < num_vectors_; ++b) {
    const float* row = vectors_.data() + b * embed_dim_;
    float acc = 0.0f;
    for (std::size_t c = 0; c < embed_dim_; ++c) {
      acc += row[c] * row[c];
    }
    vectors_sq_[b] = acc;
  }
}

MemoryBank MemoryBank::load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    throw std::runtime_error("could not open memory bank file: " + path.string());
  }

  const std::string magic = read_exact(in, 6);
  if (std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("not a valid .npy file (bad magic): " + path.string());
  }
  const std::string version = read_exact(in, 2);
  const auto major = static_cast<unsigned char>(version[0]);

  std::size_t header_len = 0;
  if (major == 1) {
    const std::string len_bytes = read_exact(in, 2);
    header_len = static_cast<unsigned char>(len_bytes[0]) |
                (static_cast<unsigned char>(len_bytes[1]) << 8);
  } else {
    const std::string len_bytes = read_exact(in, 4);
    header_len = static_cast<unsigned char>(len_bytes[0]) |
                (static_cast<unsigned char>(len_bytes[1]) << 8) |
                (static_cast<unsigned char>(len_bytes[2]) << 16) |
                (static_cast<unsigned char>(len_bytes[3]) << 24);
  }
  const std::string header_dict = read_exact(in, header_len);
  const NpyHeader header = parse_header_dict(header_dict);

  MemoryBank bank;
  bank.num_vectors_ = header.rows;
  bank.embed_dim_ = header.cols;
  bank.vectors_.resize(header.rows * header.cols);
  if (!bank.vectors_.empty()) {
    in.read(reinterpret_cast<char*>(bank.vectors_.data()),
           static_cast<std::streamsize>(bank.vectors_.size() * sizeof(float)));
    if (static_cast<std::size_t>(in.gcount()) != bank.vectors_.size() * sizeof(float)) {
      throw std::runtime_error("memory bank .npy payload is truncated: " + path.string());
    }
  }
  bank.compute_squared_norms();
  return bank;
}

void MemoryBank::save(const std::filesystem::path& path) const {
  if (empty()) {
    throw std::runtime_error("memory bank has no vectors to save");
  }

  std::ostringstream dict;
  dict << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << num_vectors_ << ", "
      << embed_dim_ << "), }";
  std::string header_dict = dict.str();

  const std::size_t prefix = 6 + 2 + 2;
  std::size_t total = prefix + header_dict.size() + 1;
  const std::size_t remainder = total % 64;
  const std::size_t pad = remainder == 0 ? 0 : 64 - remainder;
  header_dict.append(pad, ' ');
  header_dict.push_back('\n');

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("could not open memory bank file for writing: " + path.string());
  }
  out.write(kMagic, sizeof(kMagic));
  const char version[2] = {1, 0};
  out.write(version, sizeof(version));
  const auto header_len = static_cast<std::uint16_t>(header_dict.size());
  const char len_bytes[2] = {static_cast<char>(header_len & 0xFF),
                             static_cast<char>((header_len >> 8) & 0xFF)};
  out.write(len_bytes, sizeof(len_bytes));
  out.write(header_dict.data(), static_cast<std::streamsize>(header_dict.size()));
  out.write(reinterpret_cast<const char*>(vectors_.data()),
           static_cast<std::streamsize>(vectors_.size() * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("failed writing memory bank file: " + path.string());
  }
}

// ---------------------------------------------------------------------------
// Greedy k-center coreset
// ---------------------------------------------------------------------------

std::vector<std::size_t> greedy_coreset_indices(const std::vector<float>& vectors,
                                                std::size_t embed_dim, double ratio,
                                                std::uint64_t seed) {
  if (embed_dim == 0 || vectors.size() % embed_dim != 0) {
    throw std::runtime_error("greedy_coreset_indices: vectors size is not a multiple of embed_dim");
  }
  const std::size_t n = vectors.size() / embed_dim;
  const std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(
                                                      static_cast<double>(n) * ratio)));
  std::vector<std::size_t> selected;
  selected.reserve(std::min(k, n));
  if (k >= n) {
    for (std::size_t i = 0; i < n; ++i) {
      selected.push_back(i);
    }
    return selected;
  }

  auto row = [&](std::size_t idx) { return vectors.data() + idx * embed_dim; };
  auto sq_dist = [&](std::size_t a, std::size_t b) {
    const float* ra = row(a);
    const float* rb = row(b);
    float acc = 0.0f;
    for (std::size_t c = 0; c < embed_dim; ++c) {
      const float d = ra[c] - rb[c];
      acc += d * d;
    }
    return acc;
  };

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> start_dist(0, n - 1);
  const std::size_t start = start_dist(rng);
  selected.push_back(start);

  std::vector<float> min_sq_dist(n);
  for (std::size_t i = 0; i < n; ++i) {
    min_sq_dist[i] = sq_dist(i, start);
  }

  for (std::size_t iter = 1; iter < k; ++iter) {
    const std::size_t next_idx = static_cast<std::size_t>(
        std::max_element(min_sq_dist.begin(), min_sq_dist.end()) - min_sq_dist.begin());
    selected.push_back(next_idx);
    for (std::size_t i = 0; i < n; ++i) {
      min_sq_dist[i] = std::min(min_sq_dist[i], sq_dist(i, next_idx));
    }
  }
  return selected;
}

MemoryBank MemoryBank::build(const std::vector<PatchEmbeddings>& per_image_embeddings,
                             double coreset_ratio, std::uint64_t seed) {
  if (per_image_embeddings.empty()) {
    throw std::runtime_error("no reference ('normal') images given to build the memory bank");
  }
  const std::size_t embed_dim = static_cast<std::size_t>(per_image_embeddings.front().channels);

  std::size_t total_patches = 0;
  for (const auto& img : per_image_embeddings) {
    if (static_cast<std::size_t>(img.channels) != embed_dim) {
      throw std::runtime_error("all reference images must share the same embedding dimension");
    }
    total_patches += img.patch_count();
  }

  std::vector<float> pooled;
  pooled.reserve(total_patches * embed_dim);
  for (const auto& img : per_image_embeddings) {
    pooled.insert(pooled.end(), img.values.begin(), img.values.end());
  }

  const auto selected = greedy_coreset_indices(pooled, embed_dim, coreset_ratio, seed);

  MemoryBank bank;
  bank.embed_dim_ = embed_dim;
  bank.num_vectors_ = selected.size();
  bank.vectors_.resize(selected.size() * embed_dim);
  for (std::size_t i = 0; i < selected.size(); ++i) {
    std::copy_n(pooled.begin() + static_cast<std::ptrdiff_t>(selected[i] * embed_dim), embed_dim,
               bank.vectors_.begin() + static_cast<std::ptrdiff_t>(i * embed_dim));
  }
  bank.compute_squared_norms();
  return bank;
}

// ---------------------------------------------------------------------------
// Scoring, with the PatchCore neighborhood-reweighting term
// ---------------------------------------------------------------------------

namespace {

/// Dot product of two length-`n` float arrays. On aarch64 this runs as
/// explicit NEON with two independent accumulators to hide FMA latency,
/// instead of relying on the compiler to auto-vectorize the scalar reduction
/// below. `n` is always embed_dim_ (1536, a multiple of 8); the tail loop
/// covers any remainder for safety if that ever changes.
inline float dot_product(const float* a, const float* b, std::size_t n) {
#if defined(__aarch64__)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  std::size_t c = 0;
  for (; c + 8 <= n; c += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + c), vld1q_f32(b + c));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + c + 4), vld1q_f32(b + c + 4));
  }
  float total = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; c < n; ++c) {
    total += a[c] * b[c];
  }
  return total;
#else
  float acc = 0.0f;
  for (std::size_t c = 0; c < n; ++c) {
    acc += a[c] * b[c];
  }
  return acc;
#endif
}

} // namespace

AnomalyResult MemoryBank::score(const PatchEmbeddings& embeddings, int num_neighbors) const {
  if (empty()) {
    throw std::runtime_error("memory bank is empty; call build() or load() first");
  }
  if (static_cast<std::size_t>(embeddings.channels) != embed_dim_) {
    throw std::runtime_error("patch embedding dimension does not match the memory bank");
  }

  const std::size_t num_patches = embeddings.patch_count();
  AnomalyResult result;
  result.score_map.resize(num_patches);
  std::vector<std::size_t> locations(num_patches);

  auto bank_row = [&](std::size_t idx) { return vectors_.data() + idx * embed_dim_; };
  // Expand-and-dot L2 identity (matches patchcore_scoring.py's _pairwise_l2):
  // ||q-b||^2 = ||q||^2 + ||b||^2 - 2*q.b. `vectors_sq_` (||b||^2 per bank row) is
  // precomputed once for the whole bank instead of recomputed on every call, and
  // the per-pair cost drops from a fused subtract-square-sum to a plain dot
  // product, which auto-vectorizes better -- both matter at video/RTSP frame rates.
  auto dist_to_bank_row = [&](const float* q, float q_sq, std::size_t b) {
    const float dot = dot_product(q, bank_row(b), embed_dim_);
    const float sq_dist = q_sq + vectors_sq_[b] - 2.0f * dot;
    return std::sqrt(std::max(0.0f, sq_dist));
  };
  auto squared_norm = [&](const float* v) { return dot_product(v, v, embed_dim_); };

  // This loop (an O(num_patches * num_vectors_) brute-force nearest-neighbor
  // search) dominates score()'s cost. Patches are independent -- each only
  // reads embeddings/vectors_/vectors_sq_ and writes its own index of
  // score_map/locations -- so it parallelizes safely with no locking. Thread
  // count is capped below hardware_concurrency() to leave headroom for the
  // rest of the pipeline (GStreamer decode/encode, MLA driver).
  auto scan_patch_range = [&](std::size_t start, std::size_t stop, float& best_score,
                              std::size_t& best_patch) {
    best_score = -std::numeric_limits<float>::infinity();
    best_patch = start;
    for (std::size_t p = start; p < stop; ++p) {
      const float* q = embeddings.values.data() + p * embed_dim_;
      const float q_sq = squared_norm(q);
      float best = std::numeric_limits<float>::infinity();
      std::size_t best_idx = 0;
      for (std::size_t b = 0; b < num_vectors_; ++b) {
        const float d = dist_to_bank_row(q, q_sq, b);
        if (d < best) {
          best = d;
          best_idx = b;
        }
      }
      result.score_map[p] = best;
      locations[p] = best_idx;
      if (best > best_score) {
        best_score = best;
        best_patch = p;
      }
    }
  };

  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const std::size_t num_threads =
      std::max<std::size_t>(1, std::min<std::size_t>(hw_threads > 0 ? hw_threads / 2 : 4, 8));

  std::size_t max_patch = 0;
  float max_patch_score = -std::numeric_limits<float>::infinity();

  if (num_threads > 1 && num_patches >= num_threads) {
    std::vector<std::size_t> local_max_patch(num_threads, 0);
    std::vector<float> local_max_score(num_threads, -std::numeric_limits<float>::infinity());
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    const std::size_t chunk = (num_patches + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
      const std::size_t start = t * chunk;
      const std::size_t stop = std::min(start + chunk, num_patches);
      if (start >= stop) {
        continue;
      }
      workers.emplace_back([&, t, start, stop]() {
        scan_patch_range(start, stop, local_max_score[t], local_max_patch[t]);
      });
    }
    for (auto& w : workers) {
      w.join();
    }
    for (std::size_t t = 0; t < num_threads; ++t) {
      if (local_max_score[t] > max_patch_score) {
        max_patch_score = local_max_score[t];
        max_patch = local_max_patch[t];
      }
    }
  } else {
    scan_patch_range(0, num_patches, max_patch_score, max_patch);
  }

  const int k = std::min<int>(num_neighbors, static_cast<int>(num_vectors_));
  if (k <= 1) {
    result.image_score = max_patch_score;
    return result;
  }

  // m^test,* (the test patch with the largest nearest-neighbor distance) and its
  // own nearest bank neighbor m^*.
  const float* q_star = embeddings.values.data() + max_patch * embed_dim_;
  const float s_star = max_patch_score;
  const std::size_t nn_index = locations[max_patch];
  const float* m_star = bank_row(nn_index);
  const float m_star_sq = vectors_sq_[nn_index];

  // N_b(m^*): the k nearest bank neighbors of m^* itself (ascending by distance
  // to m^*; index 0 is m^* itself, at distance 0).
  std::vector<std::pair<float, std::size_t>> dist_to_m_star(num_vectors_);
  for (std::size_t b = 0; b < num_vectors_; ++b) {
    dist_to_m_star[b] = {dist_to_bank_row(m_star, m_star_sq, b), b};
  }
  std::partial_sort(dist_to_m_star.begin(), dist_to_m_star.begin() + k, dist_to_m_star.end());

  // Distance from the test patch (not m^*) to each support sample.
  const float q_star_sq = squared_norm(q_star);
  std::vector<float> support_dists(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    support_dists[static_cast<std::size_t>(i)] =
        dist_to_bank_row(q_star, q_star_sq, dist_to_m_star[static_cast<std::size_t>(i)].second);
  }
  const float max_dist = *std::max_element(support_dists.begin(), support_dists.end());
  float sum_exp = 0.0f;
  for (const float d : support_dists) {
    sum_exp += std::exp(d - max_dist);
  }
  const float softmax0 = std::exp(support_dists[0] - max_dist) / sum_exp;
  const float weight = 1.0f - softmax0;

  result.image_score = weight * s_star;
  return result;
}

// ---------------------------------------------------------------------------
// Threshold
// ---------------------------------------------------------------------------

float percentile_threshold(std::vector<float> scores, double percentile) {
  if (scores.empty()) {
    throw std::runtime_error("no scores given to derive a threshold from");
  }
  std::sort(scores.begin(), scores.end());
  const double rank = (percentile / 100.0) * static_cast<double>(scores.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(rank));
  const auto hi = static_cast<std::size_t>(std::ceil(rank));
  if (lo == hi) {
    return scores[lo];
  }
  const double frac = rank - static_cast<double>(lo);
  return static_cast<float>(scores[lo] + frac * (scores[hi] - scores[lo]));
}

// ---------------------------------------------------------------------------
// bank_meta.json
// ---------------------------------------------------------------------------

BankMeta load_bank_meta(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.good()) {
    throw std::runtime_error("could not open bank meta file: " + path.string());
  }
  nlohmann::json j;
  in >> j;

  BankMeta meta;
  meta.model_sha256 = j.value("model_sha256", "");
  meta.bank_sha256 = j.value("bank_sha256", "");
  meta.model_filename = j.value("model_filename", "");
  meta.backbone = j.value("backbone", "");
  meta.torchvision_weights = j.value("torchvision_weights", "");
  meta.embed_dim = j.value("embed_dim", 0);
  if (j.contains("patch_grid") && j["patch_grid"].is_array() && j["patch_grid"].size() == 2) {
    meta.patch_grid_h = j["patch_grid"][0].get<int>();
    meta.patch_grid_w = j["patch_grid"][1].get<int>();
  }
  meta.coreset_ratio = j.value("coreset_ratio", 0.0);
  meta.seed = j.value("seed", static_cast<std::uint64_t>(0));
  meta.num_nominal_images = j.value("num_nominal_images", 0);
  meta.bank_size = j.value("bank_size", 0);
  meta.num_neighbors = j.value("num_neighbors", 0);
  meta.gaussian_sigma = j.value("gaussian_sigma", 0.0);
  if (j.contains("threshold") && j["threshold"].is_object()) {
    const auto& t = j["threshold"];
    meta.threshold_value = t.value("value", 0.0);
    meta.threshold_percentile = t.value("percentile", 0.0);
    meta.threshold_num_images = t.value("num_images", 0);
  }
  meta.created_at = j.value("created_at", "");
  return meta;
}

void save_bank_meta(const std::filesystem::path& path, const BankMeta& meta) {
  nlohmann::json j;
  j["model_sha256"] = meta.model_sha256;
  j["bank_sha256"] = meta.bank_sha256;
  j["model_filename"] = meta.model_filename;
  j["backbone"] = meta.backbone;
  j["torchvision_weights"] = meta.torchvision_weights;
  j["embed_dim"] = meta.embed_dim;
  j["patch_grid"] = {meta.patch_grid_h, meta.patch_grid_w};
  j["coreset_ratio"] = meta.coreset_ratio;
  j["seed"] = meta.seed;
  j["num_nominal_images"] = meta.num_nominal_images;
  j["bank_size"] = meta.bank_size;
  j["num_neighbors"] = meta.num_neighbors;
  j["gaussian_sigma"] = meta.gaussian_sigma;
  j["threshold"] = {
      {"value", meta.threshold_value},
      {"percentile", meta.threshold_percentile},
      {"num_images", meta.threshold_num_images},
  };
  j["created_at"] = meta.created_at;

  std::ofstream out(path, std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("could not open bank meta file for writing: " + path.string());
  }
  out << j.dump(2) << "\n";
}

void verify_bank_matches_model(const BankMeta& meta, const std::filesystem::path& model_path) {
  const std::string actual = sha256_file(model_path);
  if (meta.model_sha256 != actual) {
    throw std::runtime_error(
        "memory bank was built against a different model package than the one configured now "
        "(bank_meta.json model_sha256=" +
        meta.model_sha256 + ", configured model sha256=" + actual +
        "); rebuild the bank with --calibrate against the current model.path");
  }
}

void verify_bank_hash(const BankMeta& meta, const std::filesystem::path& bank_path) {
  if (meta.bank_sha256.empty()) {
    return;
  }
  const std::string actual = sha256_file(bank_path);
  if (meta.bank_sha256 != actual) {
    throw std::runtime_error(
        "memory_bank.npy does not match the bank bank_meta.json was saved with "
        "(bank_meta.json bank_sha256=" +
        meta.bank_sha256 + ", actual=" + actual + "); rebuild both together with --calibrate");
  }
}

std::string current_utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&now, &tm_utc);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return buf;
}

} // namespace patchcore
