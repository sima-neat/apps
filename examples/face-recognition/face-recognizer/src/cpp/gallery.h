#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace face_recog {

constexpr int kEmbeddingDim = 512;

using Embedding = std::vector<float>;

struct GalleryEntry {
    std::string name;
    Embedding   embedding;    // L2-normalized, length kEmbeddingDim
    uint32_t    sample_count; // number of raw embeddings averaged into this centroid
};

struct Gallery {
    std::vector<GalleryEntry> entries;
};

// ── persistence ─────────────────────────────────────────────────────────────
// File format (little-endian):
//   magic     : char[8]     = "FRGAL1\n\0"
//   version   : uint32_t    = 1
//   n_entries : uint32_t
//   for each entry:
//     name_len   : uint16_t
//     name       : char[name_len]   (UTF-8, no null)
//     embedding  : float32[512]

void save_gallery(const Gallery& g, const std::filesystem::path& path);
Gallery load_gallery(const std::filesystem::path& path);

// ── helpers ──────────────────────────────────────────────────────────────────
void l2_normalize(Embedding& emb);

// Accumulates embeddings and produces a gallery with properly-weighted centroids.
// Use add() for individual raw embeddings; use add_weighted() when importing an
// existing centroid with a known sample count so weights are preserved correctly.
struct GalleryBuilder {
    struct Accum {
        std::string name;
        Embedding   weighted_sum;  // sum of (embedding * sample_count) for each add
        uint32_t    total_count;   // total number of raw samples represented
    };
    std::vector<Accum> accum;

    void add(const std::string& name, const Embedding& raw_emb, uint32_t count = 1);
    Gallery finish() const;  // divide by total_count + L2-normalize each entry
};

} // namespace face_recog
