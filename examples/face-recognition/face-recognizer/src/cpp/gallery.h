#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace face_recog {

constexpr int kEmbeddingDim = 512;

using Embedding = std::vector<float>;

struct GalleryEntry {
    std::string name;
    Embedding   embedding;  // L2-normalized, length kEmbeddingDim
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

// Add or update an entry (accumulates multiple embeddings by mean-pooling).
// Call finish_gallery() after all images for a person have been enrolled.
struct GalleryBuilder {
    struct Accum {
        std::string name;
        std::vector<Embedding> embeddings;
    };
    std::vector<Accum> accum;

    void add(const std::string& name, const Embedding& raw_emb);
    Gallery finish() const;  // mean-pool + L2-normalize each entry
};

} // namespace face_recog
