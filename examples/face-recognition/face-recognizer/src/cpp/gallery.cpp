#include "gallery.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace face_recog {

static constexpr char kMagic[8] = {'F','R','G','A','L','1','\n','\0'};
static constexpr uint32_t kVersion = 1;

// ── l2-normalization ──────────────────────────────────────────────────────────

void l2_normalize(Embedding& emb) {
    float norm2 = 0.f;
    for (float v : emb) norm2 += v * v;
    if (norm2 < 1e-12f) return;
    const float inv = 1.f / std::sqrt(norm2);
    for (float& v : emb) v *= inv;
}

// ── persistence ───────────────────────────────────────────────────────────────

void save_gallery(const Gallery& g, const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("save_gallery: cannot open for writing: " + path.string());

    f.write(kMagic, 8);
    const uint32_t ver = kVersion;
    f.write(reinterpret_cast<const char*>(&ver), 4);
    const uint32_t n = static_cast<uint32_t>(g.entries.size());
    f.write(reinterpret_cast<const char*>(&n), 4);

    for (const auto& e : g.entries) {
        if (e.embedding.size() != kEmbeddingDim)
            throw std::runtime_error("save_gallery: embedding dim mismatch for '" + e.name + "'");
        const uint16_t nl = static_cast<uint16_t>(e.name.size());
        f.write(reinterpret_cast<const char*>(&nl), 2);
        f.write(e.name.data(), nl);
        f.write(reinterpret_cast<const char*>(e.embedding.data()),
                kEmbeddingDim * sizeof(float));
    }

    if (!f)
        throw std::runtime_error("save_gallery: write error: " + path.string());
}

Gallery load_gallery(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("load_gallery: cannot open: " + path.string());

    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, kMagic, 8) != 0)
        throw std::runtime_error("load_gallery: invalid magic in: " + path.string());

    uint32_t ver = 0;
    f.read(reinterpret_cast<char*>(&ver), 4);
    if (ver != kVersion)
        throw std::runtime_error("load_gallery: unsupported version " + std::to_string(ver));

    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);

    Gallery g;
    g.entries.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        uint16_t nl = 0;
        f.read(reinterpret_cast<char*>(&nl), 2);
        if (!f)
            throw std::runtime_error("load_gallery: truncated file reading name length: " + path.string());
        std::string name(nl, '\0');
        f.read(name.data(), nl);
        Embedding emb(kEmbeddingDim);
        f.read(reinterpret_cast<char*>(emb.data()), kEmbeddingDim * sizeof(float));
        if (!f)
            throw std::runtime_error("load_gallery: truncated file: " + path.string());
        g.entries.push_back({std::move(name), std::move(emb)});
    }

    return g;
}

// ── GalleryBuilder ────────────────────────────────────────────────────────────

void GalleryBuilder::add(const std::string& name, const Embedding& raw_emb) {
    for (auto& acc : accum)
        if (acc.name == name) { acc.embeddings.push_back(raw_emb); return; }
    accum.push_back({name, {raw_emb}});
}

Gallery GalleryBuilder::finish() const {
    Gallery g;
    g.entries.reserve(accum.size());
    for (const auto& acc : accum) {
        if (acc.embeddings.empty()) continue;
        const size_t dim = acc.embeddings.front().size();
        Embedding mean(dim, 0.f);
        for (const auto& e : acc.embeddings)
            for (size_t j = 0; j < dim; ++j)
                mean[j] += e[j];
        const float inv = 1.f / acc.embeddings.size();
        for (float& v : mean) v *= inv;
        l2_normalize(mean);
        g.entries.push_back({acc.name, std::move(mean)});
    }
    return g;
}

} // namespace face_recog
