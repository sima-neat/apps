#include "gallery.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace face_recog {

static constexpr char kMagic[8] = {'F','R','G','A','L','1','\n','\0'};
static constexpr uint32_t kVersion = 3;  // v3 adds float32[512] raw_mean after sample_count

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
        f.write(reinterpret_cast<const char*>(&e.sample_count), 4);
        // v3: raw_mean is the unnormalized mean (weighted_sum/count) before L2-norm,
        // so re-enrollment can reconstruct the original weighted sum accurately.
        const Embedding& rm = (e.raw_mean.size() == kEmbeddingDim) ? e.raw_mean : e.embedding;
        f.write(reinterpret_cast<const char*>(rm.data()), kEmbeddingDim * sizeof(float));
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
    if (ver < 1 || ver > 3)
        throw std::runtime_error("load_gallery: unsupported version " + std::to_string(ver));

    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f)
        throw std::runtime_error("load_gallery: truncated header: " + path.string());

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
        uint32_t sample_count = 1;
        if (ver >= 2) {
            f.read(reinterpret_cast<char*>(&sample_count), 4);
            if (!f)
                throw std::runtime_error("load_gallery: truncated file reading sample_count: " + path.string());
        }
        // v3: read the unnormalized mean for accurate re-enrollment weight reconstruction.
        // For v1/v2, approximate with the normalized embedding (introduces small error on re-enroll).
        Embedding raw_mean(kEmbeddingDim);
        if (ver >= 3) {
            f.read(reinterpret_cast<char*>(raw_mean.data()), kEmbeddingDim * sizeof(float));
            if (!f)
                throw std::runtime_error("load_gallery: truncated file reading raw_mean: " + path.string());
        } else {
            raw_mean = emb;
        }
        g.entries.push_back({std::move(name), std::move(emb), std::move(raw_mean), sample_count});
    }

    return g;
}

// ── GalleryBuilder ────────────────────────────────────────────────────────────

void GalleryBuilder::add(const std::string& name, const Embedding& raw_emb, uint32_t count) {
    for (auto& acc : accum) {
        if (acc.name == name) {
            for (size_t j = 0; j < raw_emb.size(); ++j)
                acc.weighted_sum[j] += raw_emb[j] * static_cast<float>(count);
            acc.total_count += count;
            return;
        }
    }
    Embedding ws(raw_emb.size());
    for (size_t j = 0; j < raw_emb.size(); ++j)
        ws[j] = raw_emb[j] * static_cast<float>(count);
    accum.push_back({name, std::move(ws), count});
}

Gallery GalleryBuilder::finish() const {
    Gallery g;
    g.entries.reserve(accum.size());
    for (const auto& acc : accum) {
        if (acc.total_count == 0) continue;
        // raw_mean = weighted_sum / count (unnormalized); stored so re-enrollment can
        // reconstruct the original weighted sum as raw_mean * count without magnitude loss.
        Embedding raw_mean = acc.weighted_sum;
        const float inv = 1.f / static_cast<float>(acc.total_count);
        for (float& v : raw_mean) v *= inv;
        Embedding normalized = raw_mean;
        l2_normalize(normalized);
        g.entries.push_back({acc.name, std::move(normalized), std::move(raw_mean), acc.total_count});
    }
    return g;
}

} // namespace face_recog
