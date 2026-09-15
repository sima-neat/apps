#include "match.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace face_recog {

float cosine_similarity(const Embedding& a, const Embedding& b) {
    float dot = 0.f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;  // assumes both a and b are already L2-normalized
}

MatchResult match_embedding(
    const Embedding& raw_embedding,
    const Gallery&   gallery,
    const MatchConfig& cfg)
{
    if (gallery.entries.empty())
        return {cfg.unknown_label, -2.f, -1, "", -2.f};

    // Compute L2 norm inline and apply as a scale factor during dot-product,
    // avoiding the 512-float heap copy that l2_normalize(copy) would require.
    float sq_sum = 0.f;
    for (const float v : raw_embedding) sq_sum += v * v;
    const float inv_norm = (sq_sum > 0.f) ? (1.f / std::sqrt(sq_sum)) : 1.f;

    float best_sim   = -2.f;
    float second_sim = -2.f;
    int   best_idx   = -1;
    int   second_idx = -1;

    for (int i = 0; i < static_cast<int>(gallery.entries.size()); ++i) {
        // Gallery embeddings are pre-normalized; apply inv_norm to raw_embedding inline.
        float dot = 0.f;
        const auto& gal = gallery.entries[i].embedding;
        for (size_t k = 0; k < raw_embedding.size(); ++k)
            dot += raw_embedding[k] * gal[k];
        const float sim = dot * inv_norm;
        if (sim > best_sim) {
            second_sim = best_sim; second_idx = best_idx;
            best_sim = sim;        best_idx   = i;
        } else if (sim > second_sim) {
            second_sim = sim; second_idx = i;
        }
    }

    const std::string second_name = (second_idx >= 0)
        ? gallery.entries[second_idx].name : "";

    if (best_sim < cfg.threshold)
        return {cfg.unknown_label, best_sim, -1, second_name, second_sim};
    if (gallery.entries.size() > 1 && (best_sim - second_sim) < cfg.margin)
        return {cfg.unknown_label, best_sim, -1, second_name, second_sim};
    return {gallery.entries[best_idx].name, best_sim, best_idx, second_name, second_sim};
}

} // namespace face_recog
