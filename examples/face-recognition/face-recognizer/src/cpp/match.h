#pragma once

#include "gallery.h"

#include <string>

namespace face_recog {

struct MatchResult {
    std::string name;         // Gallery label or "Unknown"
    float       score;        // Best cosine similarity
    int         index;        // Index in gallery (-1 if unknown/empty)
    std::string second_name;  // Second-best gallery label (for diagnostics)
    float       second_score; // Second-best cosine similarity
};

struct MatchConfig {
    float       threshold     = 0.35f;
    float       margin        = 0.08f;  // min gap between best and second-best score
    std::string unknown_label = "Unknown";
};

MatchResult match_embedding(
    const Embedding& raw_embedding,
    const Gallery&   gallery,
    const MatchConfig& cfg);

float cosine_similarity(const Embedding& a, const Embedding& b);

} // namespace face_recog
