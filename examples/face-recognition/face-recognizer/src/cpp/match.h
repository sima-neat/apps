#pragma once

#include "gallery.h"

#include <string>

namespace face_recog {

struct MatchConfig {
  float threshold = 0.55f;
  float margin = 0.12f;
  std::string unknown_label = "Unknown";
};

struct Match {
  std::string name;
  float similarity = 0.0f;
};

// The enrolled person most similar to `embedding`, by cosine similarity. The name is the
// unknown label when that similarity is below the threshold, or when the runner-up is closer
// than the margin.
Match match_embedding(const Embedding& embedding, const Gallery& gallery, const MatchConfig& cfg);

} // namespace face_recog
