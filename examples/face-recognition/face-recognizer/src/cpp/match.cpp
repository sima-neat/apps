#include "match.h"

#include <cmath>

namespace face_recog {

namespace {

float cosine_similarity(const Embedding& a, const Embedding& b) {
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  const float scale = std::sqrt(norm_a * norm_b);
  return scale > 0.0f ? dot / scale : 0.0f;
}

} // namespace

Match match_embedding(const Embedding& embedding, const Gallery& gallery, const MatchConfig& cfg) {
  Match best{cfg.unknown_label, 0.0f};
  float runner_up = -1.0f;
  bool found = false;
  for (const auto& person : gallery.people) {
    const float similarity = cosine_similarity(embedding, person.mean);
    if (!found || similarity > best.similarity) {
      if (found) {
        runner_up = best.similarity;
      }
      best = {person.name, similarity};
      found = true;
    } else if (similarity > runner_up) {
      runner_up = similarity;
    }
  }
  if (!found || best.similarity < cfg.threshold || best.similarity - runner_up < cfg.margin) {
    best.name = cfg.unknown_label;
  }
  return best;
}

} // namespace face_recog
