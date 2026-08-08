#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace multi_stream_people_tracker {
namespace {

constexpr float kBlockedCost = 1.0e6f;
constexpr float kUnmatchedCost = 1.0e3f;

struct TrackState {
  int track_id = 0;
  Detection detection;
  float velocity_x = 0.0f;
  float velocity_y = 0.0f;
  float velocity_w = 0.0f;
  float velocity_h = 0.0f;
  int last_frame_index = 0;
  int missing_frames = 0;
  int hits = 1;
  bool confirmed = true;
};

float width(const Detection& detection) {
  return std::max(0.0f, detection.x2 - detection.x1);
}

float height(const Detection& detection) {
  return std::max(0.0f, detection.y2 - detection.y1);
}

float center_x(const Detection& detection) {
  return 0.5f * (detection.x1 + detection.x2);
}

float center_y(const Detection& detection) {
  return 0.5f * (detection.y1 + detection.y2);
}

float iou_xyxy(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  const float union_area = width(a) * height(a) + width(b) * height(b) - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

float normalized_center_distance(const Detection& a, const Detection& b) {
  const float dx = center_x(a) - center_x(b);
  const float dy = center_y(a) - center_y(b);
  const float scale =
      std::max(1.0f, 0.5f * (std::hypot(width(a), height(a)) + std::hypot(width(b), height(b))));
  return std::hypot(dx, dy) / scale;
}

Detection predict(const TrackState& track, int frame_index) {
  const int elapsed = std::max(0, frame_index - track.last_frame_index);
  const float predicted_x = center_x(track.detection) + track.velocity_x * elapsed;
  const float predicted_y = center_y(track.detection) + track.velocity_y * elapsed;
  const float predicted_w = std::max(1.0f, width(track.detection) + track.velocity_w * elapsed);
  const float predicted_h = std::max(1.0f, height(track.detection) + track.velocity_h * elapsed);
  return Detection{predicted_x - predicted_w * 0.5f,
                   predicted_y - predicted_h * 0.5f,
                   predicted_x + predicted_w * 0.5f,
                   predicted_y + predicted_h * 0.5f,
                   track.detection.score,
                   track.detection.class_id};
}

void validate_config(const TrackerConfig& config) {
  if (!std::isfinite(config.high_score_threshold) || config.high_score_threshold < 0.0f ||
      config.high_score_threshold > 1.0f) {
    throw std::invalid_argument("high_score_threshold must be in [0, 1]");
  }
  if (!std::isfinite(config.new_track_threshold) ||
      config.new_track_threshold < config.high_score_threshold ||
      config.new_track_threshold > 1.0f) {
    throw std::invalid_argument("new_track_threshold must be in [high_score_threshold, 1]");
  }
  if (!std::isfinite(config.match_iou_threshold) || config.match_iou_threshold < 0.0f ||
      config.match_iou_threshold > 1.0f) {
    throw std::invalid_argument("match_iou_threshold must be in [0, 1]");
  }
  if (!std::isfinite(config.max_center_distance) || config.max_center_distance < 0.0f) {
    throw std::invalid_argument("max_center_distance must be >= 0");
  }
  if (!std::isfinite(config.velocity_momentum) || config.velocity_momentum < 0.0f ||
      config.velocity_momentum >= 1.0f) {
    throw std::invalid_argument("velocity_momentum must be in [0, 1)");
  }
  if (config.max_missing_frames < 0) {
    throw std::invalid_argument("max_missing_frames must be >= 0");
  }
  if (config.min_confirmed_hits < 1) {
    throw std::invalid_argument("min_confirmed_hits must be >= 1");
  }
  if (config.max_prediction_frames < 0 ||
      config.max_prediction_frames > config.max_missing_frames) {
    throw std::invalid_argument("max_prediction_frames must be in [0, max_missing_frames]");
  }
}

struct StageGeometry {
  std::vector<float> x1;
  std::vector<float> y1;
  std::vector<float> x2;
  std::vector<float> y2;
  std::vector<int> class_id;
  std::vector<int> detection_index;

  void prepare(const std::vector<Detection>& detections, const std::vector<int>& indices,
               const std::vector<unsigned char>& matched) {
    x1.clear();
    y1.clear();
    x2.clear();
    y2.clear();
    class_id.clear();
    detection_index.clear();
    x1.reserve(indices.size());
    y1.reserve(indices.size());
    x2.reserve(indices.size());
    y2.reserve(indices.size());
    class_id.reserve(indices.size());
    detection_index.reserve(indices.size());
    for (const int index : indices) {
      if (matched[static_cast<std::size_t>(index)] != 0) {
        continue;
      }
      const auto& detection = detections[static_cast<std::size_t>(index)];
      x1.push_back(detection.x1);
      y1.push_back(detection.y1);
      x2.push_back(detection.x2);
      y2.push_back(detection.y2);
      class_id.push_back(detection.class_id);
      detection_index.push_back(index);
    }
  }
};

void geometry_costs(const Detection& reference, const StageGeometry& stage, std::size_t begin,
                    std::size_t count, float* ious, float* distances) {
#if defined(__aarch64__)
  if (count == 4) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t ax1 = vdupq_n_f32(reference.x1);
    const float32x4_t ay1 = vdupq_n_f32(reference.y1);
    const float32x4_t ax2 = vdupq_n_f32(reference.x2);
    const float32x4_t ay2 = vdupq_n_f32(reference.y2);
    const float32x4_t bx1 = vld1q_f32(stage.x1.data() + begin);
    const float32x4_t by1 = vld1q_f32(stage.y1.data() + begin);
    const float32x4_t bx2 = vld1q_f32(stage.x2.data() + begin);
    const float32x4_t by2 = vld1q_f32(stage.y2.data() + begin);

    const float32x4_t aw = vmaxq_f32(zero, vsubq_f32(ax2, ax1));
    const float32x4_t ah = vmaxq_f32(zero, vsubq_f32(ay2, ay1));
    const float32x4_t bw = vmaxq_f32(zero, vsubq_f32(bx2, bx1));
    const float32x4_t bh = vmaxq_f32(zero, vsubq_f32(by2, by1));
    const float32x4_t iw = vmaxq_f32(zero, vsubq_f32(vminq_f32(ax2, bx2), vmaxq_f32(ax1, bx1)));
    const float32x4_t ih = vmaxq_f32(zero, vsubq_f32(vminq_f32(ay2, by2), vmaxq_f32(ay1, by1)));
    const float32x4_t intersection = vmulq_f32(iw, ih);
    const float32x4_t union_area =
        vsubq_f32(vaddq_f32(vmulq_f32(aw, ah), vmulq_f32(bw, bh)), intersection);
    const uint32x4_t valid_union = vcgtq_f32(union_area, zero);
    const float32x4_t safe_union = vbslq_f32(valid_union, union_area, one);
    vst1q_f32(ious, vbslq_f32(valid_union, vdivq_f32(intersection, safe_union), zero));

    const float32x4_t dx =
        vsubq_f32(vmulq_f32(vaddq_f32(ax1, ax2), half), vmulq_f32(vaddq_f32(bx1, bx2), half));
    const float32x4_t dy =
        vsubq_f32(vmulq_f32(vaddq_f32(ay1, ay2), half), vmulq_f32(vaddq_f32(by1, by2), half));
    const float32x4_t diagonal_a = vsqrtq_f32(vaddq_f32(vmulq_f32(aw, aw), vmulq_f32(ah, ah)));
    const float32x4_t diagonal_b = vsqrtq_f32(vaddq_f32(vmulq_f32(bw, bw), vmulq_f32(bh, bh)));
    const float32x4_t scale = vmaxq_f32(one, vmulq_f32(vaddq_f32(diagonal_a, diagonal_b), half));
    const float32x4_t distance =
        vdivq_f32(vsqrtq_f32(vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy))), scale);
    vst1q_f32(distances, distance);
    return;
  }
#endif
  for (std::size_t lane = 0; lane < count; ++lane) {
    const std::size_t index = begin + lane;
    const Detection detection{
        stage.x1[index],      stage.y1[index], stage.x2[index], stage.y2[index], 0.0f,
        stage.class_id[index]};
    ious[lane] = iou_xyxy(reference, detection);
    distances[lane] = normalized_center_distance(reference, detection);
  }
}

void solve_assignment(const std::vector<float>& costs, int rows, int columns, std::vector<float>& u,
                      std::vector<float>& v, std::vector<float>& minv, std::vector<int>& p,
                      std::vector<int>& way, std::vector<int>& row_to_column,
                      std::vector<unsigned char>& used) {
  row_to_column.assign(static_cast<std::size_t>(rows), -1);
  if (rows == 0) {
    return;
  }
  u.assign(static_cast<std::size_t>(rows + 1), 0.0f);
  v.assign(static_cast<std::size_t>(columns + 1), 0.0f);
  p.assign(static_cast<std::size_t>(columns + 1), 0);
  way.resize(static_cast<std::size_t>(columns + 1));
  minv.resize(static_cast<std::size_t>(columns + 1));
  used.resize(static_cast<std::size_t>(columns + 1));

  for (int row = 1; row <= rows; ++row) {
    p[0] = row;
    int column0 = 0;
    std::fill(minv.begin(), minv.end(), std::numeric_limits<float>::infinity());
    std::fill(used.begin(), used.end(), 0);
    do {
      used[static_cast<std::size_t>(column0)] = 1;
      const int row0 = p[static_cast<std::size_t>(column0)];
      float delta = std::numeric_limits<float>::infinity();
      int column1 = 0;
      for (int column = 1; column <= columns; ++column) {
        if (used[static_cast<std::size_t>(column)] != 0) {
          continue;
        }
        const float current = costs[static_cast<std::size_t>((row0 - 1) * columns + column - 1)] -
                              u[static_cast<std::size_t>(row0)] -
                              v[static_cast<std::size_t>(column)];
        if (current < minv[static_cast<std::size_t>(column)]) {
          minv[static_cast<std::size_t>(column)] = current;
          way[static_cast<std::size_t>(column)] = column0;
        }
        if (minv[static_cast<std::size_t>(column)] < delta) {
          delta = minv[static_cast<std::size_t>(column)];
          column1 = column;
        }
      }
      for (int column = 0; column <= columns; ++column) {
        if (used[static_cast<std::size_t>(column)] != 0) {
          u[static_cast<std::size_t>(p[static_cast<std::size_t>(column)])] += delta;
          v[static_cast<std::size_t>(column)] -= delta;
        } else {
          minv[static_cast<std::size_t>(column)] -= delta;
        }
      }
      column0 = column1;
    } while (p[static_cast<std::size_t>(column0)] != 0);

    do {
      const int column1 = way[static_cast<std::size_t>(column0)];
      p[static_cast<std::size_t>(column0)] = p[static_cast<std::size_t>(column1)];
      column0 = column1;
    } while (column0 != 0);
  }
  for (int column = 1; column <= columns; ++column) {
    if (p[static_cast<std::size_t>(column)] != 0) {
      row_to_column[static_cast<std::size_t>(p[static_cast<std::size_t>(column)] - 1)] = column - 1;
    }
  }
}

} // namespace

struct ObjectTracker::Impl {
  std::vector<TrackState> tracks;
  std::vector<int> high_detections;
  std::vector<int> low_detections;
  std::vector<int> track_indices;
  std::vector<int> assignments;
  std::vector<unsigned char> matched_tracks;
  std::vector<unsigned char> matched_detections;
  StageGeometry stage;
  std::vector<float> costs;
  std::vector<float> u;
  std::vector<float> v;
  std::vector<float> minv;
  std::vector<int> p;
  std::vector<int> way;
  std::vector<int> row_to_column;
  std::vector<unsigned char> used;
  int last_frame_index = -1;
};

ObjectTracker::ObjectTracker(TrackerConfig config)
    : config_(config), impl_(std::make_unique<Impl>()) {
  validate_config(config_);
}

ObjectTracker::~ObjectTracker() = default;
ObjectTracker::ObjectTracker(ObjectTracker&&) noexcept = default;
ObjectTracker& ObjectTracker::operator=(ObjectTracker&&) noexcept = default;

int ObjectTracker::active_track_count() const {
  return static_cast<int>(impl_->tracks.size());
}

std::vector<TrackedDetection> ObjectTracker::update(const std::vector<Detection>& detections,
                                                    int frame_index) {
  std::vector<TrackedDetection> tracked;
  update_into(detections, frame_index, tracked);
  return tracked;
}

void ObjectTracker::update_into(const std::vector<Detection>& detections, int frame_index,
                                std::vector<TrackedDetection>& tracked) {
  if (frame_index < 0) {
    throw std::invalid_argument("frame_index must be >= 0");
  }
  if (frame_index < impl_->last_frame_index) {
    throw std::invalid_argument("frame_index must be monotonic");
  }
  impl_->last_frame_index = frame_index;

  std::erase_if(impl_->tracks, [&](const TrackState& track) {
    const int intervening_frames = std::max(0, frame_index - track.last_frame_index - 1);
    return (!track.confirmed && intervening_frames > 0) ||
           intervening_frames > config_.max_missing_frames;
  });

  impl_->high_detections.clear();
  impl_->low_detections.clear();
  for (std::size_t index = 0; index < detections.size(); ++index) {
    auto& stage = detections[index].score >= config_.high_score_threshold ? impl_->high_detections
                                                                          : impl_->low_detections;
    stage.push_back(static_cast<int>(index));
  }

  impl_->matched_tracks.assign(impl_->tracks.size(), 0);
  impl_->matched_detections.assign(detections.size(), 0);
  impl_->assignments.assign(detections.size(), -1);

  const auto associate = [&](const std::vector<int>& detection_indices) {
    impl_->track_indices.clear();
    for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
      if (impl_->matched_tracks[index] == 0) {
        impl_->track_indices.push_back(static_cast<int>(index));
      }
    }
    impl_->stage.prepare(detections, detection_indices, impl_->matched_detections);
    const int rows = static_cast<int>(impl_->track_indices.size());
    const int detection_columns = static_cast<int>(impl_->stage.detection_index.size());
    if (rows == 0 || detection_columns == 0) {
      return;
    }

    const int columns = detection_columns + rows;
    impl_->costs.assign(static_cast<std::size_t>(rows * columns), kUnmatchedCost);
    for (int row = 0; row < rows; ++row) {
      const int track_index = impl_->track_indices[static_cast<std::size_t>(row)];
      const TrackState& track = impl_->tracks[static_cast<std::size_t>(track_index)];
      const Detection reference =
          config_.center_distance_enabled ? predict(track, frame_index) : track.detection;
      for (int column = 0; column < detection_columns; column += 4) {
        const int count = std::min(4, detection_columns - column);
        float ious[4]{};
        float distances[4]{};
        geometry_costs(reference, impl_->stage, static_cast<std::size_t>(column),
                       static_cast<std::size_t>(count), ious, distances);
        for (int lane = 0; lane < count; ++lane) {
          const int candidate_column = column + lane;
          if (impl_->stage.class_id[static_cast<std::size_t>(candidate_column)] !=
              track.detection.class_id) {
            impl_->costs[static_cast<std::size_t>(row * columns + candidate_column)] = kBlockedCost;
            continue;
          }
          const bool center_match =
              config_.center_distance_enabled && distances[lane] <= config_.max_center_distance;
          if (ious[lane] < config_.match_iou_threshold && !center_match) {
            impl_->costs[static_cast<std::size_t>(row * columns + candidate_column)] = kBlockedCost;
            continue;
          }
          const float affinity = config_.center_distance_enabled
                                     ? ious[lane] + 1.0f / (1.0f + distances[lane])
                                     : ious[lane];
          impl_->costs[static_cast<std::size_t>(row * columns + candidate_column)] =
              config_.center_distance_enabled ? 2.0f - affinity : 1.0f - affinity;
        }
      }
    }

    solve_assignment(impl_->costs, rows, columns, impl_->u, impl_->v, impl_->minv, impl_->p,
                     impl_->way, impl_->row_to_column, impl_->used);
    for (int row = 0; row < rows; ++row) {
      const int column = impl_->row_to_column[static_cast<std::size_t>(row)];
      if (column < 0 || column >= detection_columns ||
          impl_->costs[static_cast<std::size_t>(row * columns + column)] >= kBlockedCost) {
        continue;
      }
      const int track_index = impl_->track_indices[static_cast<std::size_t>(row)];
      const int detection_index = impl_->stage.detection_index[static_cast<std::size_t>(column)];
      impl_->matched_tracks[static_cast<std::size_t>(track_index)] = 1;
      impl_->matched_detections[static_cast<std::size_t>(detection_index)] = 1;
      impl_->assignments[static_cast<std::size_t>(detection_index)] = track_index;
    }
  };

  // ByteTrack-style staging lets low-score detections recover established
  // tracks, while only high-score detections may create identities.
  associate(impl_->high_detections);
  associate(impl_->low_detections);

  for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
    const int track_index = impl_->assignments[detection_index];
    if (track_index < 0) {
      continue;
    }
    TrackState& track = impl_->tracks[static_cast<std::size_t>(track_index)];
    const Detection& detection = detections[detection_index];
    const int elapsed = std::max(1, frame_index - track.last_frame_index);
    const float measured_vx = (center_x(detection) - center_x(track.detection)) / elapsed;
    const float measured_vy = (center_y(detection) - center_y(track.detection)) / elapsed;
    const float measured_vw = (width(detection) - width(track.detection)) / elapsed;
    const float measured_vh = (height(detection) - height(track.detection)) / elapsed;
    const float previous = config_.velocity_momentum;
    const float measured = 1.0f - previous;
    track.velocity_x = previous * track.velocity_x + measured * measured_vx;
    track.velocity_y = previous * track.velocity_y + measured * measured_vy;
    track.velocity_w = previous * track.velocity_w + measured * measured_vw;
    track.velocity_h = previous * track.velocity_h + measured * measured_vh;
    track.detection = detection;
    track.last_frame_index = frame_index;
    track.missing_frames = 0;
    ++track.hits;
    track.confirmed = track.hits >= config_.min_confirmed_hits;
  }

  for (const int detection_index : impl_->high_detections) {
    if (impl_->matched_detections[static_cast<std::size_t>(detection_index)] != 0 ||
        detections[static_cast<std::size_t>(detection_index)].score < config_.new_track_threshold) {
      continue;
    }
    const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
    const int track_index = static_cast<int>(impl_->tracks.size());
    impl_->tracks.push_back(TrackState{next_track_id_++, detection, 0.0f, 0.0f, 0.0f, 0.0f,
                                       frame_index, 0, 1, config_.min_confirmed_hits <= 1});
    impl_->matched_tracks.push_back(1);
    impl_->matched_detections[static_cast<std::size_t>(detection_index)] = 1;
    impl_->assignments[static_cast<std::size_t>(detection_index)] = track_index;
  }

  for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
    if (impl_->matched_tracks[index] == 0) {
      impl_->tracks[index].missing_frames = frame_index - impl_->tracks[index].last_frame_index;
    }
  }

  tracked.clear();
  tracked.reserve(detections.size() + impl_->tracks.size());
  for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
    const int track_index = impl_->assignments[detection_index];
    if (track_index < 0) {
      continue;
    }
    const TrackState& track = impl_->tracks[static_cast<std::size_t>(track_index)];
    if (!track.confirmed) {
      continue;
    }
    const Detection& detection = detections[detection_index];
    tracked.push_back(TrackedDetection{track.track_id, detection.x1, detection.y1, detection.x2,
                                       detection.y2, detection.score, detection.class_id, false});
  }

  if (config_.max_prediction_frames > 0) {
    for (const auto& track : impl_->tracks) {
      if (!track.confirmed || track.missing_frames <= 0 ||
          track.missing_frames > config_.max_prediction_frames ||
          track.detection.score < config_.high_score_threshold) {
        continue;
      }
      Detection prediction = predict(track, frame_index);
      const bool duplicate = std::any_of(tracked.begin(), tracked.end(), [&](const auto& output) {
        const Detection observed{output.x1, output.y1,    output.x2,
                                 output.y2, output.score, output.class_id};
        return observed.class_id == prediction.class_id && iou_xyxy(observed, prediction) > 0.5f;
      });
      if (duplicate) {
        continue;
      }
      prediction.score *= std::pow(0.9f, static_cast<float>(track.missing_frames));
      tracked.push_back(TrackedDetection{track.track_id, prediction.x1, prediction.y1,
                                         prediction.x2, prediction.y2, prediction.score,
                                         prediction.class_id, true});
    }
  }

  std::erase_if(impl_->tracks, [&](const TrackState& track) {
    return (!track.confirmed && track.missing_frames > 0) ||
           track.missing_frames > config_.max_missing_frames;
  });
}

} // namespace multi_stream_people_tracker
