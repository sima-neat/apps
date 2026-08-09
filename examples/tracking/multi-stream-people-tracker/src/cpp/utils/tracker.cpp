#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
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
  Detection filtered_detection;
  float velocity_x = 0.0f;
  float velocity_y = 0.0f;
  float velocity_w = 0.0f;
  float velocity_h = 0.0f;
  int last_frame_index = 0;
  int missing_frames = 0;
  int hits = 1;
  bool confirmed = true;
};

struct MotionPair {
  float distance = 0.0f;
  int track_index = -1;
  int detection_index = -1;
};

struct MotionVote {
  float x = 0.0f;
  float y = 0.0f;
  int bin_x = 0;
  int bin_y = 0;
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
  const float predicted_x = center_x(track.filtered_detection) + track.velocity_x * elapsed;
  const float predicted_y = center_y(track.filtered_detection) + track.velocity_y * elapsed;
  const float predicted_w =
      std::max(1.0f, width(track.filtered_detection) + track.velocity_w * elapsed);
  const float predicted_h =
      std::max(1.0f, height(track.filtered_detection) + track.velocity_h * elapsed);
  return Detection{predicted_x - predicted_w * 0.5f,
                   predicted_y - predicted_h * 0.5f,
                   predicted_x + predicted_w * 0.5f,
                   predicted_y + predicted_h * 0.5f,
                   track.detection.score,
                   track.detection.class_id};
}

Detection translate(Detection detection, float x, float y) {
  detection.x1 += x;
  detection.x2 += x;
  detection.y1 += y;
  detection.y2 += y;
  return detection;
}

bool valid_camera_transform(const CameraTransform& transform) {
  if (!transform.valid || !std::isfinite(transform.a) || !std::isfinite(transform.b) ||
      !std::isfinite(transform.tx) || !std::isfinite(transform.c) || !std::isfinite(transform.d) ||
      !std::isfinite(transform.ty)) {
    return false;
  }
  const float determinant = transform.a * transform.d - transform.b * transform.c;
  return std::abs(determinant) > 0.01f;
}

Detection transform_detection(const Detection& detection, const CameraTransform& transform) {
  const auto transform_point = [&](float x, float y) {
    return std::pair{transform.a * x + transform.b * y + transform.tx,
                     transform.c * x + transform.d * y + transform.ty};
  };
  const auto p0 = transform_point(detection.x1, detection.y1);
  const auto p1 = transform_point(detection.x2, detection.y1);
  const auto p2 = transform_point(detection.x1, detection.y2);
  const auto p3 = transform_point(detection.x2, detection.y2);
  return Detection{std::min({p0.first, p1.first, p2.first, p3.first}),
                   std::min({p0.second, p1.second, p2.second, p3.second}),
                   std::max({p0.first, p1.first, p2.first, p3.first}),
                   std::max({p0.second, p1.second, p2.second, p3.second}),
                   detection.score,
                   detection.class_id};
}

void transform_track_state(TrackState& track, const CameraTransform& transform) {
  track.detection = transform_detection(track.detection, transform);
  track.filtered_detection = transform_detection(track.filtered_detection, transform);
  const float velocity_x = transform.a * track.velocity_x + transform.b * track.velocity_y;
  const float velocity_y = transform.c * track.velocity_x + transform.d * track.velocity_y;
  track.velocity_x = velocity_x;
  track.velocity_y = velocity_y;
  track.velocity_w *= std::hypot(transform.a, transform.c);
  track.velocity_h *= std::hypot(transform.b, transform.d);
}

float median(std::vector<float>& values) {
  if (values.empty()) {
    return 0.0f;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5f * (values[middle - 1] + values[middle]) : values[middle];
}

Detection smooth_detection(const Detection& prediction, const Detection& observation,
                           float observation_alpha) {
  const float predicted = 1.0f - observation_alpha;
  const float smoothed_x =
      predicted * center_x(prediction) + observation_alpha * center_x(observation);
  const float smoothed_y =
      predicted * center_y(prediction) + observation_alpha * center_y(observation);
  const float smoothed_w =
      std::max(1.0f, predicted * width(prediction) + observation_alpha * width(observation));
  const float smoothed_h =
      std::max(1.0f, predicted * height(prediction) + observation_alpha * height(observation));
  return Detection{smoothed_x - smoothed_w * 0.5f,
                   smoothed_y - smoothed_h * 0.5f,
                   smoothed_x + smoothed_w * 0.5f,
                   smoothed_y + smoothed_h * 0.5f,
                   observation.score,
                   observation.class_id};
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
  if (!std::isfinite(config.box_smoothing_alpha) || config.box_smoothing_alpha <= 0.0f ||
      config.box_smoothing_alpha > 1.0f) {
    throw std::invalid_argument("box_smoothing_alpha must be in (0, 1]");
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
  std::vector<int> motion_tracks;
  std::vector<int> motion_detections;
  std::vector<float> motion_values_x;
  std::vector<float> motion_values_y;
  std::vector<float> motion_scales;
  std::vector<float> motion_offsets_x;
  std::vector<float> motion_offsets_y;
  std::vector<MotionVote> motion_votes;
  std::vector<MotionPair> motion_pairs;
  std::unordered_map<std::uint64_t, int> motion_histogram;
  std::vector<unsigned char> motion_track_used;
  std::vector<unsigned char> motion_detection_used;
  int last_frame_index = -1;
};

template <typename ImplType>
CameraTransform estimate_camera_motion(ImplType& impl, const std::vector<Detection>& detections,
                                       const TrackerConfig& config, int frame_index) {
  constexpr std::size_t kMinimumObjects = 3;

  impl.motion_tracks.clear();
  impl.motion_detections.clear();
  impl.motion_scales.clear();
  for (std::size_t index = 0; index < impl.tracks.size(); ++index) {
    const auto& track = impl.tracks[index];
    if (track.last_frame_index != frame_index - 1) {
      continue;
    }
    const Detection prediction = predict(track, frame_index);
    impl.motion_tracks.push_back(static_cast<int>(index));
    impl.motion_scales.push_back(std::hypot(width(prediction), height(prediction)));
  }
  const std::size_t track_count = impl.motion_tracks.size();

  for (std::size_t index = 0; index < detections.size(); ++index) {
    const auto& detection = detections[index];
    if (detection.score < config.high_score_threshold) {
      continue;
    }
    impl.motion_detections.push_back(static_cast<int>(index));
    impl.motion_scales.push_back(std::hypot(width(detection), height(detection)));
  }
  const std::size_t detection_count = impl.motion_detections.size();
  if (track_count < kMinimumObjects || detection_count < kMinimumObjects) {
    return {};
  }

  const float typical_diagonal = std::max(1.0f, median(impl.motion_scales));
  const float maximum_shift = std::max(16.0f, 4.0f * typical_diagonal);
  const float bin_size = std::max(2.0f, 0.25f * typical_diagonal);
  const auto bin_key = [](int x, int y) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32U) |
           static_cast<std::uint32_t>(y);
  };

  // Every correct correspondence votes for nearly the same translation;
  // incorrect cross-object pairs spread across the displacement plane. A
  // compact 3x3-bin consensus is robust to individual drone motion, detector
  // dropouts, and boxes entering or leaving the image.
  impl.motion_votes.clear();
  impl.motion_histogram.clear();
  for (const int track_index : impl.motion_tracks) {
    const auto& track = impl.tracks[static_cast<std::size_t>(track_index)];
    const Detection prediction = predict(track, frame_index);
    for (const int detection_index : impl.motion_detections) {
      const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
      if (detection.class_id != track.detection.class_id) {
        continue;
      }
      const float x = center_x(detection) - center_x(prediction);
      const float y = center_y(detection) - center_y(prediction);
      if (std::hypot(x, y) > maximum_shift) {
        continue;
      }
      const int bin_x = static_cast<int>(std::floor(x / bin_size));
      const int bin_y = static_cast<int>(std::floor(y / bin_size));
      impl.motion_votes.push_back(MotionVote{x, y, bin_x, bin_y});
      ++impl.motion_histogram[bin_key(bin_x, bin_y)];
    }
  }
  if (impl.motion_votes.size() < kMinimumObjects) {
    return {};
  }

  int best_x = 0;
  int best_y = 0;
  int best_support = 0;
  for (const auto& [key, count] : impl.motion_histogram) {
    (void)count;
    const int bin_x = static_cast<std::int32_t>(key >> 32U);
    const int bin_y = static_cast<std::int32_t>(key & 0xffffffffU);
    int support = 0;
    for (int offset_x = -1; offset_x <= 1; ++offset_x) {
      for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        const auto found = impl.motion_histogram.find(bin_key(bin_x + offset_x, bin_y + offset_y));
        if (found != impl.motion_histogram.end()) {
          support += found->second;
        }
      }
    }
    const int squared_norm = bin_x * bin_x + bin_y * bin_y;
    const int best_squared_norm = best_x * best_x + best_y * best_y;
    if (support > best_support ||
        (support == best_support && (squared_norm < best_squared_norm ||
                                     (squared_norm == best_squared_norm &&
                                      std::pair{bin_x, bin_y} < std::pair{best_x, best_y})))) {
      best_support = support;
      best_x = bin_x;
      best_y = bin_y;
    }
  }

  impl.motion_values_x.clear();
  impl.motion_values_y.clear();
  for (const auto& vote : impl.motion_votes) {
    if (std::abs(vote.bin_x - best_x) <= 1 && std::abs(vote.bin_y - best_y) <= 1) {
      impl.motion_values_x.push_back(vote.x);
      impl.motion_values_y.push_back(vote.y);
    }
  }
  const float initial_x = median(impl.motion_values_x);
  const float initial_y = median(impl.motion_values_y);

  // Validate the consensus using unique pairs, then refine the translation
  // from only those correspondences.
  impl.motion_pairs.clear();
  const float candidate_gate = std::max(0.75f, config.max_center_distance);
  for (const int track_index : impl.motion_tracks) {
    const auto& track = impl.tracks[static_cast<std::size_t>(track_index)];
    const Detection reference = translate(predict(track, frame_index), initial_x, initial_y);
    for (const int detection_index : impl.motion_detections) {
      const auto& detection = detections[static_cast<std::size_t>(detection_index)];
      if (detection.class_id != track.detection.class_id) {
        continue;
      }
      const float distance = normalized_center_distance(reference, detection);
      if (distance <= candidate_gate) {
        impl.motion_pairs.push_back(MotionPair{distance, track_index, detection_index});
      }
    }
  }
  std::sort(impl.motion_pairs.begin(), impl.motion_pairs.end(),
            [](const MotionPair& lhs, const MotionPair& rhs) {
              if (lhs.distance != rhs.distance) {
                return lhs.distance < rhs.distance;
              }
              if (lhs.track_index != rhs.track_index) {
                return lhs.track_index < rhs.track_index;
              }
              return lhs.detection_index < rhs.detection_index;
            });

  impl.motion_track_used.assign(impl.tracks.size(), 0);
  impl.motion_detection_used.assign(detections.size(), 0);
  impl.motion_offsets_x.clear();
  impl.motion_offsets_y.clear();
  for (const auto& pair : impl.motion_pairs) {
    if (impl.motion_track_used[static_cast<std::size_t>(pair.track_index)] != 0 ||
        impl.motion_detection_used[static_cast<std::size_t>(pair.detection_index)] != 0) {
      continue;
    }
    const Detection prediction =
        predict(impl.tracks[static_cast<std::size_t>(pair.track_index)], frame_index);
    const Detection& detection = detections[static_cast<std::size_t>(pair.detection_index)];
    impl.motion_track_used[static_cast<std::size_t>(pair.track_index)] = 1;
    impl.motion_detection_used[static_cast<std::size_t>(pair.detection_index)] = 1;
    impl.motion_offsets_x.push_back(center_x(detection) - center_x(prediction));
    impl.motion_offsets_y.push_back(center_y(detection) - center_y(prediction));
  }

  const std::size_t possible = std::min(track_count, detection_count);
  const std::size_t required = std::max(kMinimumObjects, (possible + 2) / 3);
  if (impl.motion_offsets_x.size() < required) {
    return {};
  }
  return CameraTransform{
      1.0f, 0.0f, median(impl.motion_offsets_x), 0.0f, 1.0f, median(impl.motion_offsets_y), true};
}

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
                                                    int frame_index,
                                                    const CameraTransform& camera_transform) {
  std::vector<TrackedDetection> tracked;
  update_into(detections, frame_index, tracked, camera_transform);
  return tracked;
}

void ObjectTracker::update_into(const std::vector<Detection>& detections, int frame_index,
                                std::vector<TrackedDetection>& tracked,
                                const CameraTransform& camera_transform) {
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

  CameraTransform camera_motion;
  if (config_.camera_motion_compensation) {
    camera_motion = valid_camera_transform(camera_transform)
                        ? camera_transform
                        : estimate_camera_motion(*impl_, detections, config_, frame_index);
  }
  if (valid_camera_transform(camera_motion)) {
    for (auto& track : impl_->tracks) {
      transform_track_state(track, camera_motion);
    }
  } else {
    camera_motion = {};
  }
  impl_->matched_tracks.assign(impl_->tracks.size(), 0);
  impl_->matched_detections.assign(detections.size(), 0);
  impl_->assignments.assign(detections.size(), -1);

  const auto associate_group = [&](const std::vector<int>& detection_indices, std::size_t begin,
                                   std::size_t end) {
    if (begin == end) {
      return;
    }
    impl_->stage.prepare(detections, detection_indices, impl_->matched_detections);
    const int rows = static_cast<int>(end - begin);
    const int detection_columns = static_cast<int>(impl_->stage.detection_index.size());
    if (detection_columns == 0) {
      return;
    }

    const int columns = detection_columns + rows;
    impl_->costs.assign(static_cast<std::size_t>(rows * columns), kUnmatchedCost);
    for (int row = 0; row < rows; ++row) {
      const int track_index = impl_->track_indices[begin + static_cast<std::size_t>(row)];
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
      const int track_index = impl_->track_indices[begin + static_cast<std::size_t>(row)];
      const int detection_index = impl_->stage.detection_index[static_cast<std::size_t>(column)];
      impl_->matched_tracks[static_cast<std::size_t>(track_index)] = 1;
      impl_->matched_detections[static_cast<std::size_t>(detection_index)] = 1;
      impl_->assignments[static_cast<std::size_t>(detection_index)] = track_index;
    }
  };

  const auto associate = [&](const std::vector<int>& detection_indices) {
    impl_->track_indices.clear();
    for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
      if (impl_->matched_tracks[index] == 0) {
        impl_->track_indices.push_back(static_cast<int>(index));
      }
    }
    std::sort(impl_->track_indices.begin(), impl_->track_indices.end(), [&](int a, int b) {
      const TrackState& lhs = impl_->tracks[static_cast<std::size_t>(a)];
      const TrackState& rhs = impl_->tracks[static_cast<std::size_t>(b)];
      if (lhs.last_frame_index != rhs.last_frame_index) {
        return lhs.last_frame_index > rhs.last_frame_index;
      }
      return lhs.track_id < rhs.track_id;
    });
    std::size_t begin = 0;
    while (begin < impl_->track_indices.size()) {
      const int last_frame =
          impl_->tracks[static_cast<std::size_t>(impl_->track_indices[begin])].last_frame_index;
      std::size_t end = begin + 1;
      while (end < impl_->track_indices.size() &&
             impl_->tracks[static_cast<std::size_t>(impl_->track_indices[end])].last_frame_index ==
                 last_frame) {
        ++end;
      }
      associate_group(detection_indices, begin, end);
      begin = end;
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
    const Detection filtered_prediction = predict(track, frame_index);
    const float previous = config_.velocity_momentum;
    const float measured = 1.0f - previous;
    track.velocity_x = previous * track.velocity_x + measured * measured_vx;
    track.velocity_y = previous * track.velocity_y + measured * measured_vy;
    track.velocity_w = previous * track.velocity_w + measured * measured_vw;
    track.velocity_h = previous * track.velocity_h + measured * measured_vh;
    float smoothing_alpha = config_.box_smoothing_alpha;
    if (camera_motion.valid && smoothing_alpha < 1.0f) {
      const float innovation = normalized_center_distance(filtered_prediction, detection);
      const float response =
          std::min(1.0f, innovation / std::max(0.01f, config_.max_center_distance));
      smoothing_alpha += (1.0f - smoothing_alpha) * response;
    }
    track.filtered_detection = smooth_detection(filtered_prediction, detection, smoothing_alpha);
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
    impl_->tracks.push_back(TrackState{next_track_id_++, detection, detection, 0.0f, 0.0f, 0.0f,
                                       0.0f, frame_index, 0, 1, config_.min_confirmed_hits <= 1});
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
    const Detection& filtered = track.filtered_detection;
    const Detection& detection = detections[detection_index];
    tracked.push_back(TrackedDetection{track.track_id, filtered.x1, filtered.y1, filtered.x2,
                                       filtered.y2, detection.score, detection.class_id, false});
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
