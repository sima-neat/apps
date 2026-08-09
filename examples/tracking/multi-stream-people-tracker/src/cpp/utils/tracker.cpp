#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
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
constexpr float kMaximumLogSize = 9.70406055f; // log(16384)
constexpr float kMaximumLogSizeVelocity = 0.35f;
constexpr float kMaximumLogSizeInnovationPerFrame = 0.70f;

using BoxCorners = std::array<float, 8>;

struct KalmanAxis {
  float position = 0.0f;
  float velocity = 0.0f;
  float p00 = 4.0f;
  float p01 = 0.0f;
  float p10 = 0.0f;
  float p11 = 16.0f;
};

struct TrackState {
  int track_id = 0;
  Detection detection;
  Detection filtered_detection;
  Detection display_detection;
  BoxCorners detection_corners{};
  BoxCorners filtered_detection_corners{};
  BoxCorners display_detection_corners{};
  KalmanAxis center_x_filter;
  KalmanAxis center_y_filter;
  KalmanAxis log_width_filter;
  KalmanAxis log_height_filter;
  float legacy_velocity_x = 0.0f;
  float legacy_velocity_y = 0.0f;
  float legacy_velocity_w = 0.0f;
  float legacy_velocity_h = 0.0f;
  bool covariance_motion_enabled = true;
  float previous_observation_x = 0.0f;
  float previous_observation_y = 0.0f;
  int previous_observation_frame = -1;
  int last_frame_index = 0;
  int state_frame_index = 0;
  int missing_frames = 0;
  int hits = 1;
  bool confirmed = true;
  bool occluded = false;
  float association_confidence = 1.0f;
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

float log_size(float value) {
  return std::clamp(std::log(std::max(1.0f, value)), 0.0f, kMaximumLogSize);
}

float size_from_log(float value) {
  return std::exp(std::clamp(value, 0.0f, kMaximumLogSize));
}

void stabilize_size_axis(KalmanAxis& axis) {
  axis.position = std::clamp(axis.position, 0.0f, kMaximumLogSize);
  axis.velocity = std::clamp(axis.velocity, -kMaximumLogSizeVelocity, kMaximumLogSizeVelocity);
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

float overlap_coefficient(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  const float smaller_area = std::min(width(a) * height(a), width(b) * height(b));
  return smaller_area > 0.0f ? intersection / smaller_area : 0.0f;
}

bool swept_boxes_may_overlap(const TrackState& first, const Detection& first_prediction,
                             const TrackState& second, const Detection& second_prediction) {
  const float previous_dx = center_x(first.detection) - center_x(second.detection);
  const float previous_dy = center_y(first.detection) - center_y(second.detection);
  const float current_dx = center_x(first_prediction) - center_x(second_prediction);
  const float current_dy = center_y(first_prediction) - center_y(second_prediction);
  const float delta_x = current_dx - previous_dx;
  const float delta_y = current_dy - previous_dy;
  const float delta_squared = delta_x * delta_x + delta_y * delta_y;
  const float closest_time =
      delta_squared > 1.0e-6f
          ? std::clamp(-(previous_dx * delta_x + previous_dy * delta_y) / delta_squared, 0.0f, 1.0f)
          : 0.0f;
  const float closest_dx = previous_dx + closest_time * delta_x;
  const float closest_dy = previous_dy + closest_time * delta_y;
  const float collision_radius =
      0.35f * (std::hypot(width(first_prediction), height(first_prediction)) +
               std::hypot(width(second_prediction), height(second_prediction)));
  return std::hypot(closest_dx, closest_dy) <= std::max(1.0f, collision_radius);
}

void predict_axis(KalmanAxis& axis, float elapsed, float position_noise, float velocity_noise) {
  if (elapsed <= 0.0f) {
    return;
  }
  axis.position += elapsed * axis.velocity;
  const float p00 = axis.p00 + elapsed * (axis.p01 + axis.p10) + elapsed * elapsed * axis.p11 +
                    position_noise * elapsed;
  const float p01 = axis.p01 + elapsed * axis.p11;
  const float p10 = axis.p10 + elapsed * axis.p11;
  const float p11 = axis.p11 + velocity_noise * elapsed;
  axis.p00 = std::max(1.0e-4f, p00);
  axis.p01 = p01;
  axis.p10 = p10;
  axis.p11 = std::max(1.0e-4f, p11);
}

void update_axis(KalmanAxis& axis, float measurement, float measurement_variance) {
  const float variance = std::max(1.0e-4f, measurement_variance);
  const float innovation_variance = std::max(1.0e-4f, axis.p00 + variance);
  const float gain_position = axis.p00 / innovation_variance;
  const float gain_velocity = axis.p10 / innovation_variance;
  const float innovation = measurement - axis.position;
  axis.position += gain_position * innovation;
  axis.velocity += gain_velocity * innovation;

  // Joseph-form covariance update for H=[1,0]. This costs a few more
  // multiplies than the simplified form but keeps long board runs positive
  // semidefinite under FP32 rounding.
  const float one_minus_gain = 1.0f - gain_position;
  const float p00 =
      one_minus_gain * one_minus_gain * axis.p00 + gain_position * gain_position * variance;
  const float p01 = one_minus_gain * (axis.p01 - gain_velocity * axis.p00) +
                    gain_position * gain_velocity * variance;
  const float p10 = one_minus_gain * (axis.p10 - gain_velocity * axis.p00) +
                    gain_position * gain_velocity * variance;
  const float p11 = axis.p11 - gain_velocity * axis.p01 - gain_velocity * axis.p10 +
                    gain_velocity * gain_velocity * (axis.p00 + variance);
  axis.p00 = std::clamp(p00, 1.0e-4f, 1.0e6f);
  axis.p01 = std::clamp(p01, -1.0e6f, 1.0e6f);
  axis.p10 = std::clamp(p10, -1.0e6f, 1.0e6f);
  axis.p11 = std::clamp(p11, 1.0e-4f, 1.0e6f);
}

Detection filter_box(const TrackState& track) {
  const float filtered_width = size_from_log(track.log_width_filter.position);
  const float filtered_height = size_from_log(track.log_height_filter.position);
  return Detection{track.center_x_filter.position - 0.5f * filtered_width,
                   track.center_y_filter.position - 0.5f * filtered_height,
                   track.center_x_filter.position + 0.5f * filtered_width,
                   track.center_y_filter.position + 0.5f * filtered_height,
                   track.detection.score,
                   track.detection.class_id};
}

BoxCorners box_corners(const Detection& detection);

void freeze_unobserved_size(TrackState& track) {
  // A missed/ambiguous observation contains no evidence that an object's
  // apparent size changed.  Keep coasting the center, but anchor scale to the
  // last reliable filtered display state (already transformed by GMC).
  // Using that robust state instead of the raw detector box also prevents a
  // single scale outlier from becoming the frozen prediction. Extrapolating a
  // noisy one-pixel size derivative is especially unstable for tiny objects:
  // even a bounded log-size velocity compounds exponentially over a gap.
  track.log_width_filter.position = log_size(width(track.display_detection));
  track.log_width_filter.velocity = 0.0f;
  track.log_width_filter.p01 = 0.0f;
  track.log_width_filter.p10 = 0.0f;
  track.log_height_filter.position = log_size(height(track.display_detection));
  track.log_height_filter.velocity = 0.0f;
  track.log_height_filter.p01 = 0.0f;
  track.log_height_filter.p10 = 0.0f;
  track.filtered_detection = filter_box(track);
  track.filtered_detection_corners = box_corners(track.filtered_detection);
}

int prediction_horizon(const TrackState& track, const TrackerConfig& config) {
  return track.occluded ? std::max(config.max_prediction_frames,
                                   std::min(config.max_occlusion_frames, config.max_missing_frames))
                        : config.max_prediction_frames;
}

bool prediction_is_publishable(const TrackState& track, const TrackerConfig& config) {
  return track.missing_frames > 0 && track.missing_frames <= prediction_horizon(track, config);
}

void initialize_filter(TrackState& track, const Detection& detection, int frame_index,
                       bool covariance_motion_enabled) {
  const float diagonal = std::max(1.0f, std::hypot(width(detection), height(detection)));
  const float position_variance = std::max(0.25f, 0.01f * diagonal * diagonal);
  track.center_x_filter = KalmanAxis{center_x(detection),     0.0f, position_variance, 0.0f, 0.0f,
                                     4.0f * position_variance};
  track.center_y_filter = KalmanAxis{center_y(detection),     0.0f, position_variance, 0.0f, 0.0f,
                                     4.0f * position_variance};
  track.log_width_filter = KalmanAxis{log_size(width(detection)), 0.0f, 0.04f, 0.0f, 0.0f, 0.04f};
  track.log_height_filter = KalmanAxis{log_size(height(detection)), 0.0f, 0.04f, 0.0f, 0.0f, 0.04f};
  track.state_frame_index = frame_index;
  track.last_frame_index = frame_index;
  track.filtered_detection = detection;
  track.display_detection = detection;
  track.covariance_motion_enabled = covariance_motion_enabled;
}

void advance_track(TrackState& track, int frame_index, float camera_uncertainty) {
  if (!track.covariance_motion_enabled) {
    return;
  }
  const int elapsed_frames = std::max(0, frame_index - track.state_frame_index);
  if (elapsed_frames == 0) {
    return;
  }
  const float elapsed = static_cast<float>(elapsed_frames);
  const float diagonal =
      std::max(1.0f, std::hypot(width(track.filtered_detection), height(track.filtered_detection)));
  const float position_noise = 0.0025f * diagonal * diagonal + camera_uncertainty;
  predict_axis(track.center_x_filter, elapsed, position_noise, 0.25f * position_noise);
  predict_axis(track.center_y_filter, elapsed, position_noise, 0.25f * position_noise);
  predict_axis(track.log_width_filter, elapsed, 0.0025f, 0.001f);
  predict_axis(track.log_height_filter, elapsed, 0.0025f, 0.001f);
  stabilize_size_axis(track.log_width_filter);
  stabilize_size_axis(track.log_height_filter);

  const Detection previous = track.filtered_detection;
  const Detection predicted = filter_box(track);
  const float dx = center_x(predicted) - center_x(previous);
  const float dy = center_y(predicted) - center_y(previous);
  const bool scale_changed = std::abs(width(predicted) - width(previous)) > 1.0e-3f ||
                             std::abs(height(predicted) - height(previous)) > 1.0e-3f;
  if (scale_changed) {
    track.filtered_detection_corners = box_corners(predicted);
  } else {
    for (std::size_t index = 0; index < track.filtered_detection_corners.size(); index += 2) {
      track.filtered_detection_corners[index] += dx;
      track.filtered_detection_corners[index + 1] += dy;
    }
  }
  track.filtered_detection = predicted;
  track.state_frame_index = frame_index;
}

Detection predict(const TrackState& track, int frame_index) {
  if (!track.covariance_motion_enabled) {
    const int elapsed = std::max(0, frame_index - track.last_frame_index);
    const float predicted_x =
        center_x(track.filtered_detection) + track.legacy_velocity_x * elapsed;
    const float predicted_y =
        center_y(track.filtered_detection) + track.legacy_velocity_y * elapsed;
    const float predicted_w =
        std::max(1.0f, width(track.filtered_detection) + track.legacy_velocity_w * elapsed);
    const float predicted_h =
        std::max(1.0f, height(track.filtered_detection) + track.legacy_velocity_h * elapsed);
    return Detection{predicted_x - predicted_w * 0.5f,
                     predicted_y - predicted_h * 0.5f,
                     predicted_x + predicted_w * 0.5f,
                     predicted_y + predicted_h * 0.5f,
                     track.detection.score,
                     track.detection.class_id};
  }
  const int elapsed = std::max(0, frame_index - track.state_frame_index);
  const float predicted_x =
      track.center_x_filter.position + track.center_x_filter.velocity * elapsed;
  const float predicted_y =
      track.center_y_filter.position + track.center_y_filter.velocity * elapsed;
  const float predicted_w =
      size_from_log(track.log_width_filter.position + track.log_width_filter.velocity * elapsed);
  const float predicted_h =
      size_from_log(track.log_height_filter.position + track.log_height_filter.velocity * elapsed);
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

BoxCorners box_corners(const Detection& detection) {
  return BoxCorners{detection.x1, detection.y1, detection.x2, detection.y1,
                    detection.x1, detection.y2, detection.x2, detection.y2};
}

void transform_track_box(Detection& detection, BoxCorners& corners,
                         const CameraTransform& transform) {
  float x1 = std::numeric_limits<float>::max();
  float y1 = std::numeric_limits<float>::max();
  float x2 = std::numeric_limits<float>::lowest();
  float y2 = std::numeric_limits<float>::lowest();
  for (std::size_t index = 0; index < corners.size(); index += 2) {
    const float x = corners[index];
    const float y = corners[index + 1];
    corners[index] = transform.a * x + transform.b * y + transform.tx;
    corners[index + 1] = transform.c * x + transform.d * y + transform.ty;
    x1 = std::min(x1, corners[index]);
    y1 = std::min(y1, corners[index + 1]);
    x2 = std::max(x2, corners[index]);
    y2 = std::max(y2, corners[index + 1]);
  }
  detection.x1 = x1;
  detection.y1 = y1;
  detection.x2 = x2;
  detection.y2 = y2;
}

void transform_track_state(TrackState& track, const CameraTransform& transform) {
  if (track.previous_observation_frame >= 0) {
    const float previous_x = track.previous_observation_x;
    const float previous_y = track.previous_observation_y;
    track.previous_observation_x =
        transform.a * previous_x + transform.b * previous_y + transform.tx;
    track.previous_observation_y =
        transform.c * previous_x + transform.d * previous_y + transform.ty;
  }
  transform_track_box(track.detection, track.detection_corners, transform);
  transform_track_box(track.filtered_detection, track.filtered_detection_corners, transform);
  transform_track_box(track.display_detection, track.display_detection_corners, transform);
  if (!track.covariance_motion_enabled) {
    const float velocity_x =
        transform.a * track.legacy_velocity_x + transform.b * track.legacy_velocity_y;
    const float velocity_y =
        transform.c * track.legacy_velocity_x + transform.d * track.legacy_velocity_y;
    track.legacy_velocity_x = velocity_x;
    track.legacy_velocity_y = velocity_y;
    track.legacy_velocity_w *= std::hypot(transform.a, transform.c);
    track.legacy_velocity_h *= std::hypot(transform.b, transform.d);
    return;
  }
  const float velocity_x =
      transform.a * track.center_x_filter.velocity + transform.b * track.center_y_filter.velocity;
  const float velocity_y =
      transform.c * track.center_x_filter.velocity + transform.d * track.center_y_filter.velocity;
  const KalmanAxis previous_x_filter = track.center_x_filter;
  const KalmanAxis previous_y_filter = track.center_y_filter;
  track.center_x_filter.position = center_x(track.filtered_detection);
  track.center_y_filter.position = center_y(track.filtered_detection);
  track.center_x_filter.velocity = velocity_x;
  track.center_y_filter.velocity = velocity_y;
  track.log_width_filter.position = log_size(width(track.filtered_detection));
  track.log_height_filter.position = log_size(height(track.filtered_detection));
  // A one-time spatial similarity changes log-size position but not its time
  // derivative. For the center filters, retain the exact marginal A P A^T
  // terms obtainable from independent x/y blocks. The compact state does not
  // retain the resulting x-y cross-covariance, but every per-axis covariance
  // entry is propagated consistently instead of applying one average scale.
  const auto transform_covariance = [](KalmanAxis& output, const KalmanAxis& x, const KalmanAxis& y,
                                       float x_weight, float y_weight) {
    const float x_squared = x_weight * x_weight;
    const float y_squared = y_weight * y_weight;
    output.p00 = std::clamp(x_squared * x.p00 + y_squared * y.p00, 1.0e-4f, 1.0e6f);
    output.p01 = std::clamp(x_squared * x.p01 + y_squared * y.p01, -1.0e6f, 1.0e6f);
    output.p10 = std::clamp(x_squared * x.p10 + y_squared * y.p10, -1.0e6f, 1.0e6f);
    output.p11 = std::clamp(x_squared * x.p11 + y_squared * y.p11, 1.0e-4f, 1.0e6f);
  };
  transform_covariance(track.center_x_filter, previous_x_filter, previous_y_filter, transform.a,
                       transform.b);
  transform_covariance(track.center_y_filter, previous_x_filter, previous_y_filter, transform.c,
                       transform.d);
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

float position_measurement_variance(const Detection& detection, bool occluded) {
  const float diagonal = std::max(1.0f, std::hypot(width(detection), height(detection)));
  const float confidence = std::clamp(detection.score, 0.0f, 1.0f);
  const float standard_deviation =
      std::max(0.5f, diagonal * (0.035f + 0.10f * (1.0f - confidence)));
  const float overlap_multiplier = occluded ? 16.0f : 1.0f;
  return standard_deviation * standard_deviation * overlap_multiplier;
}

float size_measurement_variance(const Detection& detection, bool occluded) {
  const float confidence = std::clamp(detection.score, 0.0f, 1.0f);
  const float standard_deviation = 0.04f + 0.16f * (1.0f - confidence);
  return standard_deviation * standard_deviation * (occluded ? 25.0f : 1.0f);
}

float center_mahalanobis_squared(const TrackState& track, const Detection& detection,
                                 bool occluded) {
  const float measurement_variance = position_measurement_variance(detection, occluded);
  const float dx = center_x(detection) - track.center_x_filter.position;
  const float dy = center_y(detection) - track.center_y_filter.position;
  const float variance_x = std::max(1.0e-4f, track.center_x_filter.p00 + measurement_variance);
  const float variance_y = std::max(1.0e-4f, track.center_y_filter.p00 + measurement_variance);
  return dx * dx / variance_x + dy * dy / variance_y;
}

float direction_disagreement(const TrackState& track, const Detection& detection, int frame_index) {
  if (track.previous_observation_frame < 0 ||
      track.last_frame_index <= track.previous_observation_frame ||
      frame_index <= track.last_frame_index) {
    return 0.0f;
  }
  const float previous_dx = center_x(track.detection) - track.previous_observation_x;
  const float previous_dy = center_y(track.detection) - track.previous_observation_y;
  const float candidate_dx = center_x(detection) - center_x(track.detection);
  const float candidate_dy = center_y(detection) - center_y(track.detection);
  const float previous_norm = std::hypot(previous_dx, previous_dy);
  const float candidate_norm = std::hypot(candidate_dx, candidate_dy);
  if (previous_norm < 0.5f || candidate_norm < 0.5f) {
    return 0.0f;
  }
  const float cosine = std::clamp((previous_dx * candidate_dx + previous_dy * candidate_dy) /
                                      (previous_norm * candidate_norm),
                                  -1.0f, 1.0f);
  return 0.5f * (1.0f - cosine);
}

bool geometry_match(const TrackState& track, const Detection& detection,
                    const TrackerConfig& config, int frame_index) {
  if (detection.class_id != track.detection.class_id) {
    return false;
  }
  const Detection reference =
      config.center_distance_enabled ? predict(track, frame_index) : track.detection;
  const float iou = iou_xyxy(reference, detection);
  if (!config.center_distance_enabled) {
    return iou >= config.match_iou_threshold;
  }
  const float distance = normalized_center_distance(reference, detection);
  if (iou < config.match_iou_threshold && distance > config.max_center_distance) {
    return false;
  }
  if (!config.covariance_motion_enabled) {
    return true;
  }
  // A generous 2-D chi-square gate is deliberate for tiny boxes and abrupt
  // drone maneuvers. The normalized-center gate remains the hard user-facing
  // limit; covariance rejects only statistically implausible candidates.
  return center_mahalanobis_squared(track, detection, track.occluded) <= 16.0f;
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
  if (!std::isfinite(config.overlap_threshold) || config.overlap_threshold < 0.0f ||
      config.overlap_threshold > 1.0f) {
    throw std::invalid_argument("overlap_threshold must be in [0, 1]");
  }
  if (config.max_occlusion_frames < 0 || config.max_occlusion_frames > config.max_missing_frames) {
    throw std::invalid_argument("max_occlusion_frames must be in [0, max_missing_frames]");
  }
  if (config.max_active_tracks < 1) {
    throw std::invalid_argument("max_active_tracks must be >= 1");
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
  std::vector<unsigned char> ambiguous_tracks;
  std::vector<unsigned char> occlusion_visited;
  std::vector<int> occlusion_component;
  std::vector<int> occlusion_stack;
  std::vector<int> occlusion_detection_owner;
  std::vector<int> occlusion_detection_seen;
  int occlusion_visit_generation = 0;
  std::vector<Detection> predicted_tracks;
  std::vector<unsigned char> previously_occluded;
  std::vector<int> feasible_track_counts;
  std::vector<unsigned char> published_tracks;
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
  return CameraTransform{1.0f,
                         0.0f,
                         median(impl.motion_offsets_x),
                         0.0f,
                         1.0f,
                         median(impl.motion_offsets_y),
                         true,
                         0.25f,
                         typical_diagonal * 0.10f,
                         static_cast<int>(impl.motion_offsets_x.size())};
}

ObjectTracker::ObjectTracker(TrackerConfig config)
    : config_(config), impl_(std::make_unique<Impl>()) {
  validate_config(config_);
  // The detector is capped at 100 outputs in the shipped profile. Reserve
  // enough state for those observations plus retained tracks so steady-state
  // updates do not allocate on the A65 association path.
  constexpr std::size_t kTrackCapacity = 256;
  constexpr std::size_t kAssignmentCapacity = 512;
  impl_->tracks.reserve(kTrackCapacity);
  impl_->high_detections.reserve(kTrackCapacity);
  impl_->low_detections.reserve(kTrackCapacity);
  impl_->track_indices.reserve(kTrackCapacity);
  impl_->assignments.reserve(kTrackCapacity);
  impl_->matched_tracks.reserve(kTrackCapacity);
  impl_->matched_detections.reserve(kTrackCapacity);
  impl_->ambiguous_tracks.reserve(kTrackCapacity);
  impl_->occlusion_visited.reserve(kTrackCapacity);
  impl_->occlusion_component.reserve(kTrackCapacity);
  impl_->occlusion_stack.reserve(kTrackCapacity);
  impl_->occlusion_detection_owner.reserve(kTrackCapacity);
  impl_->occlusion_detection_seen.reserve(kTrackCapacity);
  impl_->predicted_tracks.reserve(kTrackCapacity);
  impl_->previously_occluded.reserve(kTrackCapacity);
  impl_->feasible_track_counts.reserve(kTrackCapacity);
  impl_->published_tracks.reserve(kTrackCapacity);
  impl_->costs.reserve(kTrackCapacity * kTrackCapacity);
  impl_->u.reserve(kAssignmentCapacity);
  impl_->v.reserve(kAssignmentCapacity);
  impl_->minv.reserve(kAssignmentCapacity);
  impl_->p.reserve(kAssignmentCapacity);
  impl_->way.reserve(kAssignmentCapacity);
  impl_->row_to_column.reserve(kTrackCapacity);
  impl_->used.reserve(kAssignmentCapacity);
  impl_->motion_tracks.reserve(kTrackCapacity);
  impl_->motion_detections.reserve(kTrackCapacity);
  impl_->motion_values_x.reserve(kTrackCapacity);
  impl_->motion_values_y.reserve(kTrackCapacity);
  impl_->motion_scales.reserve(kTrackCapacity * 2);
  impl_->motion_offsets_x.reserve(kTrackCapacity);
  impl_->motion_offsets_y.reserve(kTrackCapacity);
  impl_->motion_votes.reserve(kTrackCapacity * kTrackCapacity);
  impl_->motion_pairs.reserve(kTrackCapacity * kTrackCapacity);
  impl_->motion_histogram.reserve(kTrackCapacity * 4);
  impl_->motion_track_used.reserve(kTrackCapacity);
  impl_->motion_detection_used.reserve(kTrackCapacity);
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
  const float camera_confidence = camera_motion.valid && camera_motion.confidence > 0.0f
                                      ? std::clamp(camera_motion.confidence, 0.0f, 1.0f)
                                      : (camera_motion.valid ? 1.0f : 0.0f);
  const float camera_uncertainty =
      camera_motion.valid
          ? std::max(0.0f, camera_motion.reprojection_error * camera_motion.reprojection_error) +
                4.0f * (1.0f - camera_confidence)
          : (config_.camera_motion_compensation ? 4.0f : 0.0f);
  impl_->previously_occluded.resize(impl_->tracks.size());
  for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
    auto& track = impl_->tracks[index];
    impl_->previously_occluded[index] = track.occluded ? 1 : 0;
    advance_track(track, frame_index, camera_uncertainty);
    track.occluded = false;
    track.association_confidence = 1.0f;
  }

  // Detect components where two predicted identities compete for fewer
  // observations. Those measurements are not identity-bearing and must not be
  // allowed to corrupt velocity, size, or appearance state.
  impl_->ambiguous_tracks.assign(impl_->tracks.size(), 0);
  const bool occlusion_tracking_enabled = config_.center_distance_enabled &&
                                          config_.covariance_motion_enabled &&
                                          config_.max_occlusion_frames > 0;
  if (occlusion_tracking_enabled) {
    impl_->predicted_tracks.clear();
    for (const auto& track : impl_->tracks) {
      impl_->predicted_tracks.push_back(predict(track, frame_index));
    }
    impl_->occlusion_visited.assign(impl_->tracks.size(), 0);
    for (std::size_t root = 0; root < impl_->tracks.size(); ++root) {
      if (impl_->occlusion_visited[root] != 0) {
        continue;
      }
      impl_->occlusion_component.clear();
      impl_->occlusion_stack.clear();
      impl_->occlusion_stack.push_back(static_cast<int>(root));
      impl_->occlusion_visited[root] = 1;
      while (!impl_->occlusion_stack.empty()) {
        const int first_index = impl_->occlusion_stack.back();
        impl_->occlusion_stack.pop_back();
        impl_->occlusion_component.push_back(first_index);
        const auto first = static_cast<std::size_t>(first_index);
        for (std::size_t second = 0; second < impl_->tracks.size(); ++second) {
          if (impl_->occlusion_visited[second] != 0 ||
              impl_->tracks[first].detection.class_id != impl_->tracks[second].detection.class_id) {
            continue;
          }
          if (overlap_coefficient(impl_->predicted_tracks[first], impl_->predicted_tracks[second]) <
                  config_.overlap_threshold &&
              !swept_boxes_may_overlap(impl_->tracks[first], impl_->predicted_tracks[first],
                                       impl_->tracks[second], impl_->predicted_tracks[second])) {
            continue;
          }
          impl_->occlusion_visited[second] = 1;
          impl_->occlusion_stack.push_back(static_cast<int>(second));
        }
      }
      if (impl_->occlusion_component.size() < 2) {
        continue;
      }

      // Counting detections is insufficient: two detections may both be
      // feasible for only one of three overlapping tracks. Compute the actual
      // maximum bipartite matching cardinality for the connected component.
      impl_->occlusion_detection_owner.assign(detections.size(), -1);
      impl_->occlusion_detection_seen.assign(detections.size(), 0);
      int maximum_matches = 0;
      const auto augment = [&](auto&& self, int track_index) -> bool {
        for (std::size_t detection_index = 0; detection_index < detections.size();
             ++detection_index) {
          if (impl_->occlusion_detection_seen[detection_index] ==
                  impl_->occlusion_visit_generation ||
              !geometry_match(impl_->tracks[static_cast<std::size_t>(track_index)],
                              detections[detection_index], config_, frame_index)) {
            continue;
          }
          impl_->occlusion_detection_seen[detection_index] = impl_->occlusion_visit_generation;
          const int owner = impl_->occlusion_detection_owner[detection_index];
          if (owner < 0 || self(self, owner)) {
            impl_->occlusion_detection_owner[detection_index] = track_index;
            return true;
          }
        }
        return false;
      };
      for (const int track_index : impl_->occlusion_component) {
        if (++impl_->occlusion_visit_generation == std::numeric_limits<int>::max()) {
          impl_->occlusion_detection_seen.assign(detections.size(), 0);
          impl_->occlusion_visit_generation = 1;
        }
        maximum_matches += augment(augment, track_index) ? 1 : 0;
      }
      if (maximum_matches < static_cast<int>(impl_->occlusion_component.size())) {
        for (const int track_index : impl_->occlusion_component) {
          impl_->ambiguous_tracks[static_cast<std::size_t>(track_index)] = 1;
        }
      }
    }
  }
  for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
    impl_->tracks[index].occluded = impl_->ambiguous_tracks[index] != 0;
  }

  impl_->feasible_track_counts.assign(detections.size(), 0);
  for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
    for (const auto& track : impl_->tracks) {
      if (geometry_match(track, detections[detection_index], config_, frame_index)) {
        ++impl_->feasible_track_counts[detection_index];
      }
    }
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
          const int detection_index =
              impl_->stage.detection_index[static_cast<std::size_t>(candidate_column)];
          const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
          if (!geometry_match(track, detection, config_, frame_index)) {
            impl_->costs[static_cast<std::size_t>(row * columns + candidate_column)] = kBlockedCost;
            continue;
          }
          if (!config_.center_distance_enabled) {
            impl_->costs[static_cast<std::size_t>(row * columns + candidate_column)] =
                1.0f - ious[lane];
            continue;
          }

          const float diagonal = std::max(1.0f, std::hypot(width(reference), height(reference)));
          const float iou_weight = diagonal < 32.0f ? 0.45f : 0.60f;
          const float center_weight = diagonal < 32.0f ? 0.25f : 0.15f;
          const float mahalanobis_weight = 0.15f;
          const float direction_weight = 0.10f;
          const float size_weight = 0.05f;
          const float normalized_center =
              distances[lane] / std::max(0.01f, config_.max_center_distance);
          const float mahalanobis =
              center_mahalanobis_squared(track, detection, track.occluded) / 16.0f;
          const float direction = direction_disagreement(track, detection, frame_index);
          const float size_difference =
              0.5f * (std::abs(log_size(width(detection)) - track.log_width_filter.position) +
                      std::abs(log_size(height(detection)) - track.log_height_filter.position));
          const float staleness_penalty = 0.01f * static_cast<float>(track.missing_frames);
          impl_->costs[static_cast<std::size_t>(row * columns + candidate_column)] =
              iou_weight * (1.0f - ious[lane]) + center_weight * std::min(1.0f, normalized_center) +
              mahalanobis_weight * std::min(1.0f, mahalanobis) + direction_weight * direction +
              size_weight * std::min(1.0f, size_difference) + staleness_penalty;
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

  const auto associate = [&](const std::vector<int>& detection_indices, bool confirmed_only) {
    impl_->track_indices.clear();
    for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
      if (impl_->matched_tracks[index] == 0 &&
          (!confirmed_only || impl_->tracks[index].confirmed)) {
        impl_->track_indices.push_back(static_cast<int>(index));
      }
    }
    std::sort(impl_->track_indices.begin(), impl_->track_indices.end(), [&](int a, int b) {
      const TrackState& lhs = impl_->tracks[static_cast<std::size_t>(a)];
      const TrackState& rhs = impl_->tracks[static_cast<std::size_t>(b)];
      if (!config_.center_distance_enabled && lhs.last_frame_index != rhs.last_frame_index) {
        return lhs.last_frame_index > rhs.last_frame_index;
      }
      return lhs.track_id < rhs.track_id;
    });
    if (!config_.center_distance_enabled) {
      std::size_t begin = 0;
      while (begin < impl_->track_indices.size()) {
        const int last_frame =
            impl_->tracks[static_cast<std::size_t>(impl_->track_indices[begin])].last_frame_index;
        std::size_t end = begin + 1;
        while (
            end < impl_->track_indices.size() &&
            impl_->tracks[static_cast<std::size_t>(impl_->track_indices[end])].last_frame_index ==
                last_frame) {
          ++end;
        }
        associate_group(detection_indices, begin, end);
        begin = end;
      }
      return;
    }
    // One global pool lets the assignment optimize all currently feasible
    // identities together. Recency remains an explicit cost and uncertainty
    // input, instead of a hard cascade that can commit a locally convenient
    // match before seeing the full component.
    associate_group(detection_indices, 0, impl_->track_indices.size());
  };

  // ByteTrack-style staging lets low-score detections recover established
  // tracks, while only high-score detections may create identities.
  associate(impl_->high_detections, false);
  associate(impl_->low_detections, true);

  for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
    const int track_index = impl_->assignments[detection_index];
    if (track_index < 0) {
      continue;
    }
    TrackState& track = impl_->tracks[static_cast<std::size_t>(track_index)];
    const Detection& detection = detections[detection_index];
    const bool ambiguous_measurement =
        track.occluded && impl_->feasible_track_counts[detection_index] > 1;
    if (ambiguous_measurement) {
      track.missing_frames = frame_index - track.last_frame_index;
      track.association_confidence = 0.0f;
      freeze_unobserved_size(track);
      continue;
    }

    const int elapsed = std::max(1, frame_index - track.last_frame_index);
    const bool was_missing = track.missing_frames > 0;
    if (!config_.covariance_motion_enabled) {
      const float measured_vx = (center_x(detection) - center_x(track.detection)) / elapsed;
      const float measured_vy = (center_y(detection) - center_y(track.detection)) / elapsed;
      const float measured_vw = (width(detection) - width(track.detection)) / elapsed;
      const float measured_vh = (height(detection) - height(track.detection)) / elapsed;
      const Detection prediction = predict(track, frame_index);
      const float previous = config_.velocity_momentum;
      const float measured = 1.0f - previous;
      track.legacy_velocity_x = previous * track.legacy_velocity_x + measured * measured_vx;
      track.legacy_velocity_y = previous * track.legacy_velocity_y + measured * measured_vy;
      track.legacy_velocity_w = previous * track.legacy_velocity_w + measured * measured_vw;
      track.legacy_velocity_h = previous * track.legacy_velocity_h + measured * measured_vh;
      float smoothing_alpha = config_.box_smoothing_alpha;
      if (camera_motion.valid && smoothing_alpha < 1.0f) {
        const float innovation = normalized_center_distance(prediction, detection);
        const float response =
            std::min(1.0f, innovation / std::max(0.01f, config_.max_center_distance));
        smoothing_alpha += (1.0f - smoothing_alpha) * response;
      }
      track.filtered_detection = smooth_detection(prediction, detection, smoothing_alpha);
      track.display_detection = track.filtered_detection;
      track.detection = detection;
      track.filtered_detection_corners = box_corners(track.filtered_detection);
      track.display_detection_corners = track.filtered_detection_corners;
      track.detection_corners = box_corners(track.detection);
      track.last_frame_index = frame_index;
      track.missing_frames = 0;
      track.occluded = false;
      track.association_confidence = 1.0f;
      ++track.hits;
      track.confirmed = track.hits >= config_.min_confirmed_hits;
      continue;
    }
    const float measured_vx = (center_x(detection) - center_x(track.detection)) / elapsed;
    const float measured_vy = (center_y(detection) - center_y(track.detection)) / elapsed;
    const float measured_vw =
        (log_size(width(detection)) - log_size(width(track.detection))) / elapsed;
    const float measured_vh =
        (log_size(height(detection)) - log_size(height(track.detection))) / elapsed;
    const Detection filtered_prediction = track.filtered_detection;
    const float previous = config_.covariance_motion_enabled
                               ? std::min(0.50f, config_.velocity_momentum)
                               : config_.velocity_momentum;
    const float measured = 1.0f - previous;
    const float mahalanobis = center_mahalanobis_squared(track, detection, false);
    const float maximum_size_innovation =
        kMaximumLogSizeInnovationPerFrame * static_cast<float>(elapsed);
    const float width_measurement = std::clamp(
        log_size(width(detection)), track.log_width_filter.position - maximum_size_innovation,
        track.log_width_filter.position + maximum_size_innovation);
    const float height_measurement = std::clamp(
        log_size(height(detection)), track.log_height_filter.position - maximum_size_innovation,
        track.log_height_filter.position + maximum_size_innovation);
    update_axis(track.center_x_filter, center_x(detection),
                position_measurement_variance(detection, false));
    update_axis(track.center_y_filter, center_y(detection),
                position_measurement_variance(detection, false));
    update_axis(track.log_width_filter, width_measurement,
                size_measurement_variance(detection, false));
    update_axis(track.log_height_filter, height_measurement,
                size_measurement_variance(detection, false));

    // Observation-centric momentum corrects the recursive filter after a
    // nonlinear maneuver or an occlusion gap. The legacy momentum remains a
    // tuning input, but the covariance path caps it so the tracker cannot take
    // dozens of frames to learn a new direction.
    track.center_x_filter.velocity =
        previous * track.center_x_filter.velocity + measured * measured_vx;
    track.center_y_filter.velocity =
        previous * track.center_y_filter.velocity + measured * measured_vy;
    track.log_width_filter.velocity =
        previous * track.log_width_filter.velocity + measured * measured_vw;
    track.log_height_filter.velocity =
        previous * track.log_height_filter.velocity + measured * measured_vh;
    stabilize_size_axis(track.log_width_filter);
    stabilize_size_axis(track.log_height_filter);

    track.filtered_detection = filter_box(track);
    track.filtered_detection_corners = box_corners(track.filtered_detection);
    track.display_detection = was_missing
                                  ? track.filtered_detection
                                  : smooth_detection(filtered_prediction, track.filtered_detection,
                                                     config_.box_smoothing_alpha);
    track.display_detection_corners = box_corners(track.display_detection);
    track.previous_observation_x = center_x(track.detection);
    track.previous_observation_y = center_y(track.detection);
    track.previous_observation_frame = track.last_frame_index;
    track.detection = detection;
    track.detection_corners = box_corners(track.detection);
    track.last_frame_index = frame_index;
    track.missing_frames = 0;
    track.occluded = false;
    track.association_confidence = std::exp(-0.125f * std::min(16.0f, mahalanobis));
    ++track.hits;
    track.confirmed = track.hits >= config_.min_confirmed_hits;
  }

  for (const int detection_index : impl_->high_detections) {
    if (impl_->matched_detections[static_cast<std::size_t>(detection_index)] != 0 ||
        detections[static_cast<std::size_t>(detection_index)].score < config_.new_track_threshold) {
      continue;
    }
    const Detection& detection = detections[static_cast<std::size_t>(detection_index)];
    const bool shadows_active_track =
        config_.center_distance_enabled &&
        std::any_of(impl_->tracks.begin(), impl_->tracks.end(), [&](const TrackState& track) {
          if (!track.confirmed || track.detection.class_id != detection.class_id) {
            return false;
          }
          const Detection prediction = predict(track, frame_index);
          return overlap_coefficient(prediction, detection) >=
                 std::max(0.50f, config_.overlap_threshold);
        });
    if (shadows_active_track) {
      impl_->matched_detections[static_cast<std::size_t>(detection_index)] = 1;
      continue;
    }
    int track_index = static_cast<int>(impl_->tracks.size());
    if (track_index >= config_.max_active_tracks) {
      int replacement = -1;
      for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
        if (impl_->matched_tracks[index] != 0) {
          continue;
        }
        if (replacement < 0) {
          replacement = static_cast<int>(index);
          continue;
        }
        const auto& candidate = impl_->tracks[index];
        const auto& current = impl_->tracks[static_cast<std::size_t>(replacement)];
        const auto candidate_priority = std::tuple{candidate.occluded, candidate.confirmed,
                                                   candidate.last_frame_index, candidate.track_id};
        const auto current_priority = std::tuple{current.occluded, current.confirmed,
                                                 current.last_frame_index, current.track_id};
        if (candidate_priority < current_priority) {
          replacement = static_cast<int>(index);
        }
      }
      if (replacement < 0) {
        continue;
      }
      track_index = replacement;
    }
    TrackState track;
    track.track_id = next_track_id_++;
    track.detection = detection;
    initialize_filter(track, detection, frame_index, config_.covariance_motion_enabled);
    track.detection_corners = box_corners(detection);
    track.filtered_detection_corners = track.detection_corners;
    track.display_detection_corners = track.detection_corners;
    track.confirmed = config_.min_confirmed_hits <= 1;
    if (static_cast<std::size_t>(track_index) == impl_->tracks.size()) {
      impl_->tracks.push_back(track);
      impl_->matched_tracks.push_back(1);
    } else {
      impl_->tracks[static_cast<std::size_t>(track_index)] = track;
      impl_->matched_tracks[static_cast<std::size_t>(track_index)] = 1;
    }
    impl_->matched_detections[static_cast<std::size_t>(detection_index)] = 1;
    impl_->assignments[static_cast<std::size_t>(detection_index)] = track_index;
  }

  for (std::size_t index = 0; index < impl_->tracks.size(); ++index) {
    if (impl_->matched_tracks[index] == 0) {
      freeze_unobserved_size(impl_->tracks[index]);
      impl_->tracks[index].missing_frames = frame_index - impl_->tracks[index].last_frame_index;
      if (index < impl_->previously_occluded.size() && impl_->previously_occluded[index] != 0) {
        impl_->tracks[index].occluded = true;
      }
    }
  }

  tracked.clear();
  tracked.reserve(detections.size() + impl_->tracks.size());
  impl_->published_tracks.assign(impl_->tracks.size(), 0);
  for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
    const int track_index = impl_->assignments[detection_index];
    if (track_index < 0) {
      continue;
    }
    const TrackState& track = impl_->tracks[static_cast<std::size_t>(track_index)];
    if (!track.confirmed) {
      continue;
    }
    const bool predicted = track.last_frame_index != frame_index;
    if (predicted && !prediction_is_publishable(track, config_)) {
      continue;
    }
    const Detection& filtered = predicted ? track.filtered_detection : track.display_detection;
    const Detection& detection = detections[detection_index];
    tracked.push_back(TrackedDetection{track.track_id, filtered.x1, filtered.y1, filtered.x2,
                                       filtered.y2, detection.score, detection.class_id, predicted,
                                       track.occluded, track.association_confidence});
    impl_->published_tracks[static_cast<std::size_t>(track_index)] = 1;
  }

  if (config_.max_prediction_frames > 0 || config_.max_occlusion_frames > 0) {
    for (std::size_t track_index = 0; track_index < impl_->tracks.size(); ++track_index) {
      const auto& track = impl_->tracks[track_index];
      if (!track.confirmed || track.missing_frames <= 0 ||
          !prediction_is_publishable(track, config_) ||
          track.detection.score < config_.high_score_threshold) {
        continue;
      }
      if (impl_->published_tracks[track_index] != 0) {
        continue;
      }
      Detection prediction = predict(track, frame_index);
      const bool duplicate = std::any_of(tracked.begin(), tracked.end(), [&](const auto& output) {
        const Detection observed{output.x1, output.y1,    output.x2,
                                 output.y2, output.score, output.class_id};
        return observed.class_id == prediction.class_id && iou_xyxy(observed, prediction) > 0.5f;
      });
      if (duplicate && !track.occluded) {
        continue;
      }
      prediction.score *= std::pow(0.9f, static_cast<float>(track.missing_frames));
      tracked.push_back(TrackedDetection{track.track_id, prediction.x1, prediction.y1,
                                         prediction.x2, prediction.y2, prediction.score,
                                         prediction.class_id, true, track.occluded,
                                         track.association_confidence});
      impl_->published_tracks[track_index] = 1;
    }
  }

  std::erase_if(impl_->tracks, [&](const TrackState& track) {
    return (!track.confirmed && track.missing_frames > 0) ||
           track.missing_frames > config_.max_missing_frames;
  });
}

} // namespace multi_stream_people_tracker
