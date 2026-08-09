#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/camera_motion_api.cpp"
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"
#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_overlay_api.cpp"
#include "support/testing/test_process.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

using multi_stream_people_tracker::Detection;
using multi_stream_people_tracker::draw_tracks_bgr;
using multi_stream_people_tracker::FrameCameraMotionEstimator;
using multi_stream_people_tracker::ObjectTracker;
using multi_stream_people_tracker::track_color;
using multi_stream_people_tracker::TrackedDetection;
using multi_stream_people_tracker::TrackerConfig;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir("multi-stream-people-tracker", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_config_only_accepts_four_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_four_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "input:\n"
                                            "  codec: hevc\n"
                                            "inference:\n"
                                            "  max_inflight_per_stream: 3\n"
                                            "  max_inflight_total: 12\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "four-stream config validates") &&
      expect_contains(result.stdout_text, "streams=4", "validate output reports stream count") &&
      expect_contains(result.stdout_text, "max_inflight_per_stream=3",
                      "validate output reports per-stream inflight limit") &&
      expect_contains(result.stdout_text, "max_inflight_total=12",
                      "validate output reports total inflight limit") &&
      expect_contains(result.stdout_text, "overflow_policy=keep_latest",
                      "validate output reports default overflow policy") &&
      expect_contains(result.stdout_text, "min_score=0.55",
                      "legacy omission preserves the decoder floor") &&
      expect_contains(result.stdout_text, "match_iou_threshold=0.3",
                      "legacy omission preserves the IoU threshold") &&
      expect_contains(result.stdout_text, "center_distance_enabled=false",
                      "legacy omission preserves IoU-only matching");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_block_overflow_policy(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_accepts_block_overflow_policy",
                   "model:\n"
                   "  path: models/yolo26n-p2-tiny-drone-int8-qat-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "runtime:\n"
                   "  overflow_policy: block\n"
                   "inference:\n"
                   "  num_classes: 1\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "block overflow policy validates") &&
                  expect_contains(result.stdout_text, "overflow_policy=block",
                                  "validate output reports block overflow policy");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_overflow_policy(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_overflow_policy",
                   "model:\n"
                   "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "runtime:\n"
                   "  overflow_policy: drop_oldest\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "invalid overflow policy is rejected") &&
      expect_contains(result.stderr_text, "runtime.overflow_policy must be keep_latest or block",
                      "invalid overflow policy error names accepted values");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_block_with_shared_fan_in(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_block_with_shared_fan_in",
                   "model:\n"
                   "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "  - rtsp://127.0.0.1:8554/src2\n"
                   "runtime:\n"
                   "  overflow_policy: block\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "block policy rejects shared detector fan-in") &&
      expect_contains(result.stderr_text,
                      "runtime.overflow_policy=block requires exactly one stream",
                      "shared fan-in error explains the single-stream requirement");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_too_many_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "  - rtsp://127.0.0.1:8554/src5\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to four streams",
                                  "too-many-stream error mentions four-stream phase limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_inflight_limit(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_inflight_limit",
                   "model:\n"
                   "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "inference:\n"
                   "  max_inflight_total: 0\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "invalid inflight limit is rejected") &&
                  expect_contains(result.stderr_text, "max_inflight_total must be -1 or > 0",
                                  "invalid inflight error names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_non_positive_class_count(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_non_positive_class_count",
                   "model:\n"
                   "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "inference:\n"
                   "  num_classes: 0\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "non-positive class count is rejected") &&
                  expect_contains(result.stderr_text, "inference.num_classes must be > 0",
                                  "class-count error names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_target_outside_class_count(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_target_outside_class_count",
                   "model:\n"
                   "  path: models/yolo26n-p2-tiny-drone-int8-qat-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "inference:\n"
                   "  num_classes: 1\n"
                   "  target_class_id: 1\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "out-of-range target class is rejected") &&
      expect_contains(result.stderr_text,
                      "inference.target_class_id (1) must be less than inference.num_classes (1)",
                      "target-class error reports both configured values");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_nan_threshold(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_nan_threshold",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "inference:\n"
                                            "  min_score: nan\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "NaN detection threshold is rejected") &&
                  expect_contains(result.stderr_text, "min_score must be between 0 and 1",
                                  "NaN threshold error names min_score");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_omitted_tracking_thresholds_follow_decoder_floor(const std::string& binary) {
  const fs::path config_path = write_config("test_omitted_tracking_thresholds_follow_decoder_floor",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "inference:\n"
                                            "  min_score: 0.70\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0,
                              "omitted tracker thresholds follow a raised decoder floor");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_omitted_new_track_threshold_follows_high_threshold(const std::string& binary) {
  const fs::path config_path =
      write_config("test_omitted_new_track_threshold_follows_high_threshold",
                   "model:\n"
                   "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "tracking:\n"
                   "  high_score_threshold: 0.75\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0,
                              "omitted new-track threshold follows the resolved high threshold");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_legacy_iou_config_keeps_iou_only_matching(const std::string& binary) {
  const fs::path config_path = write_config("test_legacy_iou_config_keeps_iou_only_matching",
                                            "model:\n"
                                            "  path: models/yolo26m-det-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "tracking:\n"
                                            "  iou_threshold: 0.50\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "legacy IoU config validates") &&
                  expect_contains(result.stdout_text, "center_distance_enabled=false",
                                  "legacy IoU config preserves IoU-only matching");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_tracker_reuses_track_id_for_nearby_detection() {
  ObjectTracker tracker;
  const auto first = tracker.update({Detection{10.0f, 10.0f, 50.0f, 80.0f, 0.9f, 0}}, 0);
  const auto second = tracker.update({Detection{12.0f, 11.0f, 52.0f, 81.0f, 0.8f, 0}}, 1);
  return expect_true(first.size() == 1, "tracker returns one detection on first frame") &&
         expect_true(second.size() == 1, "tracker returns one detection on second frame") &&
         expect_true(first.front().track_id == second.front().track_id,
                     "tracker reuses track id for nearby detection");
}

bool test_motion_compensated_box_smoothing_reduces_jitter() {
  TrackerConfig config;
  config.box_smoothing_alpha = 0.5f;
  config.velocity_momentum = 0.9f;
  ObjectTracker tracker(config);
  tracker.update({Detection{0.0f, 0.0f, 4.0f, 4.0f, 0.9f, 0}}, 0);
  const auto smoothed = tracker.update({Detection{2.0f, 0.0f, 6.0f, 4.0f, 0.9f, 0}}, 1);
  return expect_true(smoothed.size() == 1, "smoothed track remains visible") &&
         expect_true(std::abs(smoothed.front().x1 - 1.0f) < 1e-4f &&
                         std::abs(smoothed.front().x2 - 5.0f) < 1e-4f,
                     "motion-compensated smoothing attenuates detector box jitter");
}

bool test_camera_motion_compensation_preserves_ids_across_fast_pan() {
  TrackerConfig config;
  config.max_center_distance = 0.5f;
  config.box_smoothing_alpha = 0.5f;
  config.camera_motion_compensation = true;
  ObjectTracker tracker(config);
  std::vector<Detection> first_detections;
  std::vector<Detection> panned_detections;
  for (int index = 0; index < 6; ++index) {
    const float x = 40.0f * index;
    first_detections.push_back(Detection{x, 20.0f, x + 4.0f, 24.0f, 0.9f, 0});
    panned_detections.push_back(Detection{x + 20.0f, 20.0f, x + 24.0f, 24.0f, 0.9f, 0});
  }
  const auto first = tracker.update(first_detections, 0);
  const auto panned = tracker.update(panned_detections, 1);
  if (!expect_true(first.size() == 6 && panned.size() == 6,
                   "camera pan keeps every tracked detection")) {
    return false;
  }
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (!expect_true(first[index].track_id == panned[index].track_id,
                     "camera pan preserves each identity") ||
        !expect_true(std::abs(panned[index].x1 - panned_detections[index].x1) < 1e-4f,
                     "camera-compensated smoothing follows the pan without lag")) {
      return false;
    }
  }
  return true;
}

bool test_camera_motion_compensation_does_not_bridge_scene_cut() {
  TrackerConfig config;
  config.max_center_distance = 0.5f;
  config.camera_motion_compensation = true;
  ObjectTracker tracker(config);
  std::vector<Detection> first_detections;
  std::vector<Detection> cut_detections;
  for (int index = 0; index < 8; ++index) {
    const float x = 20.0f * index;
    first_detections.push_back(Detection{x, 20.0f, x + 4.0f, 24.0f, 0.9f, 0});
    cut_detections.push_back(Detection{x + 100.0f, 100.0f, x + 104.0f, 104.0f, 0.9f, 0});
  }
  const auto first = tracker.update(first_detections, 0);
  const auto after_cut = tracker.update(cut_detections, 1);
  return expect_true(first.size() == 8 && after_cut.size() == 8,
                     "scene cut publishes the new observations") &&
         expect_true(after_cut.front().track_id != first.front().track_id,
                     "scene cut does not carry an identity across unrelated images") &&
         expect_true(tracker.active_track_count() == 16,
                     "scene cut is not mistaken for coherent camera motion");
}

bool test_external_camera_transform_is_not_learned_as_object_velocity() {
  TrackerConfig config;
  config.max_center_distance = 0.5f;
  config.velocity_momentum = 0.9f;
  config.camera_motion_compensation = true;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{0.0f, 20.0f, 4.0f, 24.0f, 0.9f, 0}}, 0);
  if (!expect_true(first.size() == 1, "camera transform test creates one track")) {
    return false;
  }
  const int track_id = first.front().track_id;
  const multi_stream_people_tracker::CameraTransform pan{1.0f, 0.0f, 20.0f, 0.0f, 1.0f, 0.0f, true};
  for (int frame = 1; frame <= 12; ++frame) {
    const float x = 20.0f * frame;
    const auto tracked =
        tracker.update({Detection{x, 20.0f, x + 4.0f, 24.0f, 0.9f, 0}}, frame, pan);
    if (!expect_true(tracked.size() == 1 && tracked.front().track_id == track_id,
                     "repeated global pan preserves the original identity") ||
        !expect_true(std::abs(tracked.front().x1 - x) < 1e-4f,
                     "global pan is applied exactly once")) {
      return false;
    }
  }
  return true;
}

bool test_repeated_camera_rotation_does_not_inflate_prediction() {
  TrackerConfig config;
  config.max_prediction_frames = 12;
  config.camera_motion_compensation = true;
  ObjectTracker tracker(config);
  tracker.update({Detection{-5.0f, -5.0f, 5.0f, 5.0f, 0.9f, 0}}, 0);

  constexpr float angle = 0.1f;
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  const multi_stream_people_tracker::CameraTransform rotation{cosine, -sine, 0.0f, sine,
                                                              cosine, 0.0f,  true};
  for (int frame = 1; frame <= 12; ++frame) {
    const auto tracked = tracker.update({}, frame, rotation);
    if (!expect_true(tracked.size() == 1 && tracked.front().predicted,
                     "camera rotation keeps the gap prediction visible")) {
      return false;
    }
    const float cumulative_angle = angle * static_cast<float>(frame);
    const float expected_extent =
        10.0f * (std::abs(std::cos(cumulative_angle)) + std::abs(std::sin(cumulative_angle)));
    const float actual_extent = tracked.front().x2 - tracked.front().x1;
    if (!expect_true(std::abs(actual_extent - expected_extent) < 1e-3f,
                     "camera rotation preserves oriented track geometry")) {
      return false;
    }
  }
  return true;
}

bool test_orb_camera_motion_estimator_recovers_translation() {
  cv::Mat first(256, 320, CV_8UC1);
  cv::RNG rng(12345);
  rng.fill(first, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(first, first, cv::Size(3, 3), 0.8);
  cv::Mat second;
  const cv::Mat affine = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 18.0, 0.0, 1.0, 7.0);
  cv::warpAffine(first, second, affine, first.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

  FrameCameraMotionEstimator estimator;
  const auto initial = estimator.update(first);
  const auto motion = estimator.update(second);
  return expect_true(!initial.valid, "first frame does not invent camera motion") &&
         expect_true(motion.valid, "ORB/RANSAC accepts a coherent frame translation") &&
         expect_true(std::abs(motion.tx - 18.0f) < 2.0f && std::abs(motion.ty - 7.0f) < 2.0f,
                     "ORB/RANSAC recovers the frame translation");
}

bool test_recent_track_wins_before_stale_track() {
  TrackerConfig config;
  config.max_center_distance = 2.0f;
  config.velocity_momentum = 0.9f;
  ObjectTracker tracker(config);
  const auto first = tracker.update(
      {Detection{0.0f, 0.0f, 4.0f, 4.0f, 0.9f, 0}, Detection{10.0f, 0.0f, 14.0f, 4.0f, 0.9f, 0}},
      0);
  tracker.update({Detection{6.0f, 0.0f, 10.0f, 4.0f, 0.9f, 0}}, 1);
  const auto recent = tracker.update({Detection{2.0f, 0.0f, 6.0f, 4.0f, 0.9f, 0}}, 2);
  return expect_true(first.size() == 2 && recent.size() == 1,
                     "recency scenario produces one active detection") &&
         expect_true(recent.front().track_id == first[1].track_id,
                     "active track keeps the detection ahead of a stale track");
}

bool test_tracker_drops_track_after_missing_budget() {
  TrackerConfig config;
  config.max_missing_frames = 1;
  ObjectTracker tracker(config);
  tracker.update({Detection{10.0f, 10.0f, 50.0f, 80.0f, 0.9f, 0}}, 0);
  tracker.update({}, 1);
  tracker.update({}, 2);
  return expect_true(tracker.active_track_count() == 0,
                     "tracker expires track after missing frame budget");
}

bool test_zero_missing_budget_keeps_continuous_track() {
  TrackerConfig config;
  config.max_missing_frames = 0;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{10.0f, 10.0f, 14.0f, 14.0f, 0.9f, 0}}, 0);
  const auto continuous = tracker.update({Detection{11.0f, 10.0f, 15.0f, 14.0f, 0.9f, 0}}, 1);
  return expect_true(first.size() == 1 && continuous.size() == 1,
                     "continuous detections are tracked with zero missing-frame budget") &&
         expect_true(first.front().track_id == continuous.front().track_id,
                     "zero missing-frame budget preserves a continuous track id");
}

bool test_tracker_recovers_after_exact_missing_budget() {
  TrackerConfig config;
  config.max_missing_frames = 1;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{10.0f, 10.0f, 14.0f, 14.0f, 0.9f, 0}}, 0);
  tracker.update({}, 1);
  const auto recovered = tracker.update({Detection{11.0f, 10.0f, 15.0f, 14.0f, 0.9f, 0}}, 2);
  return expect_true(first.size() == 1 && recovered.size() == 1,
                     "detection returns after the full missing-frame budget") &&
         expect_true(first.front().track_id == recovered.front().track_id,
                     "track id survives the exact missing-frame budget");
}

bool test_tracker_matches_tiny_non_overlapping_boxes_by_motion() {
  TrackerConfig config;
  config.match_iou_threshold = 0.5f;
  config.max_center_distance = 2.0f;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{10.0f, 10.0f, 12.0f, 12.0f, 0.9f, 0}}, 0);
  const auto second = tracker.update({Detection{13.0f, 10.0f, 15.0f, 12.0f, 0.8f, 0}}, 1);
  return expect_true(first.size() == 1 && second.size() == 1,
                     "tiny non-overlapping detections are tracked") &&
         expect_true(first.front().track_id == second.front().track_id,
                     "motion gate preserves the tiny-object track id");
}

bool test_tracker_can_disable_center_distance_matching() {
  TrackerConfig config;
  config.match_iou_threshold = 0.5f;
  config.max_center_distance = 2.5f;
  config.center_distance_enabled = false;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0}}, 0);
  const auto below_iou = tracker.update({Detection{5.0f, 0.0f, 15.0f, 10.0f, 0.9f, 0}}, 1);
  return expect_true(first.size() == 1 && below_iou.size() == 1,
                     "IoU-only tracker returns both detections") &&
         expect_true(first.front().track_id != below_iou.front().track_id,
                     "IoU-only tracker rejects a below-threshold center match");
}

bool test_iou_only_tracker_does_not_apply_motion_prediction() {
  TrackerConfig config;
  config.match_iou_threshold = 0.3f;
  config.velocity_momentum = 0.0f;
  config.center_distance_enabled = false;
  ObjectTracker tracker(config);
  int track_id = 0;
  const std::vector<float> positions{0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 15.0f};
  for (std::size_t frame_index = 0; frame_index < positions.size(); ++frame_index) {
    const float x1 = positions[frame_index];
    const auto tracked = tracker.update({Detection{x1, 0.0f, x1 + 10.0f, 10.0f, 0.9f, 0}},
                                        static_cast<int>(frame_index));
    if (!expect_true(tracked.size() == 1, "IoU-only reversal produces one track")) {
      return false;
    }
    if (frame_index == 0) {
      track_id = tracked.front().track_id;
    } else if (!expect_true(tracked.front().track_id == track_id,
                            "IoU-only reversal retains the legacy track id")) {
      return false;
    }
  }
  return true;
}

bool test_low_score_detection_recovers_but_does_not_create_track() {
  TrackerConfig config;
  config.high_score_threshold = 0.5f;
  config.new_track_threshold = 0.7f;
  ObjectTracker established(config);
  const auto first = established.update({Detection{10.0f, 10.0f, 14.0f, 14.0f, 0.9f, 0}}, 0);
  const auto recovered = established.update({Detection{11.0f, 10.0f, 15.0f, 14.0f, 0.2f, 0}}, 1);

  ObjectTracker fresh(config);
  const auto low_only = fresh.update({Detection{10.0f, 10.0f, 14.0f, 14.0f, 0.2f, 0}}, 0);
  return expect_true(first.size() == 1 && recovered.size() == 1,
                     "low-score detection recovers an established track") &&
         expect_true(first.front().track_id == recovered.front().track_id,
                     "recovered detection retains the track id") &&
         expect_true(low_only.empty() && fresh.active_track_count() == 0,
                     "low-score detection cannot create a track");
}

bool test_tracker_confirmation_suppresses_single_frame_noise() {
  TrackerConfig config;
  config.min_confirmed_hits = 2;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{10.0f, 10.0f, 14.0f, 14.0f, 0.9f, 0}}, 0);
  const auto second = tracker.update({Detection{10.5f, 10.0f, 14.5f, 14.0f, 0.8f, 0}}, 1);
  return expect_true(first.empty(), "unconfirmed one-frame track is not published") &&
         expect_true(second.size() == 1, "track is published after the configured hit count");
}

bool test_global_assignment_avoids_greedy_identity_loss() {
  TrackerConfig config;
  config.match_iou_threshold = 0.10f;
  config.center_distance_enabled = false;
  ObjectTracker tracker(config);
  const auto first = tracker.update(
      {Detection{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0}, Detection{8.0f, 0.0f, 18.0f, 10.0f, 0.9f, 0}},
      0);
  const auto second = tracker.update(
      {Detection{1.0f, 0.0f, 11.0f, 10.0f, 0.9f, 0}, Detection{-3.0f, 0.0f, 7.0f, 10.0f, 0.9f, 0}},
      1);
  return expect_true(first.size() == 2 && second.size() == 2,
                     "global assignment retains both feasible tracks") &&
         expect_true(second[0].track_id == first[1].track_id &&
                         second[1].track_id == first[0].track_id,
                     "global assignment chooses the minimum-cost complete matching");
}

bool test_prediction_bridges_one_high_confidence_gap() {
  TrackerConfig config;
  config.high_score_threshold = 0.5f;
  config.new_track_threshold = 0.5f;
  config.velocity_momentum = 0.0f;
  config.max_missing_frames = 3;
  config.max_prediction_frames = 1;
  ObjectTracker tracker(config);
  tracker.update({Detection{0.0f, 0.0f, 4.0f, 4.0f, 0.9f, 0}}, 0);
  const auto observed = tracker.update({Detection{1.0f, 0.0f, 5.0f, 4.0f, 0.9f, 0}}, 1);
  const auto bridged = tracker.update({}, 2);
  const auto beyond_horizon = tracker.update({}, 3);
  return expect_true(observed.size() == 1 && bridged.size() == 1,
                     "one-frame detector gap is bridged") &&
         expect_true(bridged.front().track_id == observed.front().track_id &&
                         bridged.front().predicted,
                     "bridged box keeps the identity and is marked predicted") &&
         expect_true(std::abs(bridged.front().x1 - 2.0f) < 1e-4f,
                     "bridged box follows the velocity estimate") &&
         expect_true(beyond_horizon.empty(), "prediction stops at the configured horizon");
}

bool test_unconfirmed_track_expires_on_first_miss() {
  TrackerConfig config;
  config.min_confirmed_hits = 2;
  config.max_missing_frames = 30;
  ObjectTracker tracker(config);
  tracker.update({Detection{0.0f, 0.0f, 4.0f, 4.0f, 0.9f, 0}}, 0);
  tracker.update({}, 1);
  return expect_true(tracker.active_track_count() == 0,
                     "unconfirmed one-frame noise expires on its first miss");
}

bool test_track_overlay_uses_stable_per_identity_colors() {
  const auto first = track_color(1);
  const auto repeated = track_color(1);
  const auto second = track_color(2);
  cv::Mat frame = cv::Mat::zeros(32, 32, CV_8UC3);
  draw_tracks_bgr(frame, {TrackedDetection{1, 2.0f, 10.0f, 12.0f, 20.0f, 0.9f, 0, false},
                          TrackedDetection{2, 18.0f, 10.0f, 29.0f, 20.0f, 0.9f, 0, false}});
  return expect_true(first == repeated, "track color is stable for an identity") &&
         expect_true(first != second, "different identities receive different colors") &&
         expect_true(cv::countNonZero(frame.reshape(1)) > 0, "overlay modifies the output frame");
}

bool test_tracker_does_not_revive_after_missing_budget() {
  TrackerConfig config;
  config.max_missing_frames = 1;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{10.0f, 10.0f, 14.0f, 14.0f, 0.9f, 0}}, 0);
  const auto replacement = tracker.update({Detection{11.0f, 10.0f, 15.0f, 14.0f, 0.9f, 0}}, 3);
  return expect_true(first.size() == 1 && replacement.size() == 1,
                     "detection after missing budget creates a replacement track") &&
         expect_true(first.front().track_id != replacement.front().track_id,
                     "expired track id is not revived");
}

bool test_tracker_enforces_monotonic_frames_without_active_tracks() {
  ObjectTracker tracker;
  tracker.update({}, 5);
  try {
    tracker.update({}, 4);
  } catch (const std::invalid_argument&) {
    return expect_true(true, "tracker rejects a non-monotonic empty frame");
  }
  return expect_true(false, "tracker rejects a non-monotonic empty frame");
}

bool rejects_tracker_config(const TrackerConfig& config) {
  try {
    ObjectTracker tracker(config);
    (void)tracker;
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

bool test_tracker_rejects_non_finite_thresholds() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  TrackerConfig config;
  config.high_score_threshold = nan;
  bool ok = expect_true(rejects_tracker_config(config), "tracker rejects a non-finite high score");
  config = TrackerConfig{};
  config.new_track_threshold = nan;
  ok &= expect_true(rejects_tracker_config(config), "tracker rejects a non-finite new-track score");
  config = TrackerConfig{};
  config.match_iou_threshold = nan;
  ok &= expect_true(rejects_tracker_config(config), "tracker rejects a non-finite IoU threshold");
  config = TrackerConfig{};
  config.max_center_distance = nan;
  ok &= expect_true(rejects_tracker_config(config), "tracker rejects a non-finite motion gate");
  config = TrackerConfig{};
  config.velocity_momentum = nan;
  ok &= expect_true(rejects_tracker_config(config), "tracker rejects non-finite momentum");
  config = TrackerConfig{};
  config.box_smoothing_alpha = 0.0f;
  ok &= expect_true(rejects_tracker_config(config), "tracker rejects zero smoothing alpha");
  config = TrackerConfig{};
  config.max_prediction_frames = config.max_missing_frames + 1;
  ok &= expect_true(rejects_tracker_config(config),
                    "tracker rejects an excessive prediction horizon");
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_config_only_accepts_four_streams(binary);
  ok &= test_validate_config_only_accepts_block_overflow_policy(binary);
  ok &= test_validate_config_only_rejects_invalid_overflow_policy(binary);
  ok &= test_validate_config_only_rejects_block_with_shared_fan_in(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_invalid_inflight_limit(binary);
  ok &= test_validate_config_only_rejects_non_positive_class_count(binary);
  ok &= test_validate_config_only_rejects_target_outside_class_count(binary);
  ok &= test_validate_config_only_rejects_nan_threshold(binary);
  ok &= test_omitted_tracking_thresholds_follow_decoder_floor(binary);
  ok &= test_omitted_new_track_threshold_follows_high_threshold(binary);
  ok &= test_legacy_iou_config_keeps_iou_only_matching(binary);
  ok &= test_tracker_reuses_track_id_for_nearby_detection();
  ok &= test_motion_compensated_box_smoothing_reduces_jitter();
  ok &= test_camera_motion_compensation_preserves_ids_across_fast_pan();
  ok &= test_camera_motion_compensation_does_not_bridge_scene_cut();
  ok &= test_external_camera_transform_is_not_learned_as_object_velocity();
  ok &= test_repeated_camera_rotation_does_not_inflate_prediction();
  ok &= test_orb_camera_motion_estimator_recovers_translation();
  ok &= test_recent_track_wins_before_stale_track();
  ok &= test_tracker_drops_track_after_missing_budget();
  ok &= test_zero_missing_budget_keeps_continuous_track();
  ok &= test_tracker_recovers_after_exact_missing_budget();
  ok &= test_tracker_matches_tiny_non_overlapping_boxes_by_motion();
  ok &= test_tracker_can_disable_center_distance_matching();
  ok &= test_iou_only_tracker_does_not_apply_motion_prediction();
  ok &= test_low_score_detection_recovers_but_does_not_create_track();
  ok &= test_tracker_confirmation_suppresses_single_frame_noise();
  ok &= test_global_assignment_avoids_greedy_identity_loss();
  ok &= test_prediction_bridges_one_high_confidence_gap();
  ok &= test_unconfirmed_track_expires_on_first_miss();
  ok &= test_track_overlay_uses_stable_per_identity_colors();
  ok &= test_tracker_does_not_revive_after_missing_budget();
  ok &= test_tracker_enforces_monotonic_frames_without_active_tracks();
  ok &= test_tracker_rejects_non_finite_thresholds();
  return ok ? 0 : 1;
}
