#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

using multi_stream_people_tracker::Detection;
using multi_stream_people_tracker::ObjectTracker;
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
  ok &= test_tracker_drops_track_after_missing_budget();
  ok &= test_zero_missing_budget_keeps_continuous_track();
  ok &= test_tracker_recovers_after_exact_missing_budget();
  ok &= test_tracker_matches_tiny_non_overlapping_boxes_by_motion();
  ok &= test_tracker_can_disable_center_distance_matching();
  ok &= test_iou_only_tracker_does_not_apply_motion_prediction();
  ok &= test_low_score_detection_recovers_but_does_not_create_track();
  ok &= test_tracker_confirmation_suppresses_single_frame_noise();
  ok &= test_tracker_does_not_revive_after_missing_budget();
  ok &= test_tracker_enforces_monotonic_frames_without_active_tracks();
  ok &= test_tracker_rejects_non_finite_thresholds();
  return ok ? 0 : 1;
}
