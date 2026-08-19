#include "examples/tracking/yolo26-tiny-drone-tracker/src/cpp/utils/runtime_api.cpp"
#include "examples/tracking/yolo26-tiny-drone-tracker/src/cpp/utils/tracker_api.cpp"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using yolo26_tiny_drone_tracker::append_decode_output_nodes;
using yolo26_tiny_drone_tracker::configure_output_fps;
using yolo26_tiny_drone_tracker::Detection;
using yolo26_tiny_drone_tracker::ObjectTracker;
using yolo26_tiny_drone_tracker::pull_status_has_sample;
using yolo26_tiny_drone_tracker::samples_correlate;
using yolo26_tiny_drone_tracker::TrackerConfig;

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
  const std::string temp_dir = create_test_scratch_dir("yolo26-tiny-drone-tracker", test_name);
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
                                            "  path: models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz\n"
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
                      "validate output reports total inflight limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_too_many_streams",
                                            "model:\n"
                                            "  path: models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz\n"
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
                   "  path: models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz\n"
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

bool test_validate_config_only_checks_full_port_ranges(const std::string& binary) {
  const std::string prefix =
      "model:\n"
      "  path: models/yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz\n"
      "streams:\n"
      "  - rtsp://127.0.0.1:8554/src1\n"
      "  - rtsp://127.0.0.1:8554/src2\n"
      "  - rtsp://127.0.0.1:8554/src3\n"
      "  - rtsp://127.0.0.1:8554/src4\n"
      "output:\n"
      "  insight:\n"
      "    host: 127.0.0.1\n";

  const fs::path valid_video_path = write_config(
      "test_validate_config_only_accepts_last_video_port_at_udp_limit",
      prefix + "    video_port_base: 65532\n    metadata_port_base: 9000\n");
  const auto valid_video = spawn_and_wait(
      binary, {"--config", valid_video_path.string(), "--validate-config-only"}, 20000);

  const fs::path valid_metadata_path = write_config(
      "test_validate_config_only_accepts_last_metadata_port_at_udp_limit",
      prefix + "    video_port_base: 9000\n    metadata_port_base: 65532\n");
  const auto valid_metadata = spawn_and_wait(
      binary, {"--config", valid_metadata_path.string(), "--validate-config-only"}, 20000);

  const fs::path video_path = write_config(
      "test_validate_config_only_rejects_video_port_overflow",
      prefix + "    video_port_base: 65533\n    metadata_port_base: 9000\n");
  const auto video =
      spawn_and_wait(binary, {"--config", video_path.string(), "--validate-config-only"}, 20000);

  const fs::path metadata_path = write_config(
      "test_validate_config_only_rejects_metadata_port_overflow",
      prefix + "    video_port_base: 9000\n    metadata_port_base: 65533\n");
  const auto metadata = spawn_and_wait(
      binary, {"--config", metadata_path.string(), "--validate-config-only"}, 20000);

  const fs::path overlap_path = write_config(
      "test_validate_config_only_rejects_overlapping_port_ranges",
      prefix + "    video_port_base: 9000\n    metadata_port_base: 9001\n");
  const auto overlap = spawn_and_wait(
      binary, {"--config", overlap_path.string(), "--validate-config-only"}, 20000);

  const fs::path reverse_overlap_path = write_config(
      "test_validate_config_only_rejects_reverse_overlapping_port_ranges",
      prefix + "    video_port_base: 9001\n    metadata_port_base: 9000\n");
  const auto reverse_overlap = spawn_and_wait(
      binary, {"--config", reverse_overlap_path.string(), "--validate-config-only"}, 20000);

  const fs::path metadata_only_path = write_config(
      "test_validate_config_only_allows_overlap_when_video_is_disabled",
      prefix + "    video_port_base: 9000\n    metadata_port_base: 9000\n  video_enabled: false\n");
  const auto metadata_only = spawn_and_wait(
      binary, {"--config", metadata_only_path.string(), "--validate-config-only"}, 20000);

  const bool ok =
      expect_true(valid_video.exit_code == 0, "last assigned video port may equal 65535") &&
      expect_true(valid_metadata.exit_code == 0, "last assigned metadata port may equal 65535") &&
      expect_true(video.exit_code == 1, "overflowing video port range is rejected") &&
      expect_contains(video.stderr_text, "video_port_base must be between 1 and 65532",
                      "video port error reports the safe base range") &&
      expect_true(metadata.exit_code == 1, "overflowing metadata port range is rejected") &&
      expect_contains(metadata.stderr_text, "metadata_port_base must be between 1 and 65532",
                      "metadata port error reports the safe base range") &&
      expect_true(overlap.exit_code == 1, "overlapping port ranges are rejected") &&
      expect_contains(overlap.stderr_text, "port ranges must not overlap",
                      "overlapping port error explains the conflict") &&
      expect_true(reverse_overlap.exit_code == 1,
                  "reverse overlapping port ranges are rejected") &&
      expect_true(metadata_only.exit_code == 0,
                  "overlap is allowed when video output is disabled");
  remove_dir(valid_video_path.parent_path().string());
  remove_dir(valid_metadata_path.parent_path().string());
  remove_dir(video_path.parent_path().string());
  remove_dir(metadata_path.parent_path().string());
  remove_dir(overlap_path.parent_path().string());
  remove_dir(reverse_overlap_path.parent_path().string());
  remove_dir(metadata_only_path.parent_path().string());
  return ok;
}

bool test_requested_fps_configures_videorate() {
  simaai::neat::nodes::groups::RtspDecodedInputOptions options;
  options.codec = simaai::neat::nodes::groups::RtspCodec::H264;
  options.source_fps = 30;
  options.dec_width = 640;
  options.dec_height = 512;
  options.output_caps.enable = true;
  options.output_caps.format = "NV12";
  options.output_caps.width = 640;
  options.output_caps.height = 512;

  const int output_fps = configure_output_fps(options, 30, 10);
  simaai::neat::Graph graph("decode_output");
  append_decode_output_nodes(graph, options);
  return expect_true(output_fps == 10, "configured FPS becomes the output rate") &&
         expect_true(options.source_fps == 30, "configured FPS preserves source cadence") &&
         expect_true(options.use_videorate, "configured FPS enables videorate") &&
         expect_true(options.video_rate_fps == 10, "configured FPS sets videorate target") &&
         expect_true(options.output_caps.fps == 10, "configured FPS sets output caps") &&
         expect_contains(graph.describe(), "VideoRate", "decode graph contains videorate");
}

bool test_closed_detection_output_is_terminal() {
  simaai::neat::PullError pull_error;
  try {
    (void)pull_status_has_sample(simaai::neat::PullStatus::Closed, "detections", pull_error,
                                 "source reached EOS");
  } catch (const std::runtime_error& error) {
    return expect_contains(error.what(), "detections output closed unexpectedly",
                           "closed output reports a terminal error") &&
           expect_contains(error.what(), "source reached EOS",
                           "closed output preserves runtime context");
  }
  return expect_true(false, "closed output throws instead of spinning");
}

bool test_debug_frames_require_matching_identity() {
  simaai::neat::Sample detection;
  detection.frame_id = 42;
  detection.pts_ns = 1'000'000;
  simaai::neat::Sample matching_frame;
  matching_frame.frame_id = 42;
  matching_frame.pts_ns = 1'000'000;
  simaai::neat::Sample newer_frame;
  newer_frame.frame_id = 43;
  newer_frame.pts_ns = 2'000'000;
  simaai::neat::Sample pts_only_detection;
  pts_only_detection.pts_ns = 3'000'000;
  simaai::neat::Sample pts_only_frame;
  pts_only_frame.pts_ns = 3'000'000;
  simaai::neat::Sample partially_identified_frame;
  partially_identified_frame.frame_id = 42;
  partially_identified_frame.pts_ns = 3'000'000;

  return expect_true(samples_correlate(detection, matching_frame),
                     "matching frame IDs correlate") &&
         expect_true(!samples_correlate(detection, newer_frame),
                     "different frame IDs do not correlate") &&
         expect_true(samples_correlate(pts_only_detection, pts_only_frame),
                     "matching PTS correlates when frame IDs are absent") &&
         expect_true(!samples_correlate(pts_only_detection, partially_identified_frame),
                     "PTS does not override a partially available frame ID");
}

bool test_tracker_reuses_track_id_for_nearby_detection() {
  TrackerConfig config;
  config.match_iou_threshold = 0.3f;
  config.max_missing_frames = 2;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{10.0f, 10.0f, 50.0f, 80.0f, 0.9f, 0}}, 0);
  const auto second = tracker.update({Detection{12.0f, 11.0f, 52.0f, 81.0f, 0.8f, 0}}, 1);
  return expect_true(first.size() == 1, "tracker returns one detection on first frame") &&
         expect_true(second.size() == 1, "tracker returns one detection on second frame") &&
         expect_true(first.front().track_id == second.front().track_id,
                     "tracker reuses track id for nearby detection");
}

bool test_tracker_drops_track_after_missing_budget() {
  TrackerConfig config;
  config.match_iou_threshold = 0.3f;
  config.max_missing_frames = 1;
  ObjectTracker tracker(config);
  tracker.update({Detection{10.0f, 10.0f, 50.0f, 80.0f, 0.9f, 0}}, 0);
  tracker.update({}, 1);
  tracker.update({}, 2);
  return expect_true(tracker.active_track_count() == 0,
                     "tracker expires track after missing frame budget");
}

bool test_tracker_matches_zero_iou_motion() {
  TrackerConfig config;
  config.match_iou_threshold = 0.3f;
  config.max_center_distance = 3.0f;
  config.velocity_momentum = 0.0f;
  config.center_distance_enabled = true;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0}}, 0);
  const auto second = tracker.update({Detection{20.0f, 0.0f, 30.0f, 10.0f, 0.8f, 0}}, 1);
  const auto third = tracker.update({Detection{40.0f, 0.0f, 50.0f, 10.0f, 0.8f, 0}}, 2);
  return expect_true(first.size() == 1 && second.size() == 1 && third.size() == 1,
                     "motion tracker returns all zero-IoU detections") &&
         expect_true(first.front().track_id == second.front().track_id &&
                         second.front().track_id == third.front().track_id,
                     "motion tracker preserves identity across zero-IoU movement");
}

bool test_low_score_detection_only_recovers_confirmed_track() {
  TrackerConfig config;
  config.high_score_threshold = 0.5f;
  config.new_track_threshold = 0.5f;
  config.match_iou_threshold = 0.1f;
  config.min_confirmed_hits = 2;
  ObjectTracker tracker(config);
  const Detection high{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0};
  const Detection low{0.0f, 0.0f, 10.0f, 10.0f, 0.2f, 0};

  const auto tentative = tracker.update({high}, 0);
  const auto low_before_confirmation = tracker.update({low}, 1);
  const auto confirmed = tracker.update({high}, 2);
  const auto recovered = tracker.update({low}, 3);

  return expect_true(tentative.empty(), "tentative track is not published") &&
         expect_true(low_before_confirmation.empty(),
                     "low-score detection cannot confirm tentative track") &&
         expect_true(confirmed.size() == 1 && recovered.size() == 1,
                     "confirmed track accepts low-score recovery") &&
         expect_true(confirmed.front().track_id == recovered.front().track_id,
                     "low-score recovery preserves confirmed identity");
}

bool test_tracker_bounds_active_state() {
  TrackerConfig config;
  config.max_active_tracks = 2;
  ObjectTracker tracker(config);
  const std::vector<Detection> detections = {
      Detection{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0},
      Detection{20.0f, 0.0f, 30.0f, 10.0f, 0.9f, 0},
      Detection{40.0f, 0.0f, 50.0f, 10.0f, 0.9f, 0},
  };
  const auto tracked = tracker.update(detections, 0);
  return expect_true(tracked.size() == 2, "tracker publishes only bounded new tracks") &&
         expect_true(tracker.active_track_count() == 2, "tracker bounds active state");
}

bool test_tracker_expires_stale_state_before_creating_replacement() {
  TrackerConfig config;
  config.max_active_tracks = 1;
  config.max_missing_frames = 0;
  config.center_distance_enabled = false;
  ObjectTracker tracker(config);
  const auto first = tracker.update({Detection{0.0f, 0.0f, 10.0f, 10.0f, 0.9f, 0}}, 0);
  const auto replacement =
      tracker.update({Detection{100.0f, 0.0f, 110.0f, 10.0f, 0.9f, 0}}, 1);
  return expect_true(replacement.size() == 1, "expired track capacity accepts replacement") &&
         expect_true(replacement.front().track_id != first.front().track_id,
                     "replacement receives a new track id") &&
         expect_true(tracker.active_track_count() == 1,
                     "replacement preserves the active-track bound");
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
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_invalid_inflight_limit(binary);
  ok &= test_validate_config_only_checks_full_port_ranges(binary);
  ok &= test_requested_fps_configures_videorate();
  ok &= test_closed_detection_output_is_terminal();
  ok &= test_debug_frames_require_matching_identity();
  ok &= test_tracker_reuses_track_id_for_nearby_detection();
  ok &= test_tracker_drops_track_after_missing_budget();
  ok &= test_tracker_matches_zero_iou_motion();
  ok &= test_low_score_detection_only_recovers_confirmed_track();
  ok &= test_tracker_bounds_active_state();
  ok &= test_tracker_expires_stale_state_before_creating_replacement();
  return ok ? 0 : 1;
}
