// Unit tests for the tiny-drone tracker: the checks the application leaves to its config, and the
// tracker's matching rules. tests/python/test_unit.py runs the same tests in Python. This example
// also holds the test of the shared C++ detection_egress::serialize_tracks, since support/ has no
// C++ test program; tests/scripts/test_detection_egress.py tests the Python one.

#include "examples/tracking/yolo26-tiny-drone-tracker/src/cpp/utils/tracker.h"
#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using drone_tracker::ObjectTracker;
using drone_tracker::TrackedDetection;
using drone_tracker::TrackerSettings;
using sima_examples::testing::expect_true;

namespace {

simaai::neat::Box box(float x1, float y1, float x2, float y2, float score = 0.9f) {
  return {x1, y1, x2, y2, score, 0};
}

constexpr int kFps = 30;
constexpr std::int64_t kFrameNs = 33'333'333;

// The presentation time of a 30 FPS stream's frame, from an origin that is not 0.
std::int64_t pts(int frame) {
  return 5'000'000'000 + frame * kFrameNs;
}

TrackerSettings settings() {
  TrackerSettings values;
  values.high_score_threshold = 0.5;
  values.new_track_threshold = 0.5;
  values.max_missing_frames = 30;
  values.min_confirmed_hits = 1;
  return values;
}

std::vector<int> ids(const std::vector<TrackedDetection>& tracks) {
  std::vector<int> result;
  for (const auto& track : tracks) {
    result.push_back(track.track_id);
  }
  return result;
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_is_usable() {
  const fs::path common_dir = "examples/tracking/yolo26-tiny-drone-tracker/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const std::string codec = config.string("input.codec");
  const auto sources = config.string_list("input.source");
  bool ok = expect_true(!sources.empty() && sources.size() <= 4U,
                        "the shipped config lists one to four streams");
  ok &= expect_true(codec == "h264" || codec == "h265", "the shipped codec is supported");
  const fs::path model_path = common_dir / config.string("model.path");
  ok &= expect_true(model_path.filename() == "yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz",
                    "model.path names the tiny-drone package");
  // The packaged path reaches prebuilt-apps/models from the config file.
  ok &= expect_true(fs::absolute(model_path).lexically_normal().parent_path() ==
                        fs::absolute("models").lexically_normal(),
                    "model.path reaches models/ from the config file");
  ok &= expect_true(
      config.integer("input.width") >= 0 && config.integer("input.height") >= 0 &&
          config.integer("input.fps") >= 0 && config.integer("input.rtsp.latency_ms") >= 0 &&
          config.integer("inference.frames") >= 0 && config.integer("inference.max_detections") > 0,
      "the shipped sizes, latency, and frame counts are in range");
  ok &= expect_true(config.number("inference.nms_iou") >= 0.0 &&
                        config.number("inference.nms_iou") <= 1.0,
                    "the shipped NMS IoU is from 0 to 1");
  for (const std::string key :
       {"output.insight.video_port_base", "output.insight.metadata_port_base"}) {
    ok &= expect_true(config.integer(key) >= 1 && config.integer(key) <= 65535, key + " is a port");
  }
  // Core sends to 127.0.0.1 when the host is empty.
  ok &=
      expect_true(!config.string("output.insight.host").empty(), "the shipped Insight host is set");
  // Read for its type only: true and false are both valid.
  (void)config.boolean("input.rtsp.tcp");
  // Detections below inference.min_score never reach the tracker, so a lower
  // high_score_threshold would make every detection a high-score one.
  const double min_score = config.number("inference.min_score");
  const double high = config.number("app.tracking.high_score_threshold");
  const double start = config.number("app.tracking.new_track_threshold");
  ok &= expect_true(0.0 <= min_score && min_score <= high && high <= start && start <= 1.0,
                    "0 <= min_score <= high_score_threshold <= new_track_threshold <= 1");
  ok &= expect_true(config.integer("app.tracking.max_missing_frames") >= 0 &&
                        config.integer("app.tracking.min_confirmed_hits") >= 1,
                    "the shipped track lifetime and confirmation count are in range");
  return ok;
}

bool test_nearby_detection_keeps_its_track_id() {
  ObjectTracker object_tracker(settings(), kFps);
  bool ok = expect_true(ids(object_tracker.update({box(10, 10, 50, 80)}, pts(0))) == std::vector{1},
                        "the first box starts track 1");
  ok &=
      expect_true(ids(object_tracker.update({box(12, 11, 52, 81, 0.8f)}, pts(1))) == std::vector{1},
                  "a box 2 px away continues track 1");
  return ok;
}

bool test_higher_affinity_wins_then_lower_track_id() {
  const auto two_tracks = [](simaai::neat::Box first, simaai::neat::Box second) {
    ObjectTracker object_tracker(settings(), kFps);
    object_tracker.update({first, second}, pts(0));
    return object_tracker;
  };
  // Affinity is IoU plus 1 / (1 + center distance). Track 1 overlaps the box with IoU 0.64 at
  // 0.11 diagonals, track 2 with IoU 0.04 at 0.08 diagonals: track 1 scores 1.54, track 2 0.96.
  auto big_and_near = two_tracks(box(0, 0, 25, 25), box(9, 9, 13, 13));
  bool ok = expect_true(ids(big_and_near.update({box(0, 0, 20, 20)}, pts(1))) == std::vector{1},
                        "the track with the higher affinity wins");
  // No overlap with either track: the closer one wins, at 0.85 diagonals against 1.27.
  auto apart = two_tracks(box(0, 0, 10, 10), box(30, 0, 40, 10));
  ok &= expect_true(ids(apart.update({box(18, 0, 28, 10)}, pts(1))) == std::vector{2},
                    "with no overlap, the closer track wins");
  // The same IoU (0.54) and center distance with both tracks: the lower ID wins.
  auto side_by_side = two_tracks(box(0, 0, 10, 10), box(6, 0, 16, 10));
  ok &= expect_true(ids(side_by_side.update({box(3, 0, 13, 10)}, pts(1))) == std::vector{1},
                    "on equal affinity, the lower track ID wins");
  return ok;
}

bool test_track_ends_after_max_missing_frames() {
  bool ok = true;
  for (const auto& [max_missing_frames, track_id] : {std::pair{2, 1}, std::pair{1, 2}}) {
    TrackerSettings values = settings();
    values.max_missing_frames = max_missing_frames;
    ObjectTracker object_tracker(values, kFps);
    object_tracker.update({box(10, 10, 50, 80)}, pts(0));
    object_tracker.update({}, pts(1));
    object_tracker.update({}, pts(2));
    ok &= expect_true(
        ids(object_tracker.update({box(10, 10, 50, 80)}, pts(3))) == std::vector{track_id},
        "after 2 missed frames with max_missing_frames " + std::to_string(max_missing_frames) +
            ", the box gets track " + std::to_string(track_id));
  }
  return ok;
}

// The shared model's link drops frames 1 and 2, so the tracker sees frames 0 and 3.
bool test_dropped_frames_age_a_track() {
  bool ok = true;
  for (const auto& [max_missing_frames, track_id] : {std::pair{2, 1}, std::pair{1, 2}}) {
    TrackerSettings values = settings();
    values.max_missing_frames = max_missing_frames;
    ObjectTracker object_tracker(values, kFps);
    object_tracker.update({box(10, 10, 50, 80)}, pts(0));
    ok &= expect_true(
        ids(object_tracker.update({box(10, 10, 50, 80)}, pts(3))) == std::vector{track_id},
        "after 2 dropped frames with max_missing_frames " + std::to_string(max_missing_frames) +
            ", the box gets track " + std::to_string(track_id));
  }
  return ok;
}

bool test_predicted_motion_keeps_a_fast_box_on_its_track() {
  ObjectTracker object_tracker(settings(), kFps);
  object_tracker.update({box(0, 0, 10, 10)}, pts(0));
  // 40 px is 2.8 box diagonals, within the 3-diagonal gate, and sets a velocity of 10 px.
  bool ok = expect_true(ids(object_tracker.update({box(40, 0, 50, 10)}, pts(1))) == std::vector{1},
                        "a box 40 px away continues track 1");
  // 44 px from the last box is 3.1 diagonals, outside the gate. The predicted box is 34 px away,
  // 2.4 diagonals, so the track continues.
  ok &= expect_true(ids(object_tracker.update({box(84, 0, 94, 10)}, pts(2))) == std::vector{1},
                    "the predicted motion keeps the box on track 1");
  return ok;
}

bool test_velocity_takes_a_quarter_of_the_measured_motion() {
  bool ok = true;
  for (const auto& third_box : {box(90, 0, 100, 10), box(10, 0, 20, 10)}) {
    ObjectTracker object_tracker(settings(), kFps);
    object_tracker.update({box(0, 0, 10, 10)}, pts(0));
    object_tracker.update({box(40, 0, 50, 10)}, pts(1));
    // The 40 px move sets a velocity of 10 px, so the box is predicted at x 50..60. Both third
    // boxes are 40 px from there, within the 42.4 px gate. A velocity below 7.6 px loses the
    // first, and one above 12.4 px loses the second.
    ok &= expect_true(ids(object_tracker.update({third_box}, pts(2))) == std::vector{1},
                      "a box 40 px from the predicted box at x " +
                          std::to_string(static_cast<int>(third_box.x1)) + " continues track 1");
  }
  return ok;
}

bool test_box_beyond_the_distance_gate_starts_a_new_track() {
  ObjectTracker object_tracker(settings(), kFps);
  object_tracker.update({box(0, 0, 10, 10)}, pts(0));
  // 50 px is 3.5 box diagonals from the predicted box, outside the 3-diagonal gate.
  return expect_true(ids(object_tracker.update({box(50, 0, 60, 10)}, pts(1))) == std::vector{2},
                     "a box 50 px away starts track 2");
}

bool test_track_is_reported_from_its_min_confirmed_hits_th_detection() {
  TrackerSettings values = settings();
  values.min_confirmed_hits = 3;
  ObjectTracker object_tracker(values, kFps);
  bool ok = expect_true(object_tracker.update({box(0, 0, 10, 10)}, pts(0)).empty() &&
                            object_tracker.update({box(0, 0, 10, 10)}, pts(1)).empty(),
                        "the first two detections report no track");
  ok &= expect_true(ids(object_tracker.update({box(0, 0, 10, 10)}, pts(2))) == std::vector{1},
                    "the third detection reports track 1");
  return ok;
}

bool test_low_score_detection_continues_only_a_confirmed_track() {
  TrackerSettings values = settings();
  values.min_confirmed_hits = 2;
  ObjectTracker object_tracker(values, kFps);
  const auto high = box(0, 0, 10, 10, 0.9f);
  const auto low = box(0, 0, 10, 10, 0.2f);
  bool ok = expect_true(object_tracker.update({high}, pts(0)).empty(), "one hit reports no track");
  ok &= expect_true(object_tracker.update({low}, pts(1)).empty(),
                    "a low score cannot confirm the new track");
  ok &= expect_true(ids(object_tracker.update({high}, pts(2))) == std::vector{1},
                    "a second high score confirms track 1");
  ok &= expect_true(ids(object_tracker.update({low}, pts(3))) == std::vector{1},
                    "a confirmed track continues on a low score");
  ok &= expect_true(object_tracker.tracks_started() == 1, "one track started");
  return ok;
}

bool test_new_track_needs_new_track_threshold() {
  TrackerSettings values = settings();
  values.high_score_threshold = 0.2;
  values.new_track_threshold = 0.3;
  ObjectTracker object_tracker(values, kFps);
  bool ok = expect_true(object_tracker.update({box(0, 0, 10, 10, 0.25f)}, pts(0)).empty() &&
                            object_tracker.tracks_started() == 0,
                        "a score below new_track_threshold starts no track");
  ok &= expect_true(ids(object_tracker.update({box(0, 0, 10, 10, 0.3f)}, pts(1))) == std::vector{1},
                    "a score at new_track_threshold starts track 1");
  return ok;
}

bool test_tracks_stop_at_max_active_tracks() {
  ObjectTracker object_tracker(settings(), kFps);
  std::vector<simaai::neat::Box> boxes;
  for (int index = 0; index < 300; ++index) {
    boxes.push_back(box(20.0f * index, 0, 20.0f * index + 10, 10));
  }
  return expect_true(object_tracker.update(boxes, pts(0)).size() == 256U &&
                         drone_tracker::kMaxActiveTracks == 256 &&
                         object_tracker.tracks_started() == 256,
                     "300 boxes start 256 tracks");
}

// The tracking message carries each track's ID, label, score, and box as [x, y, width, height],
// in the envelope the object-detection message uses. decode_bbox_tensor already clamps every
// corner to the frame. The second box ends past the right edge, as a box from another source may,
// and is cut at the frame.
bool test_serialize_tracks_builds_the_tracking_message() {
  const std::vector<TrackedDetection> tracks{
      {7, 10.9f, 2.0f, 35.9f, 20.8f, 0.75f, 0},
      {12, 90.0f, 80.0f, 120.0f, 95.0f, 0.5f, 3},
  };
  sima_examples::detection_egress::FrameMetadata frame;
  frame.stream_id = "decoder";
  frame.frame_id = 42;
  frame.pts_ns = 1'234'567'890;
  const auto parsed = nlohmann::json::parse(sima_examples::detection_egress::serialize_tracks(
      tracks, std::vector<std::string>{"drone"}, 100, 100, frame));
  const nlohmann::json expected_tracks = {
      {{"id", "7"},
       {"label", "drone"},
       {"confidence", 0.75f},
       {"bbox", {10.0f, 2.0f, 25.0f, 18.0f}}},
      {{"id", "12"},
       {"label", "unknown"},
       {"confidence", 0.5f},
       {"bbox", {90.0f, 80.0f, 10.0f, 15.0f}}},
  };
  bool ok = expect_true(parsed.at("type") == "tracking" && parsed.at("timestamp") == 1234 &&
                            parsed.at("frame_id") == "42" && parsed.at("stream_id") == "decoder",
                        "the message carries the type and the frame envelope");
  ok &= expect_true(parsed.at("data") == nlohmann::json{{"tracks", expected_tracks}},
                    "each track keeps its ID and label, and a box past the edge ends there: " +
                        parsed.at("data").dump());
  return ok;
}

} // namespace

int main() {
  bool ok = test_shipped_config_is_usable();
  ok &= test_nearby_detection_keeps_its_track_id();
  ok &= test_higher_affinity_wins_then_lower_track_id();
  ok &= test_track_ends_after_max_missing_frames();
  ok &= test_dropped_frames_age_a_track();
  ok &= test_predicted_motion_keeps_a_fast_box_on_its_track();
  ok &= test_velocity_takes_a_quarter_of_the_measured_motion();
  ok &= test_box_beyond_the_distance_gate_starts_a_new_track();
  ok &= test_track_is_reported_from_its_min_confirmed_hits_th_detection();
  ok &= test_low_score_detection_continues_only_a_confirmed_track();
  ok &= test_new_track_needs_new_track_threshold();
  ok &= test_tracks_stop_at_max_active_tracks();
  ok &= test_serialize_tracks_builds_the_tracking_message();
  return ok ? 0 : 1;
}
