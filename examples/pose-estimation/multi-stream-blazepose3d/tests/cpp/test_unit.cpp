#include "examples/pose-estimation/multi-stream-blazepose3d/src/cpp/pose_logic.h"
#include "examples/pose-estimation/multi-stream-blazepose3d/src/cpp/yaml_config.h"
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_process.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

fs::path write_config(const std::string& test_name, const std::string& streams) {
  const fs::path directory = create_test_scratch_dir("multi-stream-blazepose3d", test_name);
  const fs::path path = directory / "config.yaml";
  std::ofstream output(path);
  output << "models:\n"
            "  detector_path: detector#v2.tar.gz\n"
            "  pose_path: pose#v2.tar.gz # trailing comment\n"
            "streams:\n"
         << streams
         << "output:\n"
            "  insight:\n"
            "    host: 127.0.0.1\n";
  return path;
}

bool test_math_contract() {
  bool ok = true;
  int push_attempts = 0;
  const auto retry_result = blazepose_app::retry_nonblocking_push(
      [&]() {
        ++push_attempts;
        return blazepose_app::NonblockingPushAttempt::Retry;
      },
      [&]() { return push_attempts == 1; }, []() {});
  ok &= expect(retry_result == blazepose_app::NonblockingPushResult::Aborted && push_attempts == 1,
               "non-blocking model admission stops retrying at the frame deadline");
  std::deque<std::optional<int>> detector_outputs{1, 2, 3};
  const auto expired_detector_outputs = blazepose_app::expire_pending_fifo(
      detector_outputs, [](int context) { return context <= 2; });
  ok &= expect(expired_detector_outputs == std::vector<int>({1, 2}) &&
                   !detector_outputs[0].has_value() && !detector_outputs[1].has_value() &&
                   detector_outputs[2] == 3,
               "expired detector contexts become ordered tombstones for late outputs");
  detector_outputs.pop_front();
  ok &= expect(!detector_outputs.front().has_value(),
               "a late detector output consumes its tombstone without shifting correlation");

  blazepose_app::OrderedCompletionQueue<std::optional<int>> publications;
  ok &= expect(publications.complete(2, 2).empty(),
               "ordered publication waits for the prior source frame");
  const auto ready = publications.complete(1, 1);
  ok &= expect(ready.size() == 2 && ready[0] == 1 && ready[1] == 2,
               "ordered publication releases consecutive source frames together");
  const auto skipped = publications.complete(3, std::nullopt);
  ok &= expect(skipped.size() == 1 && !skipped[0].has_value() &&
                   publications.next_sequence() == 4,
               "dropped latest-only work advances publication without metadata");

  const blazepose_app::Box box{10.0F, 20.0F, 30.0F, 60.0F, 0.9F, 0};
  const auto roi = blazepose_app::square_roi(box, 1.5);
  ok &= expect(roi.x == -10 && roi.y == 10 && roi.width == 60 && roi.height == 60,
               "ROI is square, centered, scaled, and may extend beyond the frame");

  std::vector<float> raw(39 * 5, 0.0F);
  raw[0] = 4.0F;
  raw[1] = 8.0F;
  raw[3] = 2.0F;
  raw[4] = -2.0F;
  std::vector<float> raw_world(39 * 3, 0.0F);
  raw_world[0] = 0.1F;
  raw_world[1] = -0.2F;
  raw_world[2] = 0.3F;
  const blazepose_app::Affine affine{2.0, 0.0, 10.0, 0.0, 3.0, 20.0};
  const auto pose = blazepose_app::decode_pose(raw, raw_world, affine, box, 0.93F, 2);
  ok &= expect(std::abs(pose.presence - 0.93F) < 0.001F,
               "global pose presence is retained for publication");
  ok &= expect(std::abs(blazepose_app::sigmoid(0.0F) - 0.5F) < 0.001F &&
                   blazepose_app::sigmoid(-0.01F) < 0.5F,
               "global pose presence logits activate before probability thresholding");
  ok &= expect(std::abs(pose.keypoints[0].x - 18.0F) < 0.001F &&
                   std::abs(pose.keypoints[0].y - 44.0F) < 0.001F,
               "landmarks are mapped back through ROI affine metadata");
  ok &= expect(std::abs(pose.keypoints[0].confidence - blazepose_app::sigmoid(-2.0F)) < 0.001F,
               "confidence is min(sigmoid(visibility), sigmoid(presence))");
  ok &= expect(std::abs(pose.world_keypoints[0].x - 0.1F) < 0.001F &&
                   std::abs(pose.world_keypoints[0].y + 0.2F) < 0.001F &&
                   std::abs(pose.world_keypoints[0].z - 0.3F) < 0.001F,
               "world landmarks retain the model's 3D coordinates");
  const auto data = blazepose_app::poses_data_json({pose}, "camera0");
  ok &= expect(data.at("stream_id") == "camera0",
               "2D metadata preserves the source stream identity");
  ok &= expect(data["poses"][0]["keypoints"].size() == 33,
               "Insight metadata contains exactly 33 body keypoints");
  ok &= expect(std::abs(data["poses"][0]["presence"].get<float>() - 0.93F) < 0.001F,
               "2D metadata publishes global pose presence");
  ok &= expect(data["poses"][0]["keypoints"][0]["name"] == "nose",
               "Insight metadata uses BlazePose landmark names");
  ok &= expect(data["poses"][0]["world_keypoints"].size() == 33 &&
                   data["poses"][0]["world_keypoints"][0]["name"] == "nose" &&
                   std::abs(data["poses"][0]["world_keypoints"][0]["z"].get<float>() - 0.3F) <
                       0.001F,
               "pose-estimation metadata includes named world keypoints");
  const auto auxiliary =
      blazepose_app::world_pose_auxiliary_data_json({pose}, "camera0");
  ok &= expect(auxiliary.at("stream_id") == "camera0",
               "auxiliary metadata preserves the source stream identity");
  ok &= expect(auxiliary["schema_version"] == 1 && auxiliary["id"] == "world-pose" &&
                   auxiliary["renderer"] == "blazepose-3d",
               "world landmarks use the generic auxiliary visualization envelope");
  ok &= expect(auxiliary["payload"]["poses"][0]["keypoints"].size() == 33 &&
                   auxiliary["payload"]["poses"][0]["keypoints"][0]["name"] == "nose",
               "auxiliary payload contains 33 named world keypoints");
  ok &= expect(
      std::abs(auxiliary["payload"]["poses"][0]["presence"].get<float>() - 0.93F) < 0.001F,
      "3D metadata publishes the same global pose presence");
  const nlohmann::json point_cloud = {
      {"points", {{{"x", 0.1}, {"y", 0.2}, {"z", 0.3}, {"value", 7}}}},
      {"axes", {"east", "north", "up"}},
  };
  const auto generic = blazepose_app::auxiliary_visualization_data_json(
      "depth-cloud", "point-cloud-3d", point_cloud);
  ok &= expect(generic["schema_version"] == 1 && generic["id"] == "depth-cloud" &&
                   generic["renderer"] == "point-cloud-3d" && generic["payload"] == point_cloud &&
                   !generic.contains("title"),
               "generic auxiliary envelope preserves an arbitrary renderer payload");
  ok &= expect(blazepose_app::select_frame_id(9, 8, 7, 6) == 9 &&
                   blazepose_app::select_frame_id(-1, 8, 7, 6) == 8 &&
                   blazepose_app::select_frame_id(-1, -1, 7, 6) == 7 &&
                   blazepose_app::select_frame_id(-1, -1, -1, 6) == 6,
               "frame identity falls back through source sequence fields");
  ok &= expect(!blazepose_app::stream_is_drained(true, 1, 0, 8) &&
                   blazepose_app::stream_is_drained(true, 0, 0, 8) &&
                   blazepose_app::stream_is_drained(false, 3, 8, 8),
               "closed streams drain admitted frames before finite-run completion");
  ok &= expect(blazepose_app::stream_can_admit_frame(6, 1, 8) &&
                   !blazepose_app::stream_can_admit_frame(6, 2, 8) &&
                   !blazepose_app::stream_can_admit_frame(8, 0, 8) &&
                   blazepose_app::stream_can_admit_frame(8, 100, 0),
               "finite runs never admit more frames than their remaining per-stream limit");

  blazepose_app::Pose first_pose;
  first_pose.presence = 0.9F;
  first_pose.box = {0.0F, 0.0F, 100.0F, 100.0F, 0.9F, 0};
  first_pose.keypoints[0] = {50.0F, 50.0F, 0.2F};
  first_pose.world_keypoints[0] = {0.0F, 0.0F, 0.0F, 0.2F};
  blazepose_app::PoseSmoother smoother;
  smoother.filter({first_pose}, 1'000'000'000);
  blazepose_app::Pose second_pose = first_pose;
  second_pose.keypoints[0].x = 54.0F;
  second_pose.keypoints[0].confidence = 0.4F;
  second_pose.world_keypoints[0].x = 0.04F;
  second_pose.world_keypoints[0].confidence = 0.4F;
  const auto smoothed = smoother.filter({second_pose}, 1'040'000'000).front();
  const float image_fraction = (smoothed.keypoints[0].x - 50.0F) / 4.0F;
  const float world_fraction = smoothed.world_keypoints[0].x / 0.04F;
  ok &= expect(image_fraction > 0.45F && image_fraction < 0.90F &&
                   std::abs(image_fraction - world_fraction) < 0.001F,
               "temporal filter applies one adaptive weight to 2D and world landmarks");
  ok &= expect(std::abs(smoothed.keypoints[0].confidence - 0.24F) < 0.001F &&
                   std::abs(smoothed.world_keypoints[0].confidence - 0.24F) < 0.001F,
               "temporal filter damps confidence crossings consistently");

  blazepose_app::Pose fast_pose = second_pose;
  fast_pose.keypoints[0].x = 154.0F;
  const auto fast = smoother.filter({fast_pose}, 1'080'000'000).front();
  ok &= expect(fast.keypoints[0].x > 140.0F,
               "temporal filter follows deliberate fast motion without buffering frames");
  blazepose_app::Pose reset_pose = first_pose;
  reset_pose.keypoints[0].x = 30.0F;
  const auto reset = smoother.filter({reset_pose}, 1'400'000'000).front();
  ok &= expect(std::abs(reset.keypoints[0].x - 30.0F) < 0.001F,
               "temporal filter resets after a discontinuity instead of dragging stale state");

  blazepose_app::PoseSmoother continuity;
  continuity.filter({first_pose}, 2'000'000'000);
  const auto first_gap = continuity.filter({}, 2'040'000'000);
  const auto second_gap = continuity.filter({}, 2'080'000'000);
  const auto expired = continuity.filter({}, 2'120'000'000);
  const bool continuity_sizes = first_gap.size() == 1 && second_gap.size() == 1 && expired.empty();
  ok &= expect(continuity_sizes,
               "temporal filter bridges two missing results without buffering later frames");
  if (continuity_sizes) {
    ok &= expect(first_gap[0].keypoints[0].confidence < first_pose.keypoints[0].confidence &&
                     second_gap[0].keypoints[0].confidence < first_gap[0].keypoints[0].confidence,
                 "coasted pose confidence decays while the estimate is unavailable");
  }
  return ok;
}

bool test_pose_aggregate_claims_are_exclusive() {
  struct Aggregate {
    std::uint64_t sequence = 0;
    int expected = 1;
    int completed = 0;
    bool expired = false;
    std::vector<int> poses;
  };
  const auto aggregate = [](std::uint64_t sequence, int expected) {
    Aggregate value;
    value.sequence = sequence;
    value.expected = expected;
    return value;
  };
  const auto always = [](const Aggregate&) { return true; };
  bool ok = true;

  // The last ROI output completes a live frame, then an expiry pass runs before
  // the puller publishes: the output already claimed the frame.
  std::map<std::uint64_t, Aggregate> aggregates{{7, aggregate(1, 1)}};
  auto completed = blazepose_app::record_pose_output(aggregates, 7, std::optional<int>(42));
  const auto expired = blazepose_app::claim_expired_aggregates(aggregates, always);
  blazepose_app::OrderedCompletionQueue<std::vector<int>> publications;
  std::size_t published = 0;
  for (const auto& claimed : expired) {
    published += publications.complete(claimed.sequence, claimed.poses).size();
  }
  if (completed.has_value()) {
    published += publications.complete(completed->sequence, completed->poses).size();
  }
  ok &= expect(completed.has_value() && completed->poses == std::vector<int>{42} &&
                   expired.empty() && aggregates.empty() && published == 1,
               "a completed pose aggregate is claimed before a racing expiry pass");

  // Expiry claims first and publishes the partial frame; the late output is
  // absorbed by the tombstone and claims nothing.
  aggregates = {{8, aggregate(2, 2)}};
  const bool first_output_claims =
      blazepose_app::record_pose_output(aggregates, 8, std::optional<int>(1)).has_value();
  const auto partial = blazepose_app::claim_expired_aggregates(aggregates, always);
  const bool late_output_claims =
      blazepose_app::record_pose_output(aggregates, 8, std::optional<int>(2)).has_value();
  ok &= expect(!first_output_claims && partial.size() == 1 &&
                   partial[0].poses == std::vector<int>{1} && !late_output_claims &&
                   aggregates.empty(),
               "an expired aggregate publishes partial poses once and absorbs late outputs");
  return ok;
}

bool test_cli(const std::string& binary) {
  bool ok = true;
  const auto help = spawn_and_wait(binary, {"--help"}, 20000);
  ok &= expect(help.exit_code == 0 && help.stdout_text.find("--config") != std::string::npos,
               "help documents the config option");
  const auto missing = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  ok &= expect(missing.exit_code == 2 &&
                   missing.stderr_text.find("config file not found") != std::string::npos,
               "missing configuration fails cleanly");
  return ok;
}

bool test_stream_limit(const std::string& binary) {
  const fs::path config = write_config(
      "dynamic_stream_config",
      "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n    insight_channel: 0\n"
      "  - id: camera1\n    url: rtsp://127.0.0.1/src1\n    codec: hevc\n    insight_channel: 1\n"
      "  - id: camera2\n    url: rtsp://127.0.0.1/src2\n    codec: h264\n    insight_channel: 2\n"
      "  - id: camera3\n    url: rtsp://127.0.0.1/src3\n    codec: h264\n    insight_channel: 3\n"
      "  - id: camera4\n    url: rtsp://127.0.0.1/src4\n    codec: h264\n    insight_channel: 4\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok = expect(
      result.exit_code == 1 &&
          result.stderr_text.find("streams must contain between 1 and 4 entries") !=
              std::string::npos,
      "configuration rejects more than four streams");
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_duplicate_stream_identity(const std::string& binary) {
  const fs::path config = write_config(
      "duplicate_stream_identity",
      "  - id: duplicate\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n    insight_channel: 0\n"
      "  - id: duplicate\n    url: rtsp://127.0.0.1/src1\n    codec: h264\n    insight_channel: "
      "1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect(result.exit_code == 1 &&
                 result.stderr_text.find("stream ids must be unique") != std::string::npos,
             "duplicate stream identities are rejected");
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_insight_ports_do_not_overlap(const std::string& binary) {
  const fs::path directory = create_test_scratch_dir("multi-stream-blazepose3d", "port_overlap");
  const fs::path config = directory / "config.yaml";
  std::ofstream(config) << "models:\n  detector_path: detector.tar.gz\n  pose_path: pose.tar.gz\n"
                           "streams:\n"
                           "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n"
                           "    insight_channel: 0\n"
                           "  - id: camera1\n    url: rtsp://127.0.0.1/src1\n    codec: h264\n"
                           "    insight_channel: 100\n"
                           "output:\n  insight:\n    host: 127.0.0.1\n"
                           "    video_port_base: 9000\n    metadata_port_base: 9100\n";
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok = expect(result.exit_code == 1 &&
                             result.stderr_text.find("video and metadata ports must not overlap") !=
                                 std::string::npos,
                         "Insight video and metadata destinations cannot share a UDP port");
  remove_dir(directory.string());
  return ok;
}

bool test_stream_strings_reject_yaml_non_strings(const std::string& binary) {
  bool ok = true;
  for (const auto& [name, field] :
       std::vector<std::pair<std::string, std::string>>{{"null_url", "url: null"},
                                                        {"numeric_id", "id: 17"},
                                                        {"binary_id", "id: 0b101"},
                                                        {"sexagesimal_id", "id: 1:20"},
                                                        {"date_id", "id: 2026-10-01"},
                                                        {"boolean_url", "url: true"}}) {
    const fs::path config = write_config(
        name, "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n"
              "    insight_channel: 0\n    " +
                  field + "\n");
    const auto result =
        spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
    ok &= expect(result.exit_code == 1 && result.stderr_text.find("must be a string") !=
                                              std::string::npos,
                 name + " is rejected consistently with Python YAML parsing");
    remove_dir(config.parent_path().string());
  }
  return ok;
}

bool test_scalar_config_preserves_yaml_types(const std::string& binary) {
  bool ok = true;
  struct Case {
    std::string name;
    std::string detector;
    std::string pose;
    std::string host;
  };
  for (const Case& test : {Case{"numeric_detector", "123", "pose.tar.gz", "127.0.0.1"},
                           Case{"boolean_pose", "detector.tar.gz", "false", "127.0.0.1"},
                           Case{"numeric_host", "detector.tar.gz", "pose.tar.gz", "127"}}) {
    const fs::path directory = create_test_scratch_dir("multi-stream-blazepose3d", test.name);
    const fs::path config = directory / "config.yaml";
    std::ofstream(config) << "models:\n  detector_path: " << test.detector
                          << "\n  pose_path: " << test.pose
                          << "\nstreams:\n  - id: camera0\n"
                             "    url: rtsp://127.0.0.1/src0\n    codec: h264\n"
                             "    insight_channel: 0\noutput:\n  insight:\n    host: "
                          << test.host << "\n";
    const auto result =
        spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
    ok &= expect(result.exit_code == 1 && result.stderr_text.find("must be a string") !=
                                              std::string::npos,
                 test.name + " is rejected consistently with Python YAML parsing");
    remove_dir(directory.string());
  }
  return ok;
}

bool test_typed_settings_reject_explicit_null(const std::string& binary) {
  bool ok = true;
  struct Case {
    std::string name;
    std::string section;
    std::string message;
  };
  for (const Case& test :
       {Case{"null_bool", "output:\n  video_enabled: null\n", "must be true or false"},
        Case{"null_int", "detector:\n  max_detections: null\n", "must be an integer"},
        Case{"null_double", "pose:\n  roi_scale: null\n", "must be numeric"}}) {
    const fs::path directory = create_test_scratch_dir("multi-stream-blazepose3d", test.name);
    const fs::path config = directory / "config.yaml";
    std::ofstream(config) << "models:\n  detector_path: detector.tar.gz\n"
                             "  pose_path: pose.tar.gz\n"
                             "streams:\n  - id: camera0\n"
                             "    url: rtsp://127.0.0.1/src0\n    codec: h264\n"
                             "    insight_channel: 0\n"
                          << test.section;
    const auto result =
        spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
    ok &= expect(result.exit_code == 1 &&
                     result.stderr_text.find(test.message) != std::string::npos,
                 test.name + " is rejected consistently with Python YAML parsing");
    remove_dir(directory.string());
  }
  return ok;
}

bool test_stream_mapping_order_and_explicit_caps(const std::string& binary) {
  const fs::path config = write_config(
      "stream_mapping_order",
      "  - url: rtsp://127.0.0.1/ordered#view\n"
      "    width: 1920\n"
      "    id: camera#1 # trailing comment\n"
      "    fps: 30\n"
      "    insight_channel: 0\n"
      "    height: 1080\n"
      "    codec: h264\n"
      "  - codec: h265\n"
      "    insight_channel: 1\n"
      "    id: camera#2\n"
      "    url: rtsp://127.0.0.1/second#view\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const auto raw = blazepose_config::TypedConfig::load(config);
  const bool passed = result.exit_code == 0 &&
                      result.stdout_text.find("streams=2") != std::string::npos &&
                      raw.string_or("models.detector_path", "") == "detector#v2.tar.gz" &&
                      raw.string_or("models.pose_path", "") == "pose#v2.tar.gz";
  if (!passed) {
    std::cerr << result.stderr_text;
  }
  const bool ok =
      expect(passed, "all YAML scalars preserve hashes while recognizing trailing comments");
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_yaml_scalar_parity(const std::string& binary) {
  bool ok = true;
  ok &= expect(blazepose_config::strip_yaml_inline_comment(
                   "    url: rtsp://host/cam's # camera label") == "    url: rtsp://host/cam's ",
               "plain-scalar apostrophes do not retain trailing YAML comments");
  ok &= expect(
      blazepose_config::strip_yaml_inline_comment("    url: \"rtsp://host/cam #1\" # camera label") ==
          "    url: \"rtsp://host/cam #1\" ",
      "quoted-scalar hashes remain part of the configured value");
  ok &= expect(blazepose_config::parse_yaml_integer("010", "octal") == 8 &&
                   blazepose_config::parse_yaml_integer("0b10", "binary") == 2 &&
                   blazepose_config::parse_yaml_integer("1_000", "underscored") == 1000 &&
                   blazepose_config::parse_yaml_integer("1:20", "sexagesimal") == 80,
               "YAML integer spellings use the same values as Python safe_load");
  ok &= expect(blazepose_config::parse_yaml_scalar("08").type ==
                   blazepose_config::YamlScalarType::String,
               "invalid YAML octal spellings remain strings like Python safe_load");
  ok &= expect(blazepose_config::parse_yaml_scalar("models/2026-10-blazepose.tar.gz").type ==
                       blazepose_config::YamlScalarType::String &&
                   blazepose_config::parse_yaml_scalar("2026-10-01T12:34:56.123_456").type ==
                       blazepose_config::YamlScalarType::String &&
                   blazepose_config::parse_yaml_scalar("2026-10-01").type ==
                       blazepose_config::YamlScalarType::Other &&
                   blazepose_config::parse_yaml_scalar("2026-1-1T1:02:03").type ==
                       blazepose_config::YamlScalarType::Other &&
                   blazepose_config::parse_yaml_scalar("2026-10-01T12:34:56Z").type ==
                       blazepose_config::YamlScalarType::Other,
               "only complete YAML dates and timestamps retain non-string types");
  ok &= expect(blazepose_config::parse_yaml_scalar("'rtsp://host/cam''s'").value ==
                       "rtsp://host/cam's" &&
                   blazepose_config::parse_yaml_scalar("\"rtsp://host/cam\\\"east\"").value ==
                       "rtsp://host/cam\"east" &&
                   blazepose_config::parse_yaml_scalar("\"\\u0041\\U0001F680\"").value ==
                       "A\xf0\x9f\x9a\x80",
               "quoted YAML strings decode single, double, and Unicode escapes");

  const fs::path config =
      write_config("yaml_integer_spellings",
                   "  - id: camera0\n    url: rtsp://host/cam's # camera label\n    codec: h264\n"
                   "    insight_channel: 0b10\n");
  std::ofstream(config, std::ios::app) << "runtime:\n  frames: 010\n"
                                          "detector:\n  min_score: 0.5_0\n"
                                          "pose:\n  job_timeout_ms: 1_000\n"
                                          "  roi_scale: 1:20.5\n"
                                          "test:\n  nan: .NaN\n";
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  ok &= expect(result.exit_code == 0,
               "shared and stream configuration accept equivalent YAML integer spellings");
  if (result.exit_code != 0) {
    std::cerr << result.stderr_text;
  }
  const auto typed = blazepose_config::TypedConfig::load(config);
  ok &= expect(std::abs(typed.double_or("detector.min_score", 0.0) - 0.5) < 0.001 &&
                   std::abs(typed.double_or("pose.roi_scale", 0.0) - 80.5) < 0.001 &&
                   std::isnan(typed.double_or("test.nan", 0.0)) &&
                   blazepose_config::parse_yaml_scalar("1e3").type ==
                       blazepose_config::YamlScalarType::String,
               "YAML floating-point spellings use the same values and types as Python safe_load");
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_metadata_contract_correlation() {
  using sima_examples::testing::MetadataJsonContract;
  using sima_examples::testing::MetadataJsonMessage;
  const std::vector<MetadataJsonContract> contracts{{"pose-estimation", "poses", 1},
                                                    {"auxiliary-visualization", "poses", 1}};
  std::vector<MetadataJsonMessage> messages{
      {9100, "pose-estimation", "", "camera0:1", 1000, 1},
      {9100, "auxiliary-visualization", "", "camera0:2", 1033, 1}};
  bool ok = expect(!sima_examples::testing::metadata_contracts_complete_for_frame(
                       messages, contracts, 9100, 1033, "camera0:2"),
                   "metadata types from different frames cannot complete a contract");
  messages.push_back({9100, "pose-estimation", "", "camera0:2", 1033, 1});
  ok &= expect(sima_examples::testing::metadata_contracts_complete_for_frame(
                   messages, contracts, 9100, 1033, "camera0:2"),
               "matching timestamp and frame identity complete the metadata contract");
  return ok;
}

bool test_pose_count_limit(const std::string& binary) {
  const fs::path config = write_config(
      "pose_count_limit",
      "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n    codec: h264\n    insight_channel: 0\n");
  std::ofstream(config, std::ios::app) << "pose:\n  max_people_per_frame: 11\n";
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  const bool ok = expect(
      result.exit_code == 1 &&
          result.stderr_text.find("pose.max_people_per_frame must be between 1 and 10") !=
              std::string::npos,
      "pose count is bounded by the metadata transport limit");
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_standalone_sequence_dash_starts_a_stream(const std::string& binary) {
  const std::string streams = "  -\n"
                              "    id: camera0\n"
                              "    url: rtsp://127.0.0.1/src0\n"
                              "    codec: h264\n"
                              "    insight_channel: 0\n"
                              "  - # second stream\n"
                              "    id: camera1\n"
                              "    url: rtsp://127.0.0.1/src1\n"
                              "    insight_channel: 1\n";
  const fs::path config = write_config("standalone_sequence_dash", streams);
  const auto typed = blazepose_config::TypedConfig::load(config);
  bool ok = expect(typed.string_or("output.insight.host", "") == "127.0.0.1" &&
                       typed.string_or("models.pose_path", "") == "pose#v2.tar.gz",
                   "the typed scanner skips standalone sequence dashes like PyYAML");
  const auto result =
      spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
  ok &= expect(result.exit_code == 0 && result.stdout_text.find("streams=2") != std::string::npos,
               "a standalone sequence dash starts a stream mapping like PyYAML");
  if (result.exit_code != 0) {
    std::cerr << result.stderr_text;
  }
  remove_dir(config.parent_path().string());
  return ok;
}

bool test_roi_scale_must_be_finite_and_positive(const std::string& binary) {
  bool ok = true;
  for (const std::string value : {".nan", ".inf", "-.inf", "0", "-1.5"}) {
    const fs::path config =
        write_config("roi_scale", "  - id: camera0\n    url: rtsp://127.0.0.1/src0\n"
                                  "    codec: h264\n    insight_channel: 0\n");
    std::ofstream(config, std::ios::app) << "pose:\n  roi_scale: " << value << "\n";
    const auto result =
        spawn_and_wait(binary, {"--config", config.string(), "--validate-config-only"}, 20000);
    ok &= expect(result.exit_code == 1 &&
                     result.stderr_text.find("pose.roi_scale must be finite and > 0") !=
                         std::string::npos,
                 "pose.roi_scale " + value + " is rejected before it reaches ROI rounding");
    remove_dir(config.parent_path().string());
  }
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  bool ok = test_math_contract();
  ok &= test_pose_aggregate_claims_are_exclusive();
  ok &= test_cli(argv[1]);
  ok &= test_stream_limit(argv[1]);
  ok &= test_duplicate_stream_identity(argv[1]);
  ok &= test_insight_ports_do_not_overlap(argv[1]);
  ok &= test_stream_strings_reject_yaml_non_strings(argv[1]);
  ok &= test_scalar_config_preserves_yaml_types(argv[1]);
  ok &= test_typed_settings_reject_explicit_null(argv[1]);
  ok &= test_stream_mapping_order_and_explicit_caps(argv[1]);
  ok &= test_yaml_scalar_parity(argv[1]);
  ok &= test_metadata_contract_correlation();
  ok &= test_pose_count_limit(argv[1]);
  ok &= test_roi_scale_must_be_finite_and_positive(argv[1]);
  ok &= test_standalone_sequence_dash_starts_a_stream(argv[1]);
  return ok ? 0 : 1;
}
