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

bool near(float actual, float expected) {
  return std::abs(actual - expected) < 0.001F;
}

const std::string kModels = "models:\n"
                            "  detector_path: detector#v2.tar.gz\n"
                            "  pose_path: pose#v2.tar.gz # trailing comment\n";

std::string stream_entry(int index, const std::string& codec = "h264",
                         const std::string& extra = "") {
  const std::string number = std::to_string(index);
  return "  - id: camera" + number + "\n    url: rtsp://127.0.0.1/src" + number +
         "\n    codec: " + codec + "\n    insight_channel: " + number + "\n" + extra;
}

// `output` continues the output mapping; `sections` adds top-level sections.
std::string config(const std::string& streams = stream_entry(0), const std::string& output = "",
                   const std::string& sections = "") {
  return kModels + "streams:\n" + streams + "output:\n  insight:\n    host: 127.0.0.1\n" + output +
         sections;
}

std::string replaced(std::string text, const std::string& from, const std::string& to) {
  return text.replace(text.find(from), from.size(), to);
}

fs::path write_config(const std::string& name, const std::string& yaml) {
  const fs::path path =
      fs::path(create_test_scratch_dir("multi-stream-blazepose3d", name)) / "config.yaml";
  std::ofstream(path) << yaml;
  return path;
}

const std::string kStandaloneDash = config("  -\n"
                                           "    id: camera0\n"
                                           "    url: rtsp://127.0.0.1/src0\n"
                                           "    codec: h264\n"
                                           "    insight_channel: 0\n"
                                           "  - # second stream\n"
                                           "    id: camera1\n"
                                           "    url: rtsp://127.0.0.1/src1\n"
                                           "    insight_channel: 1\n");

const std::string kYamlSpellings = config("  - id: camera0\n"
                                          "    url: rtsp://host/cam's # camera label\n"
                                          "    codec: h264\n"
                                          "    insight_channel: 0b10\n",
                                          "",
                                          "runtime:\n  frames: 010\n"
                                          "detector:\n  min_score: 0.5_0\n"
                                          "pose:\n  job_timeout_ms: 1_000\n"
                                          "  roi_scale: 1:20.5\n"
                                          "test:\n  nan: .NaN\n");

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
  ok &= expect(skipped.size() == 1 && !skipped[0].has_value() && publications.next_sequence() == 4,
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
  ok &= expect(near(pose.presence, 0.93F) && near(blazepose_app::sigmoid(0.0F), 0.5F) &&
                   blazepose_app::sigmoid(-0.01F) < 0.5F,
               "global pose presence is retained and its logit activates before thresholding");
  ok &= expect(near(pose.keypoints[0].x, 18.0F) && near(pose.keypoints[0].y, 44.0F) &&
                   near(pose.keypoints[0].confidence, blazepose_app::sigmoid(-2.0F)),
               "landmarks map through ROI affine metadata with min(visibility, presence)");
  ok &= expect(near(pose.world_keypoints[0].x, 0.1F) && near(pose.world_keypoints[0].y, -0.2F) &&
                   near(pose.world_keypoints[0].z, 0.3F),
               "world landmarks retain the model's 3D coordinates");
  const auto data = blazepose_app::poses_data_json({pose}, "camera0");
  const auto& published = data["poses"][0];
  ok &= expect(data.at("stream_id") == "camera0" && published["id"] == "pose_3" &&
                   near(published["presence"].get<float>(), 0.93F) &&
                   published["keypoints"].size() == 33 &&
                   published["keypoints"][0]["name"] == "nose" &&
                   published["world_keypoints"].size() == 33 &&
                   published["world_keypoints"][0]["name"] == "nose" &&
                   near(published["world_keypoints"][0]["z"].get<float>(), 0.3F),
               "2D metadata carries the stream, rank id, presence and 33 named image and world "
               "keypoints");
  const auto auxiliary = blazepose_app::world_pose_auxiliary_data_json({pose}, "camera0");
  ok &= expect(auxiliary.at("stream_id") == "camera0" && auxiliary["schema_version"] == 1 &&
                   auxiliary["id"] == "world-pose" && auxiliary["renderer"] == "blazepose-3d" &&
                   auxiliary["payload"]["poses"] ==
                       nlohmann::json::array({{{"id", "pose_3"},
                                               {"presence", published["presence"]},
                                               {"keypoints", published["world_keypoints"]}}}),
               "3D metadata uses the generic envelope with the 2D world keypoints and presence");
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
  return ok;
}

bool test_pose_smoother() {
  bool ok = true;
  blazepose_app::Pose first_pose;
  first_pose.presence = 0.9F;
  first_pose.box = {0.0F, 0.0F, 100.0F, 100.0F, 0.9F, 0};
  first_pose.keypoints[0] = {50.0F, 50.0F, 0.2F};
  first_pose.world_keypoints[0] = {0.0F, 0.0F, 0.0F, 0.2F};
  blazepose_app::PoseSmoother smoother;
  smoother.filter({first_pose}, 1'000'000'000);
  blazepose_app::Pose second_pose = first_pose;
  second_pose.roi_index = 1;
  second_pose.keypoints[0].x = 54.0F;
  second_pose.keypoints[0].confidence = 0.4F;
  second_pose.world_keypoints[0].x = 0.04F;
  second_pose.world_keypoints[0].confidence = 0.4F;
  const auto smoothed = smoother.filter({second_pose}, 1'040'000'000).front();
  const float image_fraction = (smoothed.keypoints[0].x - 50.0F) / 4.0F;
  const float world_fraction = smoothed.world_keypoints[0].x / 0.04F;
  ok &= expect(image_fraction > 0.45F && image_fraction < 0.90F &&
                   near(image_fraction, world_fraction) && smoothed.roi_index == 1,
               "temporal filter applies one adaptive weight to 2D and world landmarks and keeps "
               "the current detector rank as the pose id");
  ok &= expect(near(smoothed.keypoints[0].confidence, 0.24F) &&
                   near(smoothed.world_keypoints[0].confidence, 0.24F),
               "temporal filter damps confidence crossings consistently");

  blazepose_app::Pose fast_pose = second_pose;
  fast_pose.keypoints[0].x = 154.0F;
  const auto fast = smoother.filter({fast_pose}, 1'080'000'000).front();
  ok &= expect(fast.keypoints[0].x > 140.0F,
               "temporal filter follows deliberate fast motion without buffering frames");
  blazepose_app::Pose reset_pose = first_pose;
  reset_pose.keypoints[0].x = 30.0F;
  const auto reset = smoother.filter({reset_pose}, 1'400'000'000).front();
  ok &= expect(near(reset.keypoints[0].x, 30.0F),
               "temporal filter resets after a discontinuity instead of dragging stale state");

  const auto moved_fraction = [&](int64_t elapsed_ns) {
    blazepose_app::PoseSmoother elapsed;
    elapsed.filter({first_pose}, 1'000'000'000);
    blazepose_app::Pose moved = first_pose;
    moved.keypoints[0].x = 52.0F;
    return (elapsed.filter({moved}, 1'000'000'000 + elapsed_ns).front().keypoints[0].x - 50.0F) /
           2.0F;
  };
  const float three_frames = moved_fraction(120'000'000);
  ok &= expect(moved_fraction(40'000'000) < three_frames && three_frames < 1.0F,
               "temporal filter weights motion by the elapsed PTS");

  blazepose_app::PoseSmoother continuity;
  continuity.filter({first_pose}, 2'000'000'000);
  const auto first_gap = continuity.filter({}, 2'040'000'000);
  const auto second_gap = continuity.filter({}, 2'080'000'000);
  const auto expired = continuity.filter({}, 2'120'000'000);
  ok &= expect(first_gap.size() == 1 && second_gap.size() == 1 && expired.empty() &&
                   first_gap[0].keypoints[0].confidence < first_pose.keypoints[0].confidence &&
                   second_gap[0].keypoints[0].confidence < first_gap[0].keypoints[0].confidence,
               "temporal filter bridges two missing results with decaying confidence");

  blazepose_app::PoseSmoother no_pts;
  no_pts.filter({first_pose}, -1);
  no_pts.filter({}, -1);
  no_pts.filter({}, -1);
  const bool no_pts_expired = no_pts.filter({}, -1).empty();
  blazepose_app::Pose later_pose = first_pose;
  later_pose.keypoints[0].x = 54.0F;
  const auto later = no_pts.filter({later_pose}, -1).front();
  ok &= expect(no_pts_expired && near(later.keypoints[0].x, 54.0F),
               "temporal filter drops a stale subject once coasting ends without PTS");
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

bool test_failed_metadata_pair_does_not_count_toward_the_frame_limit() {
  int metadata_frames = 0;
  std::uint64_t send_failures = 0;
  std::vector<std::string> sent;
  bool fail_world = true;
  const auto send = [&](const char* type) {
    sent.emplace_back(type);
    return !(fail_world && std::string(type) == "auxiliary-visualization");
  };
  const bool failed_pair = blazepose_app::send_metadata_pair(send, metadata_frames, send_failures);
  bool ok =
      expect(!failed_pair && metadata_frames == 0 && send_failures == 1 &&
                 sent == std::vector<std::string>{"pose-estimation", "auxiliary-visualization"},
             "a pair with a failed send is attempted in full but not counted");
  fail_world = false;
  ok &= expect(blazepose_app::send_metadata_pair(send, metadata_frames, send_failures) &&
                   metadata_frames == 1 && send_failures == 1,
               "a fully queued pair counts toward runtime.frames");
  // Both frames completed, so none is outstanding; only one counts.
  ok &= expect(blazepose_app::stream_can_admit_frame(metadata_frames, 0, 2) &&
                   !blazepose_app::stream_is_drained(false, 0, metadata_frames, 2) &&
                   blazepose_app::stream_is_drained(true, 0, metadata_frames, 2),
               "failed pairs keep admitting frames and still drain when the source closes");
  return ok;
}

bool test_non_finite_landmarks_discard_only_that_pose() {
  const blazepose_app::Box box{0.0F, 0.0F, 100.0F, 100.0F, 0.9F, 0};
  const blazepose_app::Affine affine{1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  const std::vector<float> screen(195, 0.0F);
  const std::vector<float> world(117, 0.0F);
  bool ok =
      expect(blazepose_app::decode_finite_pose(screen, world, affine, box, 0.9F, 0).has_value(),
             "finite BlazePose landmarks decode a pose");
  struct Case {
    const char* name;
    bool world;
    std::size_t index;
    float value;
  };
  for (const Case& test :
       {Case{"screen NaN", false, 0, std::nanf("")}, Case{"screen infinity", false, 194, INFINITY},
        Case{"world NaN", true, 0, std::nanf("")}, Case{"world -infinity", true, 116, -INFINITY}}) {
    std::vector<float> bad_screen = screen;
    std::vector<float> bad_world = world;
    (test.world ? bad_world : bad_screen)[test.index] = test.value;
    ok &= expect(
        !blazepose_app::decode_finite_pose(bad_screen, bad_world, affine, box, 0.9F, 0).has_value(),
        std::string("a ") + test.name + " landmark discards only that pose");
  }
  return ok;
}

bool test_yaml_scalars() {
  using blazepose_config::parse_yaml_integer;
  using blazepose_config::parse_yaml_scalar;
  using blazepose_config::YamlScalarType;
  bool ok = true;
  ok &= expect(blazepose_config::strip_yaml_inline_comment(
                   "    url: rtsp://host/cam's # camera label") == "    url: rtsp://host/cam's ",
               "plain-scalar apostrophes do not retain trailing YAML comments");
  ok &= expect(blazepose_config::strip_yaml_inline_comment(
                   "    url: \"rtsp://host/cam #1\" # camera label") ==
                   "    url: \"rtsp://host/cam #1\" ",
               "quoted-scalar hashes remain part of the configured value");
  ok &=
      expect(parse_yaml_integer("010", "octal") == 8 && parse_yaml_integer("0b10", "binary") == 2 &&
                 parse_yaml_integer("1_000", "underscored") == 1000 &&
                 parse_yaml_integer("1:20", "sexagesimal") == 80 &&
                 parse_yaml_scalar("08").type == YamlScalarType::String,
             "YAML integer spellings, and invalid octal strings, match Python safe_load");
  ok &=
      expect(parse_yaml_scalar("models/2026-10-blazepose.tar.gz").type == YamlScalarType::String &&
                 parse_yaml_scalar("2026-10-01T12:34:56.123_456").type == YamlScalarType::String &&
                 parse_yaml_scalar("2026-10-01").type == YamlScalarType::Other &&
                 parse_yaml_scalar("2026-1-1T1:02:03").type == YamlScalarType::Other &&
                 parse_yaml_scalar("2026-10-01T12:34:56Z").type == YamlScalarType::Other,
             "only complete YAML dates and timestamps retain non-string types");
  ok &= expect(parse_yaml_scalar("'rtsp://host/cam''s'").value == "rtsp://host/cam's" &&
                   parse_yaml_scalar("\"rtsp://host/cam\\\"east\"").value ==
                       "rtsp://host/cam\"east" &&
                   parse_yaml_scalar("\"\\u0041\\U0001F680\"").value == "A\xf0\x9f\x9a\x80",
               "quoted YAML strings decode single, double, and Unicode escapes");

  const fs::path spellings = write_config("yaml_spellings", kYamlSpellings);
  const auto typed = blazepose_config::TypedConfig::load(spellings);
  ok &= expect(std::abs(typed.double_or("detector.min_score", 0.0) - 0.5) < 0.001 &&
                   std::abs(typed.double_or("pose.roi_scale", 0.0) - 80.5) < 0.001 &&
                   std::isnan(typed.double_or("test.nan", 0.0)) &&
                   parse_yaml_scalar("1e3").type == YamlScalarType::String,
               "YAML floating-point spellings use the same values and types as Python safe_load");
  remove_dir(spellings.parent_path().string());

  const fs::path dash = write_config("standalone_sequence_dash", kStandaloneDash);
  const auto scanned = blazepose_config::TypedConfig::load(dash);
  ok &= expect(scanned.string_or("output.insight.host", "") == "127.0.0.1" &&
                   scanned.string_or("models.detector_path", "") == "detector#v2.tar.gz" &&
                   scanned.string_or("models.pose_path", "") == "pose#v2.tar.gz",
               "the typed scanner skips standalone sequence dashes and keeps hashes in scalars");
  remove_dir(dash.parent_path().string());
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

// Mirrors REJECTED_CONFIGS and ACCEPTED_CONFIGS in the Python unit test.
bool test_config_validation(const std::string& binary) {
  struct Case {
    std::string name;
    std::string yaml;
    std::string expected; // stderr text of a rejection, or stdout text of an accepted config
    bool accepted = false;
  };
  const std::string flow_stream = "{id: camera0, url: rtsp://127.0.0.1/src0, insight_channel: 0}";
  const std::string flow_error = "flow-style YAML collections are not supported";
  const std::string base = config();
  std::vector<Case> cases = {
      {"five_streams",
       config(stream_entry(0) + stream_entry(1, "hevc") + stream_entry(2) + stream_entry(3) +
              stream_entry(4)),
       "streams must contain between 1 and 4 entries"},
      {"duplicate_id",
       config(stream_entry(0) + replaced(stream_entry(1), "id: camera1", "id: camera0")),
       "stream ids must be unique"},
      {"duplicate_channel",
       config(stream_entry(0) + replaced(stream_entry(1), "channel: 1", "channel: 0")),
       "stream insight channels must be unique"},
      {"port_overlap",
       config(stream_entry(0) + replaced(stream_entry(1), "channel: 1", "channel: 100"),
              "    video_port_base: 9000\n    metadata_port_base: 9100\n"),
       "video and metadata ports must not overlap"},
      {"video_port_range",
       config(stream_entry(60000), "    metadata_port_base: 1\n  video_enabled: true\n"),
       "stream video port must be <= 65535"},
      {"metadata_port_range",
       config(stream_entry(60000), "    video_port_base: 1\n  video_enabled: false\n"),
       "stream metadata port must be <= 65535"},
      {"unknown_stream_setting", config(stream_entry(0, "h264", "    enabled: true\n")),
       "unknown stream setting: enabled"},
      {"partial_caps", config(stream_entry(0, "h264", "    width: 1920\n")),
       "must either all be omitted or all be > 0"},
      {"unknown_codec", config(stream_entry(0, "vp9")),
       "stream codec must be h264/avc or h265/hevc"},
      {"eleven_people", config(stream_entry(0), "", "pose:\n  max_people_per_frame: 11\n"),
       "pose.max_people_per_frame must be between 1 and 10"},
      {"max_detections", config(stream_entry(0), "", "detector:\n  max_detections: 0\n"),
       "detector.max_detections must be > 0"},
      {"max_inflight", config(stream_entry(0), "", "detector:\n  max_inflight_per_stream: 0\n"),
       "detector.max_inflight_per_stream must be -1 or > 0"},
      {"max_pending_jobs", config(stream_entry(0), "", "pose:\n  max_pending_jobs: 0\n"),
       "pose.max_pending_jobs must be > 0"},
      {"temporal_filter", config(stream_entry(0), "", "pose:\n  temporal_filter_enabled: 1\n"),
       "pose.temporal_filter_enabled must be true or false"},
      {"null_bool", config(stream_entry(0), "  video_enabled: null\n"), "must be true or false"},
      {"null_int", config(stream_entry(0), "", "detector:\n  max_detections: null\n"),
       "must be an integer"},
      {"null_double", config(stream_entry(0), "", "pose:\n  roi_scale: null\n"), "must be numeric"},
      {"null_url", replaced(base, "url: rtsp://127.0.0.1/src0", "url: null"), "must be a string"},
      {"numeric_id", replaced(base, "id: camera0", "id: 17"), "must be a string"},
      {"binary_id", replaced(base, "id: camera0", "id: 0b101"), "must be a string"},
      {"sexagesimal_id", replaced(base, "id: camera0", "id: 1:20"), "must be a string"},
      {"date_id", replaced(base, "id: camera0", "id: 2026-10-01"), "must be a string"},
      {"boolean_url", replaced(base, "url: rtsp://127.0.0.1/src0", "url: true"),
       "must be a string"},
      {"numeric_detector", replaced(base, "detector#v2.tar.gz", "123"), "must be a string"},
      {"boolean_pose", replaced(base, "pose#v2.tar.gz # trailing comment", "false"),
       "must be a string"},
      {"numeric_host", replaced(base, "host: 127.0.0.1", "host: 127"), "must be a string"},
      {"flow_streams", kModels + "streams: [" + flow_stream + "]\n", flow_error},
      {"flow_stream_entry", kModels + "streams:\n  - " + flow_stream + "\n", flow_error},
      {"flow_section",
       kModels + "streams:\n" + stream_entry(0) + "output: {insight: {host: 127.0.0.1}}\n",
       flow_error},
      {"standalone_sequence_dash", kStandaloneDash, "streams=2", true},
      {"yaml_spellings", kYamlSpellings, "streams=1", true},
      {"empty_flow_mapping", base + "pose: {}\n", "streams=1", true},
      {"null_codec_trailing_quote",
       replaced(replaced(base, "id: camera0", "id: camera'"), "codec: h264", "codec: null"),
       "streams=1", true},
      {"mapping_order_and_hashes",
       config("  - url: rtsp://127.0.0.1/ordered#view\n    width: 1920\n"
              "    id: camera#1 # trailing comment\n    fps: 30\n    insight_channel: 0\n"
              "    height: 1080\n    codec: h264\n"
              "  - codec: h265\n    insight_channel: 1\n    id: camera#2\n"
              "    url: rtsp://127.0.0.1/second#view\n"),
       "streams=2", true},
      {"video_disabled_skips_video_port_range",
       config(stream_entry(60000), "    metadata_port_base: 1\n  video_enabled: false\n"),
       "streams=1", true},
      {"codec_spellings",
       config(stream_entry(0, "avc") + stream_entry(1, "H.264") + stream_entry(2, "HEVC") +
              stream_entry(3, "h.265")),
       "streams=4", true},
  };
  for (const std::string value : {".nan", ".inf", "-.inf", "0", "-1.5"}) {
    cases.push_back({"roi_scale " + value,
                     config(stream_entry(0), "", "pose:\n  roi_scale: " + value + "\n"),
                     "pose.roi_scale must be finite and > 0"});
  }

  bool ok = true;
  for (const Case& test : cases) {
    const fs::path path = write_config("config_validation", test.yaml);
    const auto result =
        spawn_and_wait(binary, {"--config", path.string(), "--validate-config-only"}, 20000);
    const std::string& output = test.accepted ? result.stdout_text : result.stderr_text;
    const bool passed = result.exit_code == (test.accepted ? 0 : 1) &&
                        output.find(test.expected) != std::string::npos;
    if (!passed) {
      std::cerr << result.stderr_text;
    }
    ok &= expect(passed, test.name + (test.accepted ? " is accepted" : " is rejected") +
                             " consistently with Python");
    remove_dir(path.parent_path().string());
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
  ok &= test_pose_smoother();
  ok &= test_pose_aggregate_claims_are_exclusive();
  ok &= test_non_finite_landmarks_discard_only_that_pose();
  ok &= test_failed_metadata_pair_does_not_count_toward_the_frame_limit();
  ok &= test_yaml_scalars();
  ok &= test_metadata_contract_correlation();
  ok &= test_cli(argv[1]);
  ok &= test_config_validation(argv[1]);
  return ok ? 0 : 1;
}
