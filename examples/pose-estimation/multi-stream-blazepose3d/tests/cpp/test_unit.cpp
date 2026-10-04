#include "examples/pose-estimation/multi-stream-blazepose3d/src/cpp/pose_logic.h"
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_process.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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

std::string stream_entry(int index, const std::string& codec = "h264",
                         const std::string& extra = "", bool with_caps = true) {
  const std::string number = std::to_string(index);
  return "  - id: camera" + number + "\n    url: rtsp://127.0.0.1/src" + number +
         "\n    codec: " + codec + "\n    insight_channel: " + number + "\n" +
         (with_caps ? "    width: 1920\n    height: 1080\n    fps: 30\n" : "") + extra;
}

// `insight` continues the output.insight mapping; `sections` adds top-level sections.
std::string config(const std::string& streams = stream_entry(0), const std::string& insight = "",
                   const std::string& sections = "") {
  return "models:\n  detector_path: detector.tar.gz\n  pose_path: pose.tar.gz\nstreams:\n" +
         streams + "output:\n  insight:\n    host: 127.0.0.1\n" + insight + sections;
}

std::string replaced(std::string text, const std::string& from, const std::string& to) {
  return text.replace(text.find(from), from.size(), to);
}

blazepose_app::Pose pose_at(float x, float world_x, float box_x = 0.0F) {
  blazepose_app::Pose pose;
  pose.presence = 0.9F;
  pose.box = {box_x, 0.0F, box_x + 100.0F, 100.0F, 0.9F, 0};
  pose.keypoints[0] = {x, 50.0F, 0.9F};
  pose.world_keypoints[0] = {world_x, 0.0F, 0.0F, 0.9F};
  return pose;
}

bool test_math_and_metadata_contract() {
  bool ok = true;
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
                   near(pose.keypoints[0].confidence, blazepose_app::sigmoid(-2.0F)) &&
                   near(pose.world_keypoints[0].x, 0.1F) &&
                   near(pose.world_keypoints[0].y, -0.2F) && near(pose.world_keypoints[0].z, 0.3F),
               "landmarks map through ROI affine metadata and world landmarks keep 3D values");

  const auto data = blazepose_app::poses_data_json({pose}, "camera0");
  const auto& published = data["poses"][0];
  ok &= expect(data.at("stream_id") == "camera0" && published["id"] == "pose_3" &&
                   near(published["presence"].get<float>(), 0.93F) &&
                   published["keypoints"].size() == 33 &&
                   published["keypoints"][0]["name"] == "nose" &&
                   published["world_keypoints"].size() == 33 &&
                   published["world_keypoints"][0]["name"] == "nose",
               "2D metadata carries the stream, presence and 33 named image and world keypoints");
  const auto auxiliary = blazepose_app::world_pose_auxiliary_from_overlay(data);
  ok &= expect(auxiliary.at("stream_id") == "camera0" && auxiliary["schema_version"] == 1 &&
                   auxiliary["id"] == "world-pose" && auxiliary["renderer"] == "blazepose-3d" &&
                   auxiliary["payload"]["poses"] ==
                       nlohmann::json::array({{{"id", "pose_3"},
                                               {"presence", published["presence"]},
                                               {"keypoints", published["world_keypoints"]}}}),
               "3D metadata uses the generic envelope with the 2D world keypoints and presence");
  const nlohmann::json point_cloud = {{"points", {{{"x", 0.1}, {"value", 7}}}}};
  const auto generic = blazepose_app::auxiliary_visualization_data_json(
      "depth-cloud", "point-cloud-3d", point_cloud);
  ok &= expect(generic["schema_version"] == 1 && generic["id"] == "depth-cloud" &&
                   generic["renderer"] == "point-cloud-3d" && generic["payload"] == point_cloud &&
                   !generic.contains("title"),
               "generic auxiliary envelope preserves an arbitrary renderer payload");
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

bool test_non_finite_detector_scores_are_discarded() {
  const blazepose_app::Box person{0.0F, 0.0F, 100.0F, 100.0F, 0.9F, 0};
  blazepose_app::Box invalid = person;
  invalid.score = std::nanf("");
  bool ok = expect(blazepose_app::is_finite_person_box(person) &&
                       !blazepose_app::is_finite_person_box(invalid),
                   "a non-finite detector score discards that person box");
  invalid.score = INFINITY;
  ok &= expect(!blazepose_app::is_finite_person_box(invalid),
               "an infinite detector score discards that person box");
  invalid = person;
  invalid.class_id = 1;
  ok &= expect(!blazepose_app::is_finite_person_box(invalid),
               "a finite non-person detector box remains excluded");
  return ok;
}

bool test_pose_smoother() {
  blazepose_app::PoseSmoother smoother;
  blazepose_app::PoseSmoother other_stream;
  smoother.filter({pose_at(50.0F, 0.0F)});
  other_stream.filter({pose_at(90.0F, 1.0F)});
  const auto matched = smoother.filter({pose_at(54.0F, 0.04F)}).front();
  const auto unmatched = smoother.filter({pose_at(400.0F, 1.0F, 300.0F)}).front();
  const auto other = other_stream.filter({pose_at(80.0F, 1.0F)}).front();
  return expect(near(matched.keypoints[0].x, 52.0F) && near(matched.world_keypoints[0].x, 0.02F) &&
                    near(unmatched.keypoints[0].x, 400.0F) && near(other.keypoints[0].x, 85.0F),
                "each stream's smoother blends matched 2D and world landmarks and passes "
                "unmatched poses through");
}

bool test_latest_work_and_metadata_pairs() {
  std::optional<int> mailbox;
  bool ok = expect(!blazepose_app::keep_latest(mailbox, 1) &&
                       blazepose_app::keep_latest(mailbox, 2) && mailbox == 2,
                   "a stream mailbox keeps only the latest work");
  std::uint64_t last_published_frame_id = 0;
  ok &= expect(blazepose_app::claim_newer_frame(2, last_published_frame_id) &&
                   !blazepose_app::claim_newer_frame(1, last_published_frame_id) &&
                   blazepose_app::claim_newer_frame(3, last_published_frame_id) &&
                   last_published_frame_id == 3,
               "a stream never publishes a completed frame behind a newer frame");
  std::vector<std::string> sent;
  const auto send_failing_world = [&](const char* type) {
    sent.emplace_back(type);
    return std::string(type) != "auxiliary-visualization";
  };
  ok &= expect(!blazepose_app::send_metadata_pair(send_failing_world) &&
                   sent == std::vector<std::string>{"pose-estimation", "auxiliary-visualization"} &&
                   blazepose_app::send_metadata_pair([](const char*) { return true; }),
               "a metadata pair attempts both sends and counts only when both succeed");
  return ok;
}

// Mirrors test_accepted_input_without_output_stops_the_app in the Python unit test.
bool test_inference_stall_timeout() {
  using blazepose_app::inference_stalled;
  using blazepose_app::kInferenceStallTimeout;
  std::optional<std::chrono::steady_clock::time_point> waiting_since;
  const auto start = std::chrono::steady_clock::now();
  const auto late = start + kInferenceStallTimeout + std::chrono::milliseconds(1);
  bool ok = expect(!inference_stalled(waiting_since, false, late) &&
                       !inference_stalled(waiting_since, true, start) &&
                       !inference_stalled(waiting_since, true, start + kInferenceStallTimeout) &&
                       inference_stalled(waiting_since, true, late),
                   "accepted input without output stalls only after the timeout");
  ok &= expect(!inference_stalled(waiting_since, false, late) &&
                   !inference_stalled(waiting_since, true, late),
               "the stall timer restarts once no input is pending");
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

// Mirrors REJECTED_CONFIGS in the Python unit test. Invalid configs fail before
// any model or stream is opened.
bool test_config_validation(const std::string& binary) {
  struct Case {
    std::string name;
    std::string yaml;
    std::string error;
  };
  std::vector<Case> cases = {
      {"no_host", replaced(config(), "host: 127.0.0.1", "host: ''"), "host must be set"},
      {"numeric_detector_path",
       replaced(config(), "detector_path: detector.tar.gz", "detector_path: 123"),
       "models.detector_path must be a string"},
      {"boolean_pose_path", replaced(config(), "pose_path: pose.tar.gz", "pose_path: true"),
       "models.pose_path must be a string"},
      {"timestamp_host", replaced(config(), "host: 127.0.0.1", "host: 2026-10-04"),
       "output.insight.host must be a string"},
      {"collection_host", replaced(config(), "host: 127.0.0.1", "host: []"),
       "output.insight.host must be a string"},
      {"five_streams",
       config(stream_entry(0) + stream_entry(1, "h265") + stream_entry(2) + stream_entry(3) +
              stream_entry(4)),
       "streams must contain between 1 and 4 entries"},
      {"duplicate_id",
       config(stream_entry(0) + replaced(stream_entry(1), "id: camera1", "id: camera0")),
       "stream ids must be unique"},
      {"quoted_stream_key", config(replaced(stream_entry(0), "- id: camera0", "- \"id\": camera0")),
       "quoted YAML mapping keys are not supported"},
      {"numeric_id", config(replaced(stream_entry(0), "id: camera0", "id: 123")),
       "stream id must be a string"},
      {"timestamp_id", config(replaced(stream_entry(0), "id: camera0", "id: 2026-10-04")),
       "stream id must be a string"},
      {"block_scalar_id", config(replaced(stream_entry(0), "id: camera0", "id: >-\n      camera0")),
       "YAML block scalar values are not supported"},
      {"boolean_url", config(replaced(stream_entry(0), "url: rtsp://127.0.0.1/src0", "url: true")),
       "stream url must be a string"},
      {"null_id", config(replaced(stream_entry(0), "id: camera0", "id: null")),
       "stream id must be set"},
      {"null_url", config(replaced(stream_entry(0), "url: rtsp://127.0.0.1/src0", "url: ~")),
       "stream url must be set"},
      {"standalone_dash",
       config(replaced(stream_entry(0), "- id: camera0", "-\n    id: camera0") +
              replaced(stream_entry(1), "id: camera1", "id: camera0")),
       "stream ids must be unique"},
      {"plain_apostrophe_comment",
       config(replaced(stream_entry(0), "id: camera0", "id: camera'one # comment") +
              replaced(stream_entry(1), "id: camera1", "id: \"camera'one\"")),
       "stream ids must be unique"},
      {"single_quote_escape",
       config(replaced(stream_entry(0), "id: camera0", "id: 'camera''s'") +
              replaced(stream_entry(1), "id: camera1", "id: \"camera's\"")),
       "stream ids must be unique"},
      {"double_quote_escape",
       config(replaced(stream_entry(0), "id: camera0", "id: \"camera\\\"one\"") +
              replaced(stream_entry(1), "id: camera1", "id: 'camera\"one'")),
       "stream ids must be unique"},
      {"duplicate_channel",
       config(stream_entry(0) + replaced(stream_entry(1), "channel: 1", "channel: 0")),
       "stream insight channels must be unique"},
      {"port_overlap",
       config(stream_entry(0) + replaced(stream_entry(1), "channel: 1", "channel: 100"),
              "    video_port_base: 9000\n"),
       "video and metadata ports must not overlap"},
      {"quoted_hash",
       config(replaced(stream_entry(0), "id: camera0", "id: \"camera #0\"") +
                  replaced(replaced(stream_entry(1), "id: camera1", "id: '\"camera'"), "channel: 1",
                           "channel: 100"),
              "    video_port_base: 9000\n"),
       "video and metadata ports must not overlap"},
      {"video_port_range", config(stream_entry(60000), "    metadata_port_base: 1\n"),
       "stream video port must be <= 65535"},
      {"metadata_port_range", config(stream_entry(60000), "    video_port_base: 1\n"),
       "stream metadata port must be <= 65535"},
      {"missing_caps", config(stream_entry(0, "h264", "", false)),
       "width, height, and fps must all be > 0"},
      {"quoted_width", config(replaced(stream_entry(0), "width: 1920", "width: \"1920\"")),
       "stream width must be an integer"},
      {"unknown_codec", config(stream_entry(0, "hevc")), "stream codec must be h264 or h265"},
      {"null_codec", config(stream_entry(0, "null")), "detector model not found: detector.tar.gz"},
      {"tilde_codec", config(stream_entry(0, "~")), "detector model not found: detector.tar.gz"},
      {"escaped_model_path",
       replaced(config(), "detector_path: detector.tar.gz",
                "detector_path: \"detector\\x2Etar.gz\""),
       "detector model not found: detector.tar.gz"},
      {"flow_streams", replaced(config(), "streams:", "streams: [{id: camera0}]"),
       "flow-style YAML collections are not supported"},
      {"flow_stream_entry", replaced(config(), "- id: camera0", "- {id: camera0}"),
       "flow-style YAML collections are not supported"},
      {"flow_section", config(stream_entry(0), "", "runtime: {frames: 1}\n"),
       "flow-style YAML collections are not supported"},
      {"empty_flow_section", config(stream_entry(0), "", "pose: {}\n"),
       "detector model not found: detector.tar.gz"},
      {"legacy_booleans",
       config(stream_entry(0), "", "input:\n  tcp: yes\npose:\n  temporal_filter_enabled: OFF\n"),
       "detector model not found: detector.tar.gz"},
      {"quoted_block_marker", config(replaced(stream_entry(0), "id: camera0", "id: \">-\"")),
       "detector model not found: detector.tar.gz"},
      {"eleven_people", config(stream_entry(0), "", "pose:\n  max_people_per_frame: 11\n"),
       "pose.max_people_per_frame must be between 1 and 10"},
      {"null_frames", config(stream_entry(0), "", "runtime:\n  frames: null\n"),
       "runtime.frames must be an integer"},
      {"quoted_frames", config(stream_entry(0), "", "runtime:\n  frames: \"1\"\n"),
       "runtime.frames must be an integer"},
      {"quoted_tcp", config(stream_entry(0), "", "input:\n  tcp: \"true\"\n"),
       "input.tcp must be true or false"},
      {"quoted_min_score", config(stream_entry(0), "", "detector:\n  min_score: \"0.5\"\n"),
       "detector.min_score must be numeric"},
      {"empty_tcp", config(stream_entry(0), "", "input:\n  tcp:\n"),
       "input.tcp must be true or false"},
      {"tilde_min_score", config(stream_entry(0), "", "detector:\n  min_score: ~\n"),
       "detector.min_score must be numeric"},
      {"uppercase_null_filter",
       config(stream_entry(0), "", "pose:\n  temporal_filter_enabled: NULL\n"),
       "pose.temporal_filter_enabled must be true or false"},
      {"quoted_null_frames", config(stream_entry(0), "", "runtime:\n  frames: \" null \"\n"),
       "runtime.frames must be an integer"},
      {"hash_suffix_null_frames", config(stream_entry(0), "", "runtime:\n  frames: null#comment\n"),
       "runtime.frames must be an integer"},
  };
  for (const std::string value : {"nan", "inf", "0"}) {
    cases.push_back({"roi_scale " + value,
                     config(stream_entry(0), "", "pose:\n  roi_scale: " + value + "\n"),
                     "pose.roi_scale must be finite and > 0"});
  }

  bool ok = true;
  for (const Case& test : cases) {
    const fs::path path =
        fs::path(create_test_scratch_dir("multi-stream-blazepose3d", "config_validation")) /
        "config.yaml";
    std::ofstream(path) << test.yaml;
    const auto result = spawn_and_wait(binary, {"--config", path.string()}, 20000);
    const bool passed =
        result.exit_code == 1 && result.stderr_text.find(test.error) != std::string::npos;
    if (!passed) {
      std::cerr << result.stderr_text;
    }
    ok &= expect(passed, test.name + " is rejected as in Python");
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
  bool ok = test_math_and_metadata_contract();
  ok &= test_non_finite_landmarks_discard_only_that_pose();
  ok &= test_non_finite_detector_scores_are_discarded();
  ok &= test_pose_smoother();
  ok &= test_latest_work_and_metadata_pairs();
  ok &= test_inference_stall_timeout();
  ok &= test_metadata_contract_correlation();
  ok &= test_cli(argv[1]);
  ok &= test_config_validation(argv[1]);
  return ok ? 0 : 1;
}
