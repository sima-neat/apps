// Unit tests for the multi-stream multi-model example: the checks the application leaves to its
// config, the Insight metadata of each task, and the pending-result deadline.
// tests/python/test_unit.py runs the same tests in Python. Python also runs its pull loop against a
// fake Run and checks the MODEL_IDS table. A C++ Run cannot be faked, and kModelIds is private to
// main.cpp, so here the deadline is tested on PendingResults.observe(), the step the loop takes
// for each stream in each round.

#include "examples/multi-model/multi-stream-multi-model/src/cpp/utils/insight_metadata.h"
#include "examples/multi-model/multi-stream-multi-model/src/cpp/utils/pending_results.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using insight_metadata::Entry;
using pending_results::PendingResults;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-multi-model";
const std::vector<std::string> kLabels = {"person", "bicycle", "car"};

// The error observe() throws for one round, or "" when it throws none.
std::string error_of(PendingResults& pending, bool frame, bool result, double now_s) {
  try {
    pending.observe(frame, result, now_s);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

// The last line the example prints to stderr for the shipped config with the list `key` set to
// `values`. The frame size and rate are set, so no stream is probed and no source is needed.
std::string run_error(const std::string& binary, const std::string& test_name,
                      const std::string& key, const std::vector<std::string>& values) {
  const std::string dir = create_test_scratch_dir(kExampleName, test_name);
  const fs::path config_path = fs::path(dir) / "config.yaml";
  write_e2e_config(kExampleName, config_path,
                   {{"input.width", "1280"}, {"input.height", "720"}, {"input.fps", "30"}},
                   {{key, values}});
  const ProcessResult result = spawn_and_wait(binary, {"--config", config_path.string()}, 60000);
  remove_dir(dir);
  std::string text = result.stderr_text;
  while (!text.empty() && text.back() == '\n') {
    text.pop_back();
  }
  const auto newline = text.rfind('\n');
  return result.exit_code == 1 ? text.substr(newline == std::string::npos ? 0 : newline + 1)
                               : "exit " + std::to_string(result.exit_code) + ": " + text;
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_is_usable() {
  const fs::path common_dir = "examples/multi-model/multi-stream-multi-model/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const auto sources = config.string_list("input.source");
  bool ok = expect_true(!sources.empty() && sources.size() <= 4U,
                        "the shipped config lists one to four streams");
  const auto model_paths = config.string_list("app.model_paths");
  const auto model_ids = config.string_list("app.model_ids");
  ok &= expect_true(model_paths.size() == sources.size() && model_ids.size() == sources.size(),
                    "one model per stream");
  ok &= expect_true(model_ids == std::vector<std::string>{"yolov8", "yolo26-seg", "yolo26-pose",
                                                          "yolo26"},
                    "app.model_ids names the four supported packages");
  const std::vector<std::string> files = {
      "yolo11s_w8a8_mpk.tar.gz", "yolo26s_seg_w8a16_mpk.tar.gz", "yolo26m_pose_w8a8_mpk.tar.gz",
      "yolo26m_w8a8_mpk.tar.gz"};
  for (std::size_t i = 0; i < model_paths.size() && i < files.size(); ++i) {
    const fs::path model_path = common_dir / model_paths[i];
    ok &= expect_true(model_path.filename() == files[i], model_paths[i] + " names " + files[i]);
    // The packaged path reaches prebuilt-apps/models from the config file.
    const std::string normal = fs::absolute(model_path).lexically_normal().string();
    const std::string models = fs::absolute("models").lexically_normal().string() + "/";
    ok &= expect_true(normal.rfind(models, 0) == 0,
                      model_paths[i] + " reaches models/ from the config file");
  }
  const std::string codec = config.string("input.codec");
  ok &= expect_true(codec == "h264" || codec == "h265", "the shipped codec is supported");
  std::ifstream labels_file(config.string("model.labels"));
  int labels = 0;
  for (std::string line; std::getline(labels_file, line);) {
    labels += line.empty() ? 0 : 1;
  }
  ok &= expect_true(labels == 80, "the COCO labels the detection and segmentation models score");
  for (const std::string key :
       {"input.width", "input.height", "input.fps", "input.rtsp.latency_ms", "inference.frames",
        "output.insight.video_port_base", "output.insight.metadata_port_base"}) {
    ok &= expect_true(config.integer(key) >= 0, key + " is not negative");
  }
  ok &= expect_true(config.integer("inference.max_detections") > 0,
                    "inference.max_detections is positive");
  for (const std::string key : {"inference.min_score", "inference.nms_iou"}) {
    ok &= expect_true(config.number(key) >= 0.0 && config.number(key) <= 1.0,
                      key + " is from 0 to 1");
  }
  ok &= expect_true(!config.string("output.insight.host").empty(), "output.insight.host is set");
  ok &= expect_true(config.boolean("input.rtsp.tcp"), "the shipped config receives over TCP");
  return ok;
}

bool test_objects_clamp_boxes_to_the_frame() {
  const std::vector<simaai::neat::Box> boxes = {{10.7f, 20.2f, 110.9f, 220.5f, 0.91234f, 2},
                                                {1200.0f, 700.0f, 1400.0f, 800.0f, 0.5f, 7}};
  const auto entries = insight_metadata::objects(boxes, kLabels, 1280, 720);
  const std::string expected =
      R"([{"id":"obj_1","label":"car","confidence":0.912,"bbox":[10.0,20.0,100.0,200.0]},)"
      R"({"id":"obj_2","label":"unknown","confidence":0.5,"bbox":[1200.0,700.0,80.0,20.0]}])";
  return expect_true(Entry(entries).dump() == expected,
                     "objects clamp boxes to the frame: " + Entry(entries).dump());
}

bool test_mask_rect_maps_the_frame_through_the_letterbox() {
  // A 1920x1080 frame fills the 640x640 input at 1/3 scale, with 140 rows of padding above.
  const cv::Rect whole = insight_metadata::mask_rect({0, 0, 1920, 1080}, 1920, 1080, 160, 160);
  const cv::Rect corner =
      insight_metadata::mask_rect({960, 540, 960, 540}, 1920, 1080, 160, 160);
  bool ok = expect_true(whole == cv::Rect(0, 35, 160, 90), "the whole frame maps to rows 35-124");
  ok &= expect_true(corner == cv::Rect(80, 80, 80, 45),
                    "the bottom-right quarter maps to cells 80-159 by 80-124");
  return ok;
}

bool test_segment_outline_follows_the_mask() {
  // In a 640x640 frame one mask cell is 4x4 pixels. The object covers cells 20-59 by 40-79,
  // pixels 80-239 by 160-319.
  std::vector<std::uint8_t> masks(2 * 160 * 160, 0);
  for (int y = 40; y < 80; ++y) {
    std::fill_n(masks.begin() + y * 160 + 20, 40, 255);
  }
  const std::vector<float> boxes = {64.0f, 144.0f, 256.0f, 336.0f, 0.80004f, 1.0f,
                                    0.0f,  0.0f,   100.0f, 100.0f, 0.9f,     0.0f};
  auto segments = insight_metadata::segments(boxes, masks, 160, 160, kLabels, 640, 640);
  // The second detection's mask is empty, so it has no outline and is left out.
  if (!expect_true(segments.size() == 1U, "one segment has an outline")) {
    return false;
  }
  const Entry outline = segments[0]["mask"];
  segments[0].erase("mask");
  const std::string expected = R"({"id":"seg_1","label":"bicycle","confidence":0.8,)"
                               R"("bbox":[64,144,192,192],"mask_format":"polygon"})";
  bool ok = expect_true(segments[0].dump() == expected,
                        "the segment names its class, score, and box: " + segments[0].dump());
  // The resize rounds the square's corners, so the outline has four corners, each within a pixel
  // of the square's, and spans the square exactly.
  std::vector<int> xs;
  std::vector<int> ys;
  for (const auto& point : outline) {
    xs.push_back(point[0].get<int>());
    ys.push_back(point[1].get<int>());
  }
  ok &= expect_true(outline.size() == 4U, "the outline has four corners: " + outline.dump());
  ok &= expect_true(!xs.empty() && *std::min_element(xs.begin(), xs.end()) == 80 &&
                        *std::max_element(xs.begin(), xs.end()) == 239 &&
                        *std::min_element(ys.begin(), ys.end()) == 160 &&
                        *std::max_element(ys.begin(), ys.end()) == 319,
                    "the outline spans pixels 80-239 by 160-319: " + outline.dump());
  return ok;
}

bool test_poses_round_to_whole_pixels_and_name_the_keypoints() {
  const std::vector<float> boxes = {10.5f, 20.4f, 110.5f, 220.6f, 0.87654f, 0.0f};
  std::vector<float> keypoints;
  for (int k = 0; k < 17; ++k) {
    keypoints.insert(keypoints.end(), {k * 10 + 0.5f, k * 5 + 0.4f, 0.5f});
  }
  const auto poses = insight_metadata::poses(boxes, keypoints);
  if (!expect_true(poses.size() == 1U, "one pose")) {
    return false;
  }
  const Entry& pose = poses[0];
  bool ok = expect_true(pose["id"] == "pose_1" && pose["label"] == "person" &&
                            pose["confidence"] == 0.877,
                        "the pose is a person with a three-decimal score: " + pose.dump());
  ok &= expect_true(pose["bbox"].dump() == "[11,20,100,200]",
                    "the box rounds to whole pixels: " + pose["bbox"].dump());
  ok &= expect_true(pose["keypoints"][0].dump() ==
                        R"({"name":"nose","x":1,"y":0,"confidence":0.5})",
                    "the first keypoint is the nose: " + pose["keypoints"][0].dump());
  ok &= expect_true(pose["keypoints"][16].dump() ==
                        R"({"name":"right_ankle","x":161,"y":80,"confidence":0.5})",
                    "the last keypoint is the right ankle: " + pose["keypoints"][16].dump());
  bool names = pose["keypoints"].size() == insight_metadata::kCocoKeypointNames.size();
  for (std::size_t k = 0; names && k < insight_metadata::kCocoKeypointNames.size(); ++k) {
    names = pose["keypoints"][k]["name"] == insight_metadata::kCocoKeypointNames[k];
  }
  ok &= expect_true(names, "the keypoints carry the 17 COCO names in order");
  return ok;
}

std::string decode_error(bool pose) {
  const std::vector<float> boxes(12, 0.0f);
  try {
    if (pose) {
      insight_metadata::poses(boxes, std::vector<float>(17 * 3, 0.0f));
    } else {
      insight_metadata::segments(boxes, std::vector<std::uint8_t>(160 * 160, 0), 160, 160,
                                 kLabels, 640, 640);
    }
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

bool test_decodes_with_fewer_rows_than_boxes_fail() {
  bool ok = expect_true(decode_error(false) == "segmentation decode returned 2 boxes but 1 masks",
                        "a short mask list fails: " + decode_error(false));
  ok &= expect_true(decode_error(true) == "pose decode returned 2 boxes but 1 keypoint sets",
                    "a short keypoint list fails: " + decode_error(true));
  return ok;
}

bool test_data_json_leaves_out_the_lowest_confidence_entries() {
  // Two entries fit the 32768-byte budget, three do not.
  const std::string pad(15000, 'x');
  const std::vector<Entry> entries = {{{"id", "a"}, {"confidence", 0.5}, {"pad", pad}},
                                      {{"id", "b"}, {"confidence", 0.9}, {"pad", pad}},
                                      {{"id", "c"}, {"confidence", 0.7}, {"pad", pad}}};
  const auto [data, left_out] = insight_metadata::data_json("objects", entries);
  std::vector<std::string> ids;
  for (const auto& entry : Entry::parse(data)["objects"]) {
    ids.push_back(entry["id"].get<std::string>());
  }
  bool ok = expect_true(left_out == 1, "one entry is left out");
  ok &= expect_true(ids == std::vector<std::string>{"b", "c"},
                    "the two highest-confidence entries are kept, best first");
  const auto empty = insight_metadata::data_json("poses", {});
  ok &= expect_true(empty.first == R"({"poses":[]})" && empty.second == 0,
                    "no entries make an empty array");
  return ok;
}

bool test_late_result_names_the_stream_and_model() {
  PendingResults pending(2, "models/m.tar.gz", 10.0, 30.0);
  pending.observe(true, true, 100.0);
  bool ok = expect_true(error_of(pending, true, false, 101.0).empty(), "a frame is pending");
  ok &= expect_true(error_of(pending, false, false, 111.0).empty(), "the deadline has not passed");
  const std::string error = error_of(pending, false, false, 111.5);
  ok &= expect_true(error == "stream 2: model models/m.tar.gz returned no result for 10 s while "
                             "its decoder sent frames",
                    "the error names the stream and model: " + error);
  return ok;
}

bool test_later_frames_keep_the_first_pending_time() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  pending.observe(false, true, 100.0);
  pending.observe(true, false, 101.0);
  pending.observe(true, false, 106.0);
  return expect_true(!error_of(pending, true, false, 111.5).empty(),
                     "the deadline counts from the first pending frame");
}

bool test_result_clears_the_pending_frame() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  pending.observe(false, true, 100.0);
  pending.observe(true, false, 101.0);
  pending.observe(false, true, 105.0);
  bool ok = expect_true(error_of(pending, false, false, 200.0).empty(),
                        "a result clears the pending frame");
  pending.observe(true, false, 205.0);
  ok &= expect_true(error_of(pending, false, false, 215.0).empty(),
                    "the next frame starts a new deadline");
  ok &= expect_true(!error_of(pending, false, false, 215.5).empty(), "the new deadline passes");
  return ok;
}

bool test_frame_with_its_result_in_one_round_is_not_pending() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  pending.observe(true, true, 100.0);
  return expect_true(error_of(pending, false, false, 1'000'000.0).empty(),
                     "a frame with its result starts no deadline");
}

bool test_first_result_has_its_own_deadline() {
  PendingResults pending(1, "m.tar.gz", 10.0, 30.0);
  pending.observe(true, false, 100.0);
  bool ok = expect_true(error_of(pending, false, false, 125.0).empty(),
                        "the first result has 30 s");
  const std::string error = error_of(pending, false, false, 130.5);
  ok &= expect_true(
      error == "stream 1: model m.tar.gz returned no result for 30 s while its decoder sent frames",
      "the first deadline passes: " + error);
  return ok;
}

bool test_paused_camera_waits_without_a_limit() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  bool ok = expect_true(error_of(pending, false, false, 1'000'000.0).empty(),
                        "no frame, no deadline");
  pending.observe(false, true, 1'000'001.0);
  ok &= expect_true(error_of(pending, false, false, 2'000'000.0).empty(),
                    "no frame after the last result, no deadline");
  return ok;
}

bool test_per_stream_lists_need_one_entry_per_stream(const std::string& binary) {
  const std::string error =
      run_error(binary, "test_per_stream_lists_need_one_entry_per_stream", "app.model_paths",
                {"a_mpk.tar.gz", "b_mpk.tar.gz", "c_mpk.tar.gz"});
  return expect_true(
      error == "[ERR] app.model_paths has 3 entries and input.source has 4: give one per stream",
      "a short list fails with its key: " + error);
}

bool test_unknown_model_id_names_the_choices(const std::string& binary) {
  const std::string error =
      run_error(binary, "test_unknown_model_id_names_the_choices", "app.model_ids",
                {"yolov8", "yolo26-obb", "yolo26-pose", "yolo26"});
  return expect_true(error == "[ERR] app.model_ids: unknown value 'yolo26-obb' (expected yolo26, "
                              "yolo26-pose, yolo26-seg, yolov8, yolov8-pose, yolov8-seg)",
                     "an unknown model ID fails with the choices: " + error);
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  bool ok = test_shipped_config_is_usable();
  ok &= test_objects_clamp_boxes_to_the_frame();
  ok &= test_mask_rect_maps_the_frame_through_the_letterbox();
  ok &= test_segment_outline_follows_the_mask();
  ok &= test_poses_round_to_whole_pixels_and_name_the_keypoints();
  ok &= test_decodes_with_fewer_rows_than_boxes_fail();
  ok &= test_data_json_leaves_out_the_lowest_confidence_entries();
  ok &= test_late_result_names_the_stream_and_model();
  ok &= test_later_frames_keep_the_first_pending_time();
  ok &= test_result_clears_the_pending_frame();
  ok &= test_frame_with_its_result_in_one_round_is_not_pending();
  ok &= test_first_result_has_its_own_deadline();
  ok &= test_paused_camera_waits_without_a_limit();
  ok &= test_per_stream_lists_need_one_entry_per_stream(binary);
  ok &= test_unknown_model_id_names_the_choices(binary);
  return ok ? 0 : 1;
}
