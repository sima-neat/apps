// tests/python/test_unit.py's tests, less those that need a Run or main.cpp's private tables: the
// deadline is tested on PendingResults.observe().

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
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
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

// One round that must not throw. A throw outside error_of would end the test binary.
bool observes(PendingResults& pending, bool frame, bool result, double now_s) {
  const std::string error = error_of(pending, frame, result, now_s);
  return expect_true(error.empty(),
                     "the round at " + std::to_string(now_s) + " s passes: " + error);
}

// The shipped model entries, each by field.
std::vector<ConfigScalars> shipped_models() {
  std::vector<ConfigScalars> models;
  for (const auto& model : example_common_config(kExampleName).mapping_list("model")) {
    ConfigScalars fields = {{"id", model.string("id")}, {"path", model.string("path")}};
    if (const auto labels = model.string_value("labels")) {
      fields["labels"] = *labels;
    }
    models.push_back(fields);
  }
  return models;
}

// The last stderr line for the shipped config with `models` as its entries, the test's directory
// as <dir>. The set frame size and rate skip the probe, so no source is needed.
std::string run_error(const std::string& binary, const std::string& test_name,
                      const std::vector<ConfigScalars>& models, std::string* all = nullptr) {
  const std::string dir = create_test_scratch_dir(kExampleName, test_name);
  const fs::path config_path = fs::path(dir) / "config.yaml";
  // The writer keeps only the shipped scalars, so the test passes the streams too.
  write_e2e_config(
      kExampleName, config_path,
      {{"input.width", "1280"}, {"input.height", "720"}, {"input.fps", "30"}},
      {{"input.source", example_common_config(kExampleName).string_list("input.source")}},
      {{"model", models}});
  const ProcessResult result = spawn_and_wait(binary, {"--config", config_path.string()}, 60000);
  remove_dir(dir);
  std::string text = result.stderr_text;
  while (!text.empty() && text.back() == '\n') {
    text.pop_back();
  }
  const auto newline = text.rfind('\n');
  std::string line = result.exit_code == 1
                         ? text.substr(newline == std::string::npos ? 0 : newline + 1)
                         : "exit " + std::to_string(result.exit_code) + ": " + text;
  for (std::string* shown : {&line, &text}) {
    for (auto at = shown->find(dir); at != std::string::npos; at = shown->find(dir, at)) {
      shown->replace(at, dir.size(), "<dir>");
    }
  }
  if (all != nullptr) {
    *all = text;
  }
  return line;
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_is_usable() {
  const fs::path common_dir = "examples/multi-model/multi-stream-multi-model/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const auto sources = config.string_list("input.source");
  bool ok = expect_true(!sources.empty() && sources.size() <= 4U,
                        "the shipped config lists one to four streams");
  const auto models = config.mapping_list("model");
  ok &= expect_true(models.size() == sources.size(), "one model per stream");
  std::vector<std::string> model_ids;
  std::vector<std::string> model_paths;
  for (const auto& model : models) {
    model_ids.push_back(model.string("id"));
    model_paths.push_back(model.string("path"));
  }
  ok &= expect_true(model_ids == std::vector<std::string>{"yolov8", "yolo26-seg", "yolo26-pose",
                                                          "yolo26"},
                    "the model entries name the four supported packages");
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
  for (std::size_t i = 0; i < models.size(); ++i) {
    if (model_ids[i] == "yolo26-pose") {
      ok &= expect_true(!models[i].string_value("labels").has_value(),
                        "the pose model has no labels");
      continue;
    }
    // The labels path starts at the working directory, the Apps root here.
    std::ifstream labels_file(models[i].string("labels"));
    int labels = 0;
    for (std::string line; std::getline(labels_file, line);) {
      labels += line.empty() ? 0 : 1;
    }
    ok &= expect_true(labels == 80, model_ids[i] + " has the 80 COCO labels it scores");
  }
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
  // A named Entry: a range-for over Entry::parse(data)["objects"] would keep only the reference
  // operator[] returns, into a parse result destroyed before the loop.
  const Entry parsed = Entry::parse(data);
  for (const auto& entry : parsed["objects"]) {
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
  bool ok = observes(pending, true, true, 100.0);
  ok &= expect_true(error_of(pending, true, false, 101.0).empty(), "a frame is pending");
  ok &= expect_true(error_of(pending, false, false, 111.0).empty(), "the deadline has not passed");
  const std::string error = error_of(pending, false, false, 111.5);
  ok &= expect_true(error == "stream 2: model models/m.tar.gz returned no result for 10 s while "
                             "its decoder sent frames",
                    "the error names the stream and model: " + error);
  return ok;
}

bool test_later_frames_keep_the_first_pending_time() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  bool ok = observes(pending, false, true, 100.0);
  ok &= observes(pending, true, false, 101.0);
  ok &= observes(pending, true, false, 106.0);
  ok &= expect_true(!error_of(pending, true, false, 111.5).empty(),
                    "the deadline counts from the first pending frame");
  return ok;
}

bool test_result_clears_the_pending_frame() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  bool ok = observes(pending, false, true, 100.0);
  ok &= observes(pending, true, false, 101.0);
  ok &= observes(pending, false, true, 105.0);
  ok &= expect_true(error_of(pending, false, false, 200.0).empty(),
                    "a result clears the pending frame");
  ok &= observes(pending, true, false, 205.0);
  ok &= expect_true(error_of(pending, false, false, 215.0).empty(),
                    "the next frame starts a new deadline");
  ok &= expect_true(!error_of(pending, false, false, 215.5).empty(), "the new deadline passes");
  return ok;
}

bool test_frame_with_its_result_in_one_round_is_not_pending() {
  PendingResults pending(0, "m.tar.gz", 10.0, 30.0);
  bool ok = observes(pending, true, true, 100.0);
  ok &= expect_true(error_of(pending, false, false, 1'000'000.0).empty(),
                    "a frame with its result starts no deadline");
  return ok;
}

bool test_first_result_has_its_own_deadline() {
  PendingResults pending(1, "m.tar.gz", 10.0, 30.0);
  bool ok = observes(pending, true, false, 100.0);
  ok &= expect_true(error_of(pending, false, false, 125.0).empty(), "the first result has 30 s");
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
  ok &= observes(pending, false, true, 1'000'001.0);
  ok &= expect_true(error_of(pending, false, false, 2'000'000.0).empty(),
                    "no frame after the last result, no deadline");
  return ok;
}

// What mapping_list("model") throws for a config of `text`, or "" when it throws nothing.
std::string mapping_list_error(const std::string& test_name, const std::string& text,
                               std::string* config_path = nullptr) {
  const std::string dir = create_test_scratch_dir(kExampleName, test_name);
  const fs::path path = fs::path(dir) / "config.yaml";
  std::ofstream(path) << text;
  if (config_path != nullptr) {
    *config_path = path.string();
  }
  std::string error;
  try {
    static_cast<void>(sima_examples::Config::load(path).mapping_list("model"));
  } catch (const std::runtime_error& thrown) {
    error = thrown.what();
  }
  remove_dir(dir);
  return error;
}

const char* const kModels = "model:\n"
                            "  - id: yolov8\n    path: a.tar.gz\n    labels: coco.txt\n"
                            "  - id: 5\n    path: b.tar.gz\n";

// The C++ side of tests/scripts/test_config.py: Config::mapping_list in support/.
bool test_mapping_list_reads_each_entry_by_field() {
  const std::string dir = create_test_scratch_dir(kExampleName, "mapping_list_reads");
  const fs::path path = fs::path(dir) / "config.yaml";
  std::ofstream(path) << kModels;
  const auto entries = sima_examples::Config::load(path).mapping_list("model");
  bool ok = expect_true(entries.size() == 2U, "two entries");
  if (entries.size() == 2U) {
    ok &= expect_true(entries[0].string("id") == "yolov8" &&
                          entries[0].string("labels") == "coco.txt" &&
                          entries[1].string("path") == "b.tar.gz",
                      "each entry reads by field");
    ok &= expect_true(entries[0].choice("id", std::map<std::string, int>{{"yolov8", 8}}) == 8,
                      "choice reads an entry's field");
    // C++ reads any scalar as a string, while Python rejects a number here.
    ok &= expect_true(entries[1].string("id") == "5", "C++ reads id: 5 as \"5\"");
    std::string error;
    try {
      static_cast<void>(entries[1].string("labels"));
    } catch (const std::runtime_error& thrown) {
      error = thrown.what();
    }
    ok &= expect_true(error == path.string() + " is missing model[1].labels",
                      "a missing field names the entry: " + error);
    error.clear();
    try {
      static_cast<void>(entries[0].choice("id", std::map<std::string, int>{{"yolo26", 26}}));
    } catch (const std::runtime_error& thrown) {
      error = thrown.what();
    }
    ok &= expect_true(error == "model[0].id: unknown value 'yolov8' (expected yolo26)",
                      "an unknown value names the entry: " + error);
  }
  remove_dir(dir);
  return ok;
}

bool test_mapping_list_rejects_what_is_not_a_list_of_mappings() {
  std::string path;
  std::string error = mapping_list_error("mapping_list_missing", "input:\n  fps: 30\n", &path);
  bool ok = expect_true(error == path + " is missing model", "a missing list: " + error);
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"model:\n  path: a.tar.gz\n", "model must be a list of mappings, got a mapping"},
      {"model: a.tar.gz\n", "model must be a list of mappings, got 'a.tar.gz'"},
      {"model: []\n", "model must be a non-empty list of mappings"},
      {"model:\n", "model must be a list of mappings, got null"},
      {"model:\n  - path: a.tar.gz\n  -\n", "model[1] must be a mapping, got null"},
      {"model:\n  - path: a.tar.gz\n  - [b, c]\n", "model[1] must be a mapping, got a list"},
      {"model:\n  - path: a.tar.gz\n  - b.tar.gz\n", "model[1] must be a mapping, got 'b.tar.gz'"},
  };
  for (const auto& [text, expected] : cases) {
    error = mapping_list_error("mapping_list_rejects", text);
    ok &= expect_true(error == expected, expected + ": " + error);
  }
  return ok;
}

bool test_model_needs_one_entry_per_stream(const std::string& binary) {
  auto models = shipped_models();
  models.pop_back();
  const std::string error = run_error(binary, "test_model_needs_one_entry_per_stream", models);
  return expect_true(error ==
                         "[ERR] model has 3 entries and input.source has 4: give one per stream",
                     "a short model list fails with its key: " + error);
}

bool test_unknown_model_id_names_the_entry_and_the_choices(const std::string& binary) {
  auto models = shipped_models();
  models[1]["id"] = "yolo26-obb";
  const std::string error =
      run_error(binary, "test_unknown_model_id_names_the_entry_and_the_choices", models);
  return expect_true(error == "[ERR] model[1].id: unknown value 'yolo26-obb' (expected yolo26, "
                              "yolo26-pose, yolo26-seg, yolov8, yolov8-pose, yolov8-seg)",
                     "an unknown model ID fails with the entry and the choices: " + error);
}

bool test_detection_model_without_labels_names_the_entry(const std::string& binary) {
  auto models = shipped_models();
  models[0].erase("labels");
  const std::string error =
      run_error(binary, "test_detection_model_without_labels_names_the_entry", models);
  return expect_true(error == "[ERR] <dir>/config.yaml is missing model[0].labels",
                     "a detection model without labels fails with the entry: " + error);
}

bool test_each_model_reads_its_own_labels(const std::string& binary) {
  auto models = shipped_models();
  models[1]["labels"] = "missing_labels.txt";
  const std::string error = run_error(binary, "test_each_model_reads_its_own_labels", models);
  return expect_true(error == "[ERR] labels file does not exist: missing_labels.txt",
                     "model[1] reads its own labels: " + error);
}

bool test_pose_model_needs_no_labels(const std::string& binary) {
  // The shipped pose entry has no labels. Every entry is read before the first model loads, so a
  // run whose first model is missing gets past the config and fails on that model.
  auto models = shipped_models();
  bool ok = expect_true(models.at(2).count("labels") == 0, "the shipped pose entry has no labels");
  models[0]["path"] = "missing_mpk.tar.gz";
  std::string all;
  const std::string error = run_error(binary, "test_pose_model_needs_no_labels", models, &all);
  // Core's error for a missing package: a line that names model[0].path, then a hint.
  const std::string first = all.substr(0, all.find('\n'));
  const std::string named = "invalid_archive: archive path does not exist or is not a regular "
                            "file: <dir>/missing_mpk.tar.gz";
  ok &= expect_true(first.rfind("[ERR] ", 0) == 0 && first.size() >= named.size() &&
                        first.compare(first.size() - named.size(), named.size(), named) == 0,
                    "a pose model without labels fails on model[0].path: " + first);
  ok &= expect_true(
      error == "Hint: Re-export the model archive and verify it contains a valid MPK contract.",
      "Core's hint follows: " + error);
  return ok;
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
  ok &= test_mapping_list_reads_each_entry_by_field();
  ok &= test_mapping_list_rejects_what_is_not_a_list_of_mappings();
  ok &= test_model_needs_one_entry_per_stream(binary);
  ok &= test_unknown_model_id_names_the_entry_and_the_choices(binary);
  ok &= test_detection_model_without_labels_names_the_entry(binary);
  ok &= test_each_model_reads_its_own_labels(binary);
  ok &= test_pose_model_needs_no_labels(binary);
  return ok ? 0 : 1;
}
