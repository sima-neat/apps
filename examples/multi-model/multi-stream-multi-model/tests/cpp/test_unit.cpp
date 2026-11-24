// Unit tests for the multi-stream multi-model example: the checks the application leaves to its
// config, the Insight metadata of each task, and the pending-result deadline.
// tests/python/test_unit.py runs the same tests in Python. Python also runs its pull loop against a
// fake Run; a C++ Run cannot be faked, so here the deadline is tested on PendingResults alone.

#include "examples/multi-model/multi-stream-multi-model/src/cpp/utils/insight_metadata.h"
#include "examples/multi-model/multi-stream-multi-model/src/cpp/utils/pending_results.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using insight_metadata::Entry;
using pending_results::PendingResults;
using sima_examples::testing::expect_true;

namespace {

const std::vector<std::string> kLabels = {"person", "bicycle", "car"};

std::string check_error(const PendingResults& pending, double now_s) {
  try {
    pending.check(now_s);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
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

// A deadline of 10 s after the stream's first result.
PendingResults started(int stream_index, const std::string& model_path) {
  PendingResults pending(stream_index, model_path, 10.0);
  pending.result_arrived();
  return pending;
}

bool test_pending_result_past_the_deadline_names_the_stream_and_model() {
  PendingResults pending = started(2, "models/m.tar.gz");
  pending.frame_arrived(100.0);
  bool ok = expect_true(check_error(pending, 110.0).empty(), "the deadline has not passed");
  ok &= expect_true(check_error(pending, 110.5) ==
                        "stream 2: model models/m.tar.gz returned no result for 10 s while its "
                        "camera sent frames",
                    "the error names the stream and model: " + check_error(pending, 110.5));
  return ok;
}

bool test_later_frames_keep_the_first_pending_time() {
  PendingResults pending = started(0, "m.tar.gz");
  pending.frame_arrived(100.0);
  pending.frame_arrived(105.0);
  return expect_true(!check_error(pending, 110.5).empty(),
                     "the deadline counts from the first pending frame");
}

bool test_result_clears_the_pending_frame() {
  PendingResults pending = started(0, "m.tar.gz");
  pending.frame_arrived(100.0);
  pending.result_arrived();
  bool ok = expect_true(check_error(pending, 200.0).empty(), "a result clears the pending frame");
  pending.frame_arrived(205.0);
  ok &= expect_true(check_error(pending, 215.0).empty(), "the next frame starts a new deadline");
  ok &= expect_true(!check_error(pending, 215.5).empty(), "the new deadline passes");
  return ok;
}

bool test_frames_before_the_first_result_are_not_pending() {
  // The decoder may wait for a keyframe before the model sees a frame.
  PendingResults pending(0, "m.tar.gz", 10.0);
  pending.frame_arrived(100.0);
  bool ok = expect_true(check_error(pending, 1'000'000.0).empty(),
                        "frames before the first result are not pending");
  pending.result_arrived();
  pending.frame_arrived(1'000'001.0);
  ok &= expect_true(!check_error(pending, 1'000'011.5).empty(),
                    "the deadline runs after the first result");
  return ok;
}

bool test_paused_camera_waits_without_a_limit() {
  PendingResults pending = started(0, "m.tar.gz");
  bool ok = expect_true(check_error(pending, 1'000'000.0).empty(), "no frame, no deadline");
  pending.frame_arrived(100.0);
  pending.result_arrived();
  ok &= expect_true(check_error(pending, 1'000'000.0).empty(),
                    "no frame after the last result, no deadline");
  return ok;
}

} // namespace

int main() {
  bool ok = test_shipped_config_is_usable();
  ok &= test_objects_clamp_boxes_to_the_frame();
  ok &= test_mask_rect_maps_the_frame_through_the_letterbox();
  ok &= test_segment_outline_follows_the_mask();
  ok &= test_poses_round_to_whole_pixels_and_name_the_keypoints();
  ok &= test_decodes_with_fewer_rows_than_boxes_fail();
  ok &= test_data_json_leaves_out_the_lowest_confidence_entries();
  ok &= test_pending_result_past_the_deadline_names_the_stream_and_model();
  ok &= test_later_frames_keep_the_first_pending_time();
  ok &= test_result_clears_the_pending_frame();
  ok &= test_frames_before_the_first_result_are_not_pending();
  ok &= test_paused_camera_waits_without_a_limit();
  return ok ? 0 : 1;
}
