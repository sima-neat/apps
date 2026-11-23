// Unit tests for the thermal face detector: the checks the application leaves to its config, and
// the yolov5s-face decode in utils/yolov5_face.h. tests/python/test_unit.py runs the same checks
// in Python.

#include "../../src/cpp/utils/yolov5_face.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;
using yolov5_face::Face;
using yolov5_face::SplitHead;

namespace {

// The planted faces sit on the 100x100 grid (stride 8), in cell x 20, y 10.
constexpr int kCellX = 20;
constexpr int kCellY = 10;
// Logits whose sigmoids are 0.8 and 0.5: a face score of 0.4.
const std::pair<float, float> kScoreLogits = {std::log(4.0f), 0.0f};

bool near(float actual, float expected) {
  return std::fabs(actual - expected) <= 1e-4f;
}

bool box_is(const Face& face, float x1, float y1, float x2, float y2) {
  return near(face.x1, x1) && near(face.y1, y1) && near(face.x2, x2) && near(face.y2, y2);
}

float sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

// Six [1, H, W, C] heads, largest grid first, with a face planted at (kCellX, kCellY) for each
// (anchor, (object logit, class logit)). Every other score is about 2e-9.
std::vector<SplitHead>
heads_with_faces(std::initializer_list<std::pair<int, std::pair<float, float>>> faces) {
  std::vector<SplitHead> heads;
  for (const int grid : {100, 50, 25}) {
    SplitHead box{{1, grid, grid, 18}, std::vector<float>(static_cast<size_t>(grid) * grid * 18)};
    SplitHead landmarks{{1, grid, grid, 30},
                        std::vector<float>(static_cast<size_t>(grid) * grid * 30)};
    for (size_t i = 0; i < box.values.size(); ++i) {
      if (i % 6 == 4 || i % 6 == 5) {
        box.values[i] = -10.0f;
      }
    }
    if (grid == 100) {
      const size_t cell = static_cast<size_t>(kCellY) * grid + kCellX;
      for (const auto& [anchor, logits] : faces) {
        box.values[cell * 18 + anchor * 6 + 4] = logits.first;
        box.values[cell * 18 + anchor * 6 + 5] = logits.second;
        // Landmark k is (k, -k) anchor sizes from the cell's corner.
        for (int k = 0; k < 5; ++k) {
          landmarks.values[cell * 30 + anchor * 10 + 2 * k] = static_cast<float>(k);
          landmarks.values[cell * 30 + anchor * 10 + 2 * k + 1] = static_cast<float>(-k);
        }
      }
    }
    heads.push_back(std::move(box));
    heads.push_back(std::move(landmarks));
  }
  return heads;
}

// The head with its axes in `order`, such as {0, 1, 3, 2} for [1, H, C, W].
SplitHead transposed(const SplitHead& head, const std::vector<int>& order) {
  SplitHead out{{}, std::vector<float>(head.values.size())};
  for (const int axis : order) {
    out.shape.push_back(head.shape[axis]);
  }
  const int64_t h = head.shape[1], w = head.shape[2], c = head.shape[3];
  for (int64_t y = 0; y < h; ++y) {
    for (int64_t x = 0; x < w; ++x) {
      for (int64_t ch = 0; ch < c; ++ch) {
        const int64_t index[4] = {0, y, x, ch};
        int64_t target = 0;
        for (int axis = 1; axis < 4; ++axis) {
          target = target * out.shape[axis] + index[order[axis]];
        }
        out.values[static_cast<size_t>(target)] =
            head.values[static_cast<size_t>((y * w + x) * c + ch)];
      }
    }
  }
  return out;
}

std::string thrown_message(const std::vector<SplitHead>& heads) {
  try {
    (void)yolov5_face::decode_faces(heads, 0.25f, 0.45f, 50);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_the_model_labels_and_usable_values() {
  const fs::path common_dir =
      "examples/face-detection/single-stream-thermal-face-detector/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  bool ok = expect_true(fs::path(config.string("model.path")).filename() ==
                            "yolov5s_face_raw_split_mpk.tar.gz",
                        "model.path names the yolov5s-face package");
  // The model has one class, and its name labels every pose.
  ok &= expect_true(sima_examples::load_labels(config.string("model.labels")) ==
                        std::vector<std::string>{"face"},
                    "model.labels lists the one class, face");
  ok &= expect_true(config.string("input.source").rfind("rtsp://", 0) == 0,
                    "input.source is an RTSP URL");
  ok &= expect_true(
      config.integer("input.width") >= 0 && config.integer("input.height") >= 0 &&
          config.integer("input.fps") >= 0 && config.integer("input.rtsp.latency_ms") >= 0 &&
          config.integer("inference.frames") >= 0 && config.integer("inference.max_detections") > 0,
      "the shipped sizes, latency, and frame counts are in range");
  ok &= expect_true(
      config.number("inference.min_score") >= 0.0 && config.number("inference.min_score") <= 1.0 &&
          config.number("inference.nms_iou") >= 0.0 && config.number("inference.nms_iou") <= 1.0,
      "the shipped score and IoU thresholds are from 0 to 1");
  for (const std::string key :
       {"output.insight.video_port_base", "output.insight.metadata_port_base"}) {
    ok &= expect_true(config.integer(key) >= 1 && config.integer(key) <= 65535, key + " is a port");
  }
  // Core sends to 127.0.0.1 when the host is empty.
  ok &=
      expect_true(!config.string("output.insight.host").empty() && config.boolean("input.rtsp.tcp"),
                  "the shipped Insight host is set and RTSP runs over TCP");
  return ok;
}

// Anchor 1 of the first level is 8x10 pixels: a zero box offset centers the face on (20.5, 10.5)
// cells of 8 pixels, the size of the anchor.
bool test_decode_faces_places_the_box_and_landmarks_of_a_planted_face() {
  const auto faces =
      yolov5_face::decode_faces(heads_with_faces({{1, kScoreLogits}}), 0.25f, 0.45f, 50);
  bool landmarks_ok = faces.size() == 1;
  for (int k = 0; landmarks_ok && k < 5; ++k) {
    landmarks_ok = near(faces[0].landmarks[k].x, 160.0f + 8.0f * k) &&
                   near(faces[0].landmarks[k].y, 80.0f - 10.0f * k);
  }
  return expect_true(faces.size() == 1 && box_is(faces[0], 160, 79, 168, 89) &&
                         near(faces[0].score, 0.4f) && landmarks_ok,
                     "a planted face decodes to its box, score, and landmarks");
}

bool test_decode_faces_drops_faces_at_or_below_min_score() {
  return expect_true(
      yolov5_face::decode_faces(heads_with_faces({{1, kScoreLogits}}), 0.45f, 0.45f, 50).empty(),
      "a face scoring 0.4 is dropped at min_score 0.45");
}

// Neat delivers raw heads as [1, H, C, W]; ONNX runs give [1, C, H, W].
bool test_decode_faces_reads_every_head_layout_alike() {
  const auto heads = heads_with_faces({{1, kScoreLogits}, {2, {5.0f, 5.0f}}});
  const auto expected = yolov5_face::decode_faces(heads, 0.25f, 0.45f, 50);
  bool ok = expect_true(expected.size() == 2, "two planted faces decode");
  for (const std::vector<int>& order :
       {std::vector<int>{0, 1, 3, 2}, std::vector<int>{0, 3, 1, 2}}) {
    std::vector<SplitHead> permuted;
    for (const SplitHead& head : heads) {
      permuted.push_back(transposed(head, order));
    }
    const auto actual = yolov5_face::decode_faces(permuted, 0.25f, 0.45f, 50);
    bool same = actual.size() == expected.size();
    for (size_t i = 0; same && i < actual.size(); ++i) {
      same = box_is(actual[i], expected[i].x1, expected[i].y1, expected[i].x2, expected[i].y2) &&
             near(actual[i].score, expected[i].score) &&
             near(actual[i].landmarks[4].x, expected[i].landmarks[4].x) &&
             near(actual[i].landmarks[4].y, expected[i].landmarks[4].y);
    }
    ok &= expect_true(same, "a permuted head layout decodes like [1, H, W, C]");
  }
  return ok;
}

// Two faces on one cell, 8x10 (score 0.4) and 13x16 (score near 1), overlap by 80 / 208 = 0.385
// of their union.
bool test_decode_faces_keeps_the_higher_score_of_two_overlapping_faces() {
  const auto heads = heads_with_faces({{1, kScoreLogits}, {2, {10.0f, 10.0f}}});
  const float high = sigmoid(10.0f) * sigmoid(10.0f);
  const auto suppressed = yolov5_face::decode_faces(heads, 0.25f, 0.3f, 50);
  const auto kept = yolov5_face::decode_faces(heads, 0.25f, 0.45f, 50);
  const auto capped = yolov5_face::decode_faces(heads, 0.25f, 0.45f, 1);
  bool ok = expect_true(suppressed.size() == 1 && box_is(suppressed[0], 157.5f, 76, 170.5f, 92),
                        "NMS at 0.3 keeps only the higher-scoring face");
  ok &= expect_true(kept.size() == 2 && near(kept[0].score, high) && near(kept[1].score, 0.4f),
                    "NMS at 0.45 keeps both faces, highest score first");
  ok &= expect_true(capped.size() == 1 && near(capped[0].score, high),
                    "max_detections 1 keeps the highest score");
  return ok;
}

bool test_decode_faces_rejects_heads_it_cannot_pair() {
  const auto heads = heads_with_faces({});
  const std::vector<SplitHead> five(heads.begin(), heads.begin() + 5);
  auto odd = five;
  odd.push_back({{1, 7, 7, 7}, std::vector<float>(343)});
  std::vector<SplitHead> unpaired(heads.begin(), heads.begin() + 4);
  unpaired.push_back(heads[0]);
  unpaired.push_back(heads[5]);
  bool ok = expect_true(thrown_message(five).find("expected 6 split heads") != std::string::npos,
                        "five heads are rejected");
  ok &= expect_true(thrown_message(odd).find("no axis of 18 or 30 channels") != std::string::npos,
                    "a head with no 18 or 30 channel axis is rejected");
  ok &= expect_true(thrown_message(unpaired).find("a box and a landmark head on 3 grids") !=
                        std::string::npos,
                    "a grid without its landmark head is rejected");
  return ok;
}

bool test_letterbox_centers_the_frame_on_the_canvas() {
  const auto wide = yolov5_face::letterbox(1600, 900);
  const auto tall = yolov5_face::letterbox(640, 512);
  // 901 x 0.5 = 450.5 rounds half away from zero to 451, as Python's letterbox does too.
  const auto half = yolov5_face::letterbox(1600, 901);
  return expect_true(wide.scale == 0.5f && wide.pad_left == 0 && wide.pad_top == 175 &&
                         tall.scale == 1.25f && tall.pad_left == 0 && tall.pad_top == 80 &&
                         half.pad_top == 174,
                     "the letterbox scale and padding center the frame");
}

bool test_to_frame_maps_canvas_pixels_to_the_frame_and_clips_only_boxes() {
  Face inside{100, 275, 300, 475, 0.9f};
  inside.landmarks.fill({400, 575});
  Face outside{700, 150, 900, 300, 0.8f};
  outside.landmarks.fill({-10, 100});
  const auto mapped =
      yolov5_face::to_frame({inside, outside}, yolov5_face::letterbox(1600, 900), 1600, 900);
  return expect_true(
      box_is(mapped[0], 200, 200, 600, 600) && box_is(mapped[1], 1400, 0, 1600, 250) &&
          near(mapped[0].landmarks[0].x, 800) && near(mapped[0].landmarks[0].y, 800) &&
          near(mapped[1].landmarks[0].x, -20) && near(mapped[1].landmarks[0].y, -150) &&
          near(mapped[0].score, 0.9f),
      "canvas pixels map to the frame, and only boxes are clipped");
}

bool test_poses_json_gives_each_face_five_named_keypoints_with_its_score() {
  Face face;
  face.score = 0.5f;
  for (int k = 0; k < 5; ++k) {
    face.landmarks[k] = {2.0f * k, 2.0f * k + 1};
  }
  const std::vector<std::string> names = {"eye_l", "eye_r", "nose_tip", "mouth_l", "mouth_r"};
  nlohmann::json keypoints = nlohmann::json::array();
  for (int k = 0; k < 5; ++k) {
    keypoints.push_back(
        {{"name", names[k]}, {"x", 2.0 * k}, {"y", 2.0 * k + 1}, {"confidence", 0.5}});
  }
  nlohmann::json pose = {{"id", "face_1"}, {"label", "face"}, {"keypoints", keypoints}};
  const nlohmann::json expected = {{"poses", nlohmann::json::array({pose})}};
  return expect_true(nlohmann::json::parse(yolov5_face::poses_json({face}, "face")) == expected,
                     "each face is one pose of five named keypoints carrying its score");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_the_model_labels_and_usable_values();
  ok &= test_decode_faces_places_the_box_and_landmarks_of_a_planted_face();
  ok &= test_decode_faces_drops_faces_at_or_below_min_score();
  ok &= test_decode_faces_reads_every_head_layout_alike();
  ok &= test_decode_faces_keeps_the_higher_score_of_two_overlapping_faces();
  ok &= test_decode_faces_rejects_heads_it_cannot_pair();
  ok &= test_letterbox_centers_the_frame_on_the_canvas();
  ok &= test_to_frame_maps_canvas_pixels_to_the_frame_and_clips_only_boxes();
  ok &= test_poses_json_gives_each_face_five_named_keypoints_with_its_score();
  return ok ? 0 : 1;
}
