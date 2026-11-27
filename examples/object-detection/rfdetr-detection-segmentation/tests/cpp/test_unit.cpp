#include "../../src/cpp/utils/postprocess.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_stream_and_labels() {
  const fs::path common_dir = "examples/object-detection/rfdetr-detection-segmentation/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const std::string codec = config.string("input.codec");
  bool ok = expect_true(config.string("model.id") == "det-small", "the shipped model is det-small");
  // The decoder package sits next to the backbone that model.path names.
  ok &= expect_true(fs::path(config.string("model.path")).filename() ==
                        "rf_detr_small_w8a8_mixed_w16a16_backbone_mpk.tar.gz",
                    "model.path names the backbone package");
  ok &= expect_true(codec == "h264" || codec == "h265" || codec == "mjpeg",
                    "the shipped codec is supported");
  // The model's class IDs index the labels directly.
  ok &= expect_true(sima_examples::load_labels(config.string("model.labels")).size() ==
                        static_cast<std::size_t>(rfdetr::kNumClasses),
                    "model.labels lists every COCO class ID");
  ok &= expect_true(config.number("inference.min_score") >= 0.0 &&
                        config.number("inference.min_score") <= 1.0 &&
                        !config.string("output.insight.host").empty(),
                    "the shipped inference and output values are usable");
  return ok;
}

std::vector<std::string> coco_labels() {
  std::vector<std::string> labels(91, "N/A");
  labels[1] = "person";
  return labels;
}

bool test_top_proposals_are_stable_and_best_first() {
  std::vector<float> scores(305, 0.0F);
  scores[3] = 2.0F;
  scores[4] = 2.0F;
  std::vector<float> proposals(scores.size() * 4U, 0.0F);
  for (std::size_t index = 0; index < scores.size(); ++index) {
    proposals[index * 4U] = static_cast<float>(index);
  }
  const auto best = rfdetr::top_proposals(scores, proposals, 300);
  return expect_true(best.size() == 1200U && best[0] == 3.0F && best[4] == 4.0F && best[8] == 0.0F,
                     "top proposals are the 300 best, best first, ties in index order");
}

bool test_detection_skips_unused_classes_and_scales_to_the_frame() {
  std::vector<float> boxes(300U * 4U, 0.0F);
  for (const std::size_t query : {0U, 1U}) {
    boxes[query * 4U] = 0.5F;
    boxes[query * 4U + 1U] = 0.5F;
    boxes[query * 4U + 2U] = 0.5F;
    boxes[query * 4U + 3U] = 0.25F;
  }
  std::vector<float> logits(300U * 91U, -20.0F);
  logits[1] = 10.0F;        // person on query 0
  logits[91U + 2U] = 12.0F; // an unused class on query 1, ranked first
  rfdetr::PostprocessOptions options;
  options.width = 1920;
  options.height = 1080;
  options.min_score = 0.5F;
  options.max_detections = 10;
  const auto objects = nlohmann::json::parse(
      rfdetr::detection_json(boxes, logits, coco_labels(), options))["objects"];
  const auto near = [](const nlohmann::json& value, float expected) {
    return std::abs(value.get<float>() - expected) < 0.01F;
  };
  return expect_true(objects.size() == 1U && objects[0]["id"] == "obj_1" &&
                         objects[0]["label"] == "person" && near(objects[0]["bbox"][0], 480.0F) &&
                         near(objects[0]["bbox"][1], 405.0F) &&
                         near(objects[0]["bbox"][2], 960.0F) && near(objects[0]["bbox"][3], 270.0F),
                     "detection skips unused classes and scales boxes to the frame");
}

bool test_segmentation_traces_a_polygon_inside_the_box() {
  bool ok = true;
  // Medium carries 200 queries and 108x108 masks; nano carries 100 queries and 78x78 masks.
  for (const auto& [queries, mask_size] : {std::pair{200, 108}, std::pair{100, 78}}) {
    const auto count = static_cast<std::size_t>(queries);
    const auto side = static_cast<std::size_t>(mask_size);
    std::vector<float> boxes(count * 4U, 0.0F);
    for (const std::size_t query : {0U, 1U}) {
      for (std::size_t value = 0; value < 4U; ++value) {
        boxes[query * 4U + value] = 0.5F;
      }
    }
    std::vector<float> logits(count * 91U, -20.0F);
    logits[1] = 11.0F;        // person on query 0
    logits[91U + 1U] = 10.0F; // person on query 1, whose mask is empty
    std::vector<float> masks(side * side * count, -20.0F);
    for (std::size_t y = side * 3U / 8U; y < side * 5U / 8U; ++y) {
      for (std::size_t x = side * 3U / 8U; x < side * 5U / 8U; ++x) {
        masks[(y * side + x) * count] = 10.0F;
      }
    }
    for (const int grid_size : {mask_size, 432, 640}) {
      rfdetr::PostprocessOptions options;
      options.width = 1280;
      options.height = 720;
      options.min_score = 0.3F;
      options.max_detections = 24;
      options.mask_threshold = 0.08F;
      options.mask_grid_size = grid_size;
      const std::string payload =
          rfdetr::segmentation_json(boxes, logits, masks, mask_size, coco_labels(), options);
      const auto segments = nlohmann::json::parse(payload)["segments"];
      bool valid = segments.size() == 1U && segments[0]["label"] == "person" &&
                   segments[0]["mask_format"] == "polygon" &&
                   segments[0]["bbox"] == nlohmann::json({320, 180, 640, 360}) &&
                   segments[0]["mask"].size() >= 3U && payload.size() <= 32'768U;
      for (const auto& point : valid ? segments[0]["mask"] : nlohmann::json::array()) {
        valid = valid && point[0].get<int>() >= 320 && point[0].get<int>() < 960 &&
                point[1].get<int>() >= 180 && point[1].get<int>() < 540;
      }
      ok &= expect_true(valid, "segmentation traces one polygon inside its box for " +
                                   std::to_string(queries) + " queries, " +
                                   std::to_string(mask_size) + " masks, grid " +
                                   std::to_string(grid_size));
    }
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= test_shipped_config_names_a_supported_model_stream_and_labels();
  ok &= test_top_proposals_are_stable_and_best_first();
  ok &= test_detection_skips_unused_classes_and_scales_to_the_frame();
  ok &= test_segmentation_traces_a_polygon_inside_the_box();
  return ok ? 0 : 1;
}
