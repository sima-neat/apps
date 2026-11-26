// Unit tests for the FastFlow anomaly detector: the checks the application leaves to its config,
// and the defect regions it sends. tests/python/test_unit.py runs the same checks in Python.

#include "../../src/cpp/utils/anomaly_regions.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_and_input() {
  const fs::path common_dir = "examples/anomaly-detection/fastflow-anomaly-detector/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const std::string model_id = config.string("model.id");
  const std::string codec = config.string("input.codec");
  bool ok = expect_true(model_id == "fastflow_demo" || model_id == "anomalib",
                        "the shipped model.id is supported");
  ok &= expect_true(codec == "h264" || codec == "h265" || codec == "mjpeg",
                    "the shipped codec is supported");
  ok &= expect_true(fs::path(config.string("model.path")).filename() == "fastflow_demo_mpk.tar.gz",
                    "model.path names the fastflow_demo package");
  ok &= expect_true(
      config.integer("input.width") >= 0 && config.integer("input.height") >= 0 &&
          config.integer("input.fps") >= 0 && config.integer("input.rtsp.latency_ms") >= 0 &&
          config.integer("inference.frames") >= 0 &&
          config.integer("inference.max_detections") > 0 && config.integer("app.min_region_px") > 0,
      "the shipped sizes, latency, and counts are in range");
  ok &= expect_true(config.number("app.anomaly_threshold") > 0.0 &&
                        config.number("app.anomaly_threshold") < 1.0,
                    "the shipped anomaly threshold is between 0 and 1");
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

// A 16x16 map with a 16-pixel region scoring 0.9, a 9-pixel one peaking at 0.7, and a 4-pixel
// one at 0.8.
cv::Mat three_region_map() {
  cv::Mat map = cv::Mat::zeros(16, 16, CV_32FC1);
  map(cv::Rect(2, 2, 4, 4)).setTo(0.9F);
  map(cv::Rect(10, 1, 3, 3)).setTo(0.6F);
  map.at<float>(2, 11) = 0.7F;
  map(cv::Rect(12, 12, 2, 2)).setTo(0.8F);
  return map;
}

// A 32x48 frame scales map x by 2 and map y by 3.
fastflow::RegionOptions options(int max_regions = 10) {
  return {
      .width = 32, .height = 48, .threshold = 0.5, .min_region_px = 5, .max_regions = max_regions};
}

const std::vector<std::array<int, 2>> kFirstOutline = {{4, 6}, {4, 15}, {10, 15}, {10, 6}};

// The 4-pixel region is dropped; the others come best first, scaled to the frame.
bool test_regions_below_min_region_px_are_speckle() {
  const auto regions = fastflow::defect_regions(three_region_map(), options());
  bool ok = expect_true(regions.size() == 2U, "two regions reach min_region_px");
  if (!ok) {
    return false;
  }
  ok &= expect_true(regions[0].bbox == std::array<int, 4>{4, 6, 8, 12} &&
                        regions[1].bbox == std::array<int, 4>{20, 3, 6, 9},
                    "the boxes are scaled to the frame, best first");
  ok &= expect_true(std::abs(regions[0].score - 0.9) < 1e-6 &&
                        std::abs(regions[1].score - 0.7) < 1e-6,
                    "each score is the region's highest probability");
  ok &= expect_true(regions[0].polygon == kFirstOutline, "the outline is scaled to the frame");
  return ok;
}

bool test_max_regions_keeps_the_highest_scores() {
  const auto regions = fastflow::defect_regions(three_region_map(), options(1));
  return expect_true(regions.size() == 1U && regions[0].bbox == std::array<int, 4>{4, 6, 8, 12},
                     "max_regions keeps the best region");
}

bool test_a_map_below_the_threshold_has_no_region() {
  const cv::Mat good_part(16, 16, CV_32FC1, cv::Scalar(0.49F));
  bool ok = expect_true(fastflow::defect_regions(good_part, options()).empty(),
                        "a good part has no region");
  ok &= expect_true(nlohmann::json::parse(fastflow::segments_json({})) ==
                        nlohmann::json{{"segments", nlohmann::json::array()}},
                    "a good part sends no segments");
  return ok;
}

// Insight draws a polygon only from three points, so a region approxPolyDP reduces to fewer
// gets its box.
bool test_a_dot_or_a_line_is_outlined_by_its_box() {
  cv::Mat map = cv::Mat::zeros(16, 16, CV_32FC1);
  map.at<float>(9, 7) = 0.9F;
  map(cv::Rect(2, 14, 5, 1)).setTo(0.8F);
  const fastflow::RegionOptions dot_options = {
      .width = 32, .height = 48, .threshold = 0.5, .min_region_px = 1, .max_regions = 10};
  const auto regions = fastflow::defect_regions(map, dot_options);
  if (!expect_true(regions.size() == 2U, "the dot and the line are regions")) {
    return false;
  }
  const std::vector<std::array<int, 2>> dot_outline = {{14, 27}, {14, 30}, {16, 30}, {16, 27}};
  const std::vector<std::array<int, 2>> line_outline = {{4, 42}, {4, 45}, {14, 45}, {14, 42}};
  return expect_true(regions[0].bbox == std::array<int, 4>{14, 27, 2, 3} &&
                         regions[1].bbox == std::array<int, 4>{4, 42, 10, 3} &&
                         regions[0].polygon == dot_outline && regions[1].polygon == line_outline,
                     "a dot and a line are outlined by their boxes");
}

bool test_segments_carry_the_box_outline_and_score() {
  const auto regions = fastflow::defect_regions(three_region_map(), options(1));
  const auto segments = nlohmann::json::parse(fastflow::segments_json(regions)).at("segments");
  if (!expect_true(segments.size() == 1U, "one segment per region")) {
    return false;
  }
  const auto& segment = segments.front();
  return expect_true(segment.at("label") == "anomaly" &&
                         segment.at("bbox") == nlohmann::json{4, 6, 8, 12} &&
                         segment.at("mask_format") == "polygon" &&
                         segment.at("mask") == nlohmann::json(kFirstOutline) &&
                         std::abs(segment.at("confidence").get<double>() - 0.9) < 1e-6,
                     "the segment carries the box, outline, and score");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_and_input();
  ok &= test_regions_below_min_region_px_are_speckle();
  ok &= test_max_regions_keeps_the_highest_scores();
  ok &= test_a_map_below_the_threshold_has_no_region();
  ok &= test_a_dot_or_a_line_is_outlined_by_its_box();
  ok &= test_segments_carry_the_box_outline_and_score();
  return ok ? 0 : 1;
}
