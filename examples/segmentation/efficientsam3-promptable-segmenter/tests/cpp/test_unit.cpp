// Unit tests for the EfficientSAM3 promptable segmenter: the checks the application leaves to its
// config, the CLIP tokenizer in utils/clip_tokenizer.h, and the host work in
// utils/efficientsam3.h. tests/python/test_unit.py runs the same checks in Python.

#include "../../src/cpp/utils/clip_tokenizer.h"
#include "../../src/cpp/utils/efficientsam3.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using efficientsam3::OverlayClock;
using efficientsam3::Segment;
using efficientsam3::SegmentOptions;
using sima_examples::testing::expect_true;

namespace {

const fs::path kCommonDir = "examples/segmentation/efficientsam3-promptable-segmenter/src/common";
constexpr int kTextTokens = 16;

// The Python unit test asserts the same token ids and the same metadata.
const std::vector<std::pair<std::string, std::vector<int64_t>>> kTokens = {
    {"person", {49406, 2533, 49407}},
    {"red car", {49406, 736, 1615, 49407}},
    {"it's a dog!", {49406, 585, 568, 320, 1929, 256, 49407}},
    {"café 2 cups", {49406, 15304, 273, 11463, 49407}},
    {"Traffic   Light", {49406, 3399, 1395, 49407}},
    {"a person wearing a yellow safety vest near the road",
     {49406, 320, 2533, 3309, 320, 4481, 3406, 12473, 2252, 518, 1759, 49407}},
};

const std::string kSegmentsJson =
    R"({"segments":[{"id":"seg_1","label":"red car","confidence":0.9123,"bbox":[390,276,420,180],)"
    R"("mask_format":"polygon","mask":[[400,284],[403,449],[799,447],[795,281]]},)"
    R"({"id":"seg_3","label":"red car","confidence":0.4001,"bbox":[1809,1071,111,9],)"
    R"("mask_format":"polygon","mask":[[1800,1063],[1800,1079],[1919,1079],[1919,1063]]}]})";

constexpr int kQueries = 200;
constexpr int kMaskSide = 192;

// Detections and mask logits of one 1920x1080 frame, as the model returns them.
struct FrameResult {
  std::vector<float> detections = std::vector<float>(kQueries * 6, 0.0F);
  std::vector<float> masks = std::vector<float>(kMaskSide * kMaskSide * kQueries, -10.0F);

  FrameResult() {
    detect(7, {205.0F, 258.0F, 425.0F, 425.0F, 0.91234F});
    fill(7, 50, 80, 40, 80);
    detect(3, {10, 10, 100, 100, 0.25F}); // below min_score
    fill(3, 10, 20, 10, 20);
    detect(12, {500, 500, 600, 600, 0.5F}); // empty mask: skipped, still numbered
    detect(30, {950.0F, 1000.0F, 1010.0F, 1015.0F, 0.40005F}); // box past the frame edge
    fill(30, 180, 192, 180, 192);
  }
  void detect(int query, std::array<float, 5> row) {
    std::copy(row.begin(), row.end(), detections.begin() + query * 6);
  }
  void fill(int query, int y0, int y1, int x0, int x1) {
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) {
        masks[(y * kMaskSide + x) * kQueries + query] = 10.0F;
      }
    }
  }
  std::vector<Segment> segments(const SegmentOptions& options) {
    const cv::Mat rows(kQueries, 6, CV_32F, detections.data());
    const cv::Mat planes(kMaskSide, kMaskSide, CV_32FC(kQueries), masks.data());
    return efficientsam3::segments_of(rows, planes, options);
  }
};

SegmentOptions options() {
  SegmentOptions options;
  options.width = 1920;
  options.height = 1080;
  options.label = "red car";
  options.min_score = 0.3;
  options.max_detections = 20;
  options.mask_threshold = 0.5;
  return options;
}

std::vector<std::string> ids(const std::vector<Segment>& segments) {
  std::vector<std::string> out;
  for (const Segment& segment : segments) {
    out.push_back(segment.id);
  }
  return out;
}

// A gray NV12 frame whose left half has luma `left_y` and right half `right_y`.
cv::Mat nv12_frame(int width, int height, int left_y, int right_y) {
  cv::Mat frame(height * 3 / 2, width, CV_8UC1, cv::Scalar(128));
  frame(cv::Rect(0, 0, width / 2, height)).setTo(left_y);
  frame(cv::Rect(width / 2, 0, width - width / 2, height)).setTo(right_y);
  return frame;
}

ClipTokenizer tokenizer() {
  return ClipTokenizer(kCommonDir / "bpe_simple_vocab_16e6.txt.gz");
}

bool rejects(const std::string& prompt, const std::string& message) {
  try {
    (void)prompt_tokens(tokenizer(), prompt, kTextTokens);
    return false;
  } catch (const std::exception& error) {
    return std::string(error.what()).find(message) != std::string::npos;
  }
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_the_model_and_usable_values() {
  const auto config = sima_examples::Config::load(kCommonDir / "config.yaml");
  const std::string codec = config.string("input.codec");
  bool ok =
      expect_true(fs::path(config.string("model.path")).filename() == "efficientsam3_mpk.tar.gz",
                  "model.path names the segmentation model package");
  ok &= expect_true(codec == "h264" || codec == "h265", "the shipped codec is supported");
  ok &= expect_true(config.string("input.source").starts_with("rtsp://"),
                    "input.source is an RTSP URL");
  ok &= expect_true(!rejects(config.string("app.prompt"), ""),
                    "the shipped prompt fits the text encoder");
  ok &= expect_true(config.integer("inference.max_detections") > 0 &&
                        config.number("inference.min_score") >= 0.0 &&
                        config.number("inference.min_score") <= 1.0,
                    "the shipped inference values are usable");
  // The threshold becomes a logit, so 0 and 1 are out of range.
  ok &= expect_true(config.number("inference.mask_threshold") > 0.0 &&
                        config.number("inference.mask_threshold") < 1.0,
                    "inference.mask_threshold is between 0 and 1");
  ok &= expect_true(!config.string("output.insight.host").empty() &&
                        config.integer("output.insight.video_port_base") > 0 &&
                        config.integer("output.insight.metadata_port_base") > 0,
                    "the shipped Insight values are usable");
  return ok;
}

bool test_tokenizer_matches_clip() {
  const ClipTokenizer clip = tokenizer();
  bool ok = true;
  for (const auto& [prompt, expected_ids] : kTokens) {
    std::vector<int64_t> expected = expected_ids;
    expected.resize(kTextTokens, 0);
    ok &= expect_true(clip.encode(prompt, kTextTokens) == expected, "tokens of '" + prompt + "'");
  }
  return ok;
}

bool test_tokenizer_truncates_and_keeps_the_end_token() {
  std::string twenty_cars;
  for (int i = 0; i < 20; ++i) {
    twenty_cars += "car ";
  }
  const std::vector<int64_t> tokens = tokenizer().encode(twenty_cars, kTextTokens);
  return expect_true(tokens.size() == kTextTokens && tokens.front() == 49406 &&
                         tokens.back() == 49407,
                     "a long text is cut and keeps the end token");
}

bool test_prompt_must_name_something() {
  return expect_true(rejects("", "app.prompt must name what to segment") &&
                         rejects("   ", "app.prompt must name what to segment"),
                     "an empty prompt is rejected");
}

bool test_prompt_longer_than_the_text_encoder_is_rejected() {
  std::string fourteen_cars;
  for (int i = 0; i < 14; ++i) {
    fourteen_cars += "car ";
  }
  return expect_true(prompt_tokens(tokenizer(), fourteen_cars, kTextTokens).at(15) == 49407 &&
                         rejects(fourteen_cars + "bus",
                                 "app.prompt is 15 tokens; the text encoder reads at most 14"),
                     "a prompt longer than the text encoder is rejected");
}

// Black on the left (luma 16) and white on the right (luma 235) become -1 and 1, at the model's
// size, whether the frame shrinks or grows to it.
bool test_preprocess_makes_rgb_from_minus_one_to_one() {
  bool ok = true;
  for (const auto& [width, height] : {std::pair{1920, 1080}, std::pair{640, 480}}) {
    cv::Mat out;
    efficientsam3::preprocess(nv12_frame(width, height, 16, 235), out);
    const int half = efficientsam3::kModelSize / 2;
    double left_min = 0.0, left_max = 0.0, right_min = 0.0, right_max = 0.0;
    cv::minMaxLoc(out.colRange(0, half - 8).reshape(1), &left_min, &left_max);
    cv::minMaxLoc(out.colRange(half + 8, efficientsam3::kModelSize).reshape(1), &right_min,
                  &right_max);
    ok &= expect_true(out.rows == efficientsam3::kModelSize &&
                          out.cols == efficientsam3::kModelSize && out.type() == CV_32FC3 &&
                          left_min > -1.0 - 1e-6 && left_max < -1.0 + 1e-6 &&
                          right_min > 1.0 - 1e-6 && right_max < 1.0 + 1e-6,
                      "preprocess makes -1..1 RGB from a " + std::to_string(width) + "x" +
                          std::to_string(height) + " frame");
  }
  return ok;
}

bool test_segments_and_metadata_of_one_frame() {
  FrameResult result;
  return expect_true(efficientsam3::segments_json(result.segments(options())) == kSegmentsJson,
                     "segments and metadata of one frame match the Python test");
}

bool test_metadata_keeps_the_best_segments_within_the_byte_budget() {
  FrameResult result;
  SegmentOptions long_label = options();
  long_label.label = std::string(20000, 'x');
  const std::string payload = efficientsam3::segments_json(result.segments(long_label));
  const auto kept = nlohmann::json::parse(payload).at("segments");
  return expect_true(payload.size() <= efficientsam3::kMetadataByteBudget && kept.size() == 1 &&
                         kept.at(0).at("id") == "seg_1",
                     "metadata keeps the best segments within the byte budget");
}

bool test_max_detections_keeps_the_best_scores() {
  FrameResult result;
  SegmentOptions one = options();
  one.max_detections = 1;
  return expect_true(ids(result.segments(one)) == std::vector<std::string>{"seg_1"},
                     "max_detections keeps the best scores");
}

bool test_min_score_drops_lower_scores() {
  FrameResult result;
  SegmentOptions strict = options();
  strict.min_score = 0.45;
  return expect_true(ids(result.segments(strict)) == std::vector<std::string>{"seg_1"},
                     "min_score drops lower scores");
}

bool test_duplicate_instances_are_dropped() {
  FrameResult duplicated;
  duplicated.detect(40, {210.0F, 262.0F, 420.0F, 400.0F, 0.45F}); // most of instance 7's box
  duplicated.fill(40, 50, 75, 40, 80);
  const std::vector<std::string> expected = {"seg_1", "seg_4"};
  SegmentOptions two = options();
  two.max_detections = 2;
  return expect_true(ids(duplicated.segments(options())) == expected,
                     "duplicate instances are dropped") &&
         expect_true(ids(duplicated.segments(two)) == expected,
                     "empty and duplicate instances do not use up max_detections");
}

bool test_empty_mask_has_no_outline() {
  const cv::Mat empty(kMaskSide, kMaskSide, CV_32F, cv::Scalar(-10.0F));
  return expect_true(
      efficientsam3::mask_outline(empty, {100, 100, 300, 300}, 1920, 1080, 0.0F).empty(),
      "an empty mask has no outline");
}

bool same(const std::vector<OverlayClock::Message>& messages,
          const std::vector<std::tuple<std::string, int64_t, std::string>>& expected) {
  if (messages.size() != expected.size()) {
    return false;
  }
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const auto& [data, timestamp_ms, frame_id] = expected[i];
    if (messages[i].data != data || messages[i].timestamp_ms != timestamp_ms ||
        messages[i].frame_id != frame_id) {
      return false;
    }
  }
  return true;
}

bool test_overlay_clock_sends_every_frame_the_nearest_result() {
  OverlayClock clock;
  for (const auto& [pts_ms, frame_id] :
       {std::pair{0, "1"}, std::pair{33, "2"}, std::pair{66, "3"}, std::pair{100, "4"}}) {
    clock.add_frame(pts_ms, frame_id);
  }
  const bool first = same(clock.add_result(33, "A"), {{"A", 0, "1"}, {"A", 33, "2"}});
  clock.add_frame(133, "5");
  clock.add_frame(166, "6");
  // Frame 66 is nearer the result at 33, frame 100 nearer the one at 133; frame 166 waits.
  const bool second =
      same(clock.add_result(133, "B"), {{"A", 66, "3"}, {"B", 100, "4"}, {"B", 133, "5"}});
  return expect_true(first && second, "every frame is sent the nearest result");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_the_model_and_usable_values();
  ok &= test_tokenizer_matches_clip();
  ok &= test_tokenizer_truncates_and_keeps_the_end_token();
  ok &= test_prompt_must_name_something();
  ok &= test_prompt_longer_than_the_text_encoder_is_rejected();
  ok &= test_preprocess_makes_rgb_from_minus_one_to_one();
  ok &= test_segments_and_metadata_of_one_frame();
  ok &= test_metadata_keeps_the_best_segments_within_the_byte_budget();
  ok &= test_max_detections_keeps_the_best_scores();
  ok &= test_min_score_drops_lower_scores();
  ok &= test_duplicate_instances_are_dropped();
  ok &= test_empty_mask_has_no_outline();
  ok &= test_overlay_clock_sends_every_frame_the_nearest_result();
  return ok ? 0 : 1;
}
