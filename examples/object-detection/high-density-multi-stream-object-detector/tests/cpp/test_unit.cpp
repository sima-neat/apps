#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

struct TestBox {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

bool test_serialize_builds_insight_payload() {
  const std::vector<TestBox> boxes{
      {10.9f, -2.0f, 35.9f, 20.8f, 0.75f, 0},
      {90.0f, 80.0f, 120.0f, 130.0f, 0.5f, 1},
      {105.0f, 10.0f, 115.0f, 18.0f, 0.25f, 99},
  };
  const std::vector<std::string> labels{"person", "bicycle\n\"quoted\""};
  sima_examples::detection_egress::FrameMetadata frame;
  frame.stream_index = 7;
  frame.stream_id = "stream7";
  frame.frame_id = 42;
  frame.pts_ns = 1'234'567'890;
  frame.dts_ns = 1'200'000'000;
  frame.duration_ns = 50'000'000;
  frame.input_seq = 88;
  frame.orig_input_seq = 77;
  frame.rtp_timestamp = 111'111;

  const auto parsed = nlohmann::json::parse(
      sima_examples::detection_egress::serialize(boxes, labels, 100, 100, frame));
  bool ok =
      expect_true(parsed.at("type") == "object-detection" && parsed.at("stream_id") == "stream7" &&
                      parsed.at("stream_index") == 7 && parsed.at("frame_id") == "42" &&
                      parsed.at("timestamp") == 1234 && parsed.at("pts_ns") == 1'234'567'890,
                  "metadata carries the frame envelope");
  ok &= expect_true(parsed.at("data").at("objects").size() == boxes.size(),
                    "metadata keeps every decoded box");
  ok &= expect_true(parsed.at("data").at("objects").at(0).at("id") == "obj_1" &&
                        parsed.at("data").at("objects").at(1).at("label") == labels.at(1),
                    "metadata preserves object IDs and escaped labels");
  ok &= expect_true(parsed.at("data").at("objects").at(1).at("bbox") ==
                        nlohmann::json({90.0f, 80.0f, 10.0f, 20.0f}),
                    "metadata clamps boxes to the frame edge");
  ok &= expect_true(parsed.at("data").at("objects").at(2).at("label") == "unknown",
                    "metadata preserves unknown-class labels");
  ok &= expect_true(parsed.at("rtp_timestamp") == *frame.rtp_timestamp,
                    "metadata preserves the source-aligned RTP timestamp");

  frame.frame_id = -1;
  frame.pts_ns = -1;
  frame.rtp_timestamp.reset();
  const std::string no_pts =
      sima_examples::detection_egress::serialize(boxes, labels, 100, 100, frame);
  const auto parsed_no_pts = nlohmann::json::parse(no_pts);
  ok &= expect_true(parsed_no_pts.at("timestamp") == -1 && parsed_no_pts.at("frame_id") == "",
                    "metadata marks a frame without PTS");
  ok &= expect_true(!parsed_no_pts.contains("rtp_timestamp"),
                    "metadata omits RTP timestamp when source PTS is absent");
  return ok;
}

// The fields frame_metadata reads from a pulled simaai::neat::Sample.
struct TestSample {
  std::string stream_id = "decoder";
  std::int64_t frame_id = 42;
  std::int64_t pts_ns = 1'000'000'000;
  std::int64_t dts_ns = 900'000'000;
  std::int64_t duration_ns = 33'333'333;
  std::int64_t input_seq = 7;
  std::int64_t orig_input_seq = 6;
};

bool test_frame_metadata_copies_the_sample_identity() {
  TestSample sample;
  const auto frame = sima_examples::detection_egress::frame_metadata(sample, 3);
  return expect_true(frame.stream_index == 3 && frame.stream_id == "decoder" &&
                         frame.frame_id == 42 && frame.pts_ns == 1'000'000'000 &&
                         frame.dts_ns == 900'000'000 && frame.duration_ns == 33'333'333 &&
                         frame.input_seq == 7 && frame.orig_input_seq == 6 && !frame.rtp_timestamp,
                     "frame_metadata copies the sample identity and sets no RTP timestamp");
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_profiles_load() {
  const fs::path common_dir =
      "examples/object-detection/high-density-multi-stream-object-detector/src/common";
  const std::vector<std::pair<std::string, int>> profiles{
      {"config.yaml", 20}, {"config-30x720p20fps.yaml", 30}, {"config-60x720p10fps.yaml", 60}};
  bool ok = true;
  for (const auto& [filename, streams] : profiles) {
    const auto config = sima_examples::Config::load(common_dir / filename);
    const std::string codec = config.string("input.codec");
    ok &= expect_true(static_cast<int>(config.string_list("input.source").size()) == streams,
                      filename + " lists every stream");
    ok &= expect_true(codec == "h264" || codec == "h265", filename + " names a supported codec");
    ok &= expect_true(config.integer("input.width") == 0 && config.integer("input.height") == 0,
                      filename + " reads each stream's frame size");
    ok &= expect_true(config.integer("inference.max_detections") > 0 &&
                          config.integer("app.queue_depth") > 0 &&
                          !config.string("output.insight.host").empty(),
                      filename + " sets usable inference and output values");
  }
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= test_serialize_builds_insight_payload();
  ok &= test_frame_metadata_copies_the_sample_identity();
  ok &= test_shipped_profiles_load();
  return ok ? 0 : 1;
}
