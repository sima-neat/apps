// Unit tests for the SuperPoint feature extractor: the checks the application leaves to its config
// and the mapping of feature points onto the source frame. tests/python/test_unit.py runs the same
// checks in Python, plus the frames luma rejects, since C++ luma is in main.cpp, which this test
// does not compile, and the pull loop's stop at end of stream, since C++ gets PullStatus::Closed.

#include "../../src/cpp/utils/feature_points.h"
#include "sima_examples/config.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

const superpoint::FrameSize kModelSize{640, 480};
const superpoint::FrameSize kSourceSize{1280, 720};

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_the_model_and_usable_values() {
  const fs::path common_dir = "examples/feature-extraction/superpoint-feature-extractor/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const fs::path model_path = common_dir / config.string("model.path");
  bool ok = expect_true(model_path.filename() == "superpoint_w8a8_mpk.tar.gz",
                        "model.path names the SuperPoint w8a8 package");
  // The packaged path reaches prebuilt-apps/models/superpoint/w8a8/ from the config file.
  ok &= expect_true(fs::absolute(model_path).lexically_normal() ==
                        fs::absolute("models/superpoint/w8a8/superpoint_w8a8_mpk.tar.gz"),
                    "model.path reaches models/superpoint/w8a8/ from the config file");
  ok &= expect_true(config.string("input.source").rfind("rtsp://", 0) == 0,
                    "input.source is an RTSP URL");
  for (const std::string key :
       {"input.width", "input.height", "input.fps", "input.rtsp.latency_ms", "inference.frames"}) {
    ok &= expect_true(config.integer(key) >= 0, key + " is not negative");
  }
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

bool test_source_points_scale_each_axis_onto_the_source_frame() {
  const auto points =
      superpoint::source_points({600.0F, 400.0F, 10.0F, 20.0F}, kModelSize, kSourceSize);
  return expect_true(points.size() == 2U && points[0].x == 1200 && points[0].y == 600 &&
                         points[1].x == 20 && points[1].y == 30,
                     "source_points scales each axis onto the source frame");
}

bool test_source_points_round_half_up_to_whole_pixels() {
  const auto points = superpoint::source_points({6.25F, 3.0F, 6.0F, 3.3F}, kModelSize, kSourceSize);
  return expect_true(points.size() == 2U && points[0].x == 13 && points[0].y == 5 &&
                         points[1].x == 12 && points[1].y == 5,
                     "source_points rounds half up to whole pixels");
}

bool test_source_points_drop_points_on_or_beyond_an_edge() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // 639.8 maps to 1279.6, which rounds onto the right edge.
  const auto points = superpoint::source_points(
      {600.0F, 400.0F, 640.0F, 200.0F, 639.8F, 200.0F, 20.0F, 480.0F, -1.0F, 5.0F, nan, 10.0F},
      kModelSize, kSourceSize);
  return expect_true(points.size() == 1U && points[0].x == 1200 && points[0].y == 600,
                     "source_points drops points on or beyond an edge");
}

bool test_source_points_of_a_frame_without_features_are_empty() {
  return expect_true(superpoint::source_points({}, kModelSize, kSourceSize).empty(),
                     "a frame without features has no source points");
}

// Insight draws each keypoint as a dot; without names and confidences it draws no skeleton line
// and no box.
bool test_points_json_sends_the_points_as_one_pose_of_unnamed_keypoints() {
  const auto data = nlohmann::json::parse(superpoint::points_json({{13, 7}, {1279, 0}}));
  const auto expected =
      nlohmann::json::parse(R"({"poses":[{"keypoints":[{"x":13,"y":7},{"x":1279,"y":0}]}]})");
  return expect_true(data == expected,
                     "points_json sends the points as one pose of unnamed keypoints: " +
                         data.dump());
}

bool test_points_json_of_a_frame_without_features_has_no_keypoints() {
  const auto data = nlohmann::json::parse(superpoint::points_json({}));
  return expect_true(data == nlohmann::json::parse(R"({"poses":[{"keypoints":[]}]})"),
                     "a frame without features sends a pose without keypoints: " + data.dump());
}

// The metadata sender rejects messages over 65,507 bytes; the decoder returns at most 600 points
// per frame.
bool test_points_json_of_600_points_fits_one_metadata_message() {
  const std::vector<superpoint::Point> points(600, {1279, 1279});
  return expect_true(superpoint::points_json(points).size() < 65'507U - 1'024U,
                     "600 points fit one metadata message with room for its envelope");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_the_model_and_usable_values();
  ok &= test_source_points_scale_each_axis_onto_the_source_frame();
  ok &= test_source_points_round_half_up_to_whole_pixels();
  ok &= test_source_points_drop_points_on_or_beyond_an_edge();
  ok &= test_source_points_of_a_frame_without_features_are_empty();
  ok &= test_points_json_sends_the_points_as_one_pose_of_unnamed_keypoints();
  ok &= test_points_json_of_a_frame_without_features_has_no_keypoints();
  ok &= test_points_json_of_600_points_fits_one_metadata_message();
  return ok ? 0 : 1;
}
