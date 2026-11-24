// Unit tests for the multi-stream OBB detector. tests/python/test_unit.py runs the same checks in
// Python.
#include "../../src/cpp/utils/oriented_boxes.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

const fs::path kExampleDir = "examples/object-detection/multi-stream-obb-detector";

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_input_and_labels() {
  const fs::path common_dir = kExampleDir / "src" / "common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const std::string codec = config.string("input.codec");
  bool ok = expect_true(config.string("model.id") == "yolo26", "the shipped model is YOLO26");
  const fs::path model_path = common_dir / config.string("model.path");
  ok &= expect_true(model_path.filename() == "yolo26n_obb_w8a8_mpk.tar.gz",
                    "model.path names the YOLO26n OBB package");
  // The packaged path reaches prebuilt-apps/models from the config file.
  ok &= expect_true((fs::absolute(common_dir) / config.string("model.path")).lexically_normal() ==
                        (fs::absolute("models") / "yolo26n_obb/w8a8/yolo26n_obb_w8a8_mpk.tar.gz")
                            .lexically_normal(),
                    "model.path starts at the config file and reaches models/");
  // The model's class IDs index the 15 DOTA labels.
  ok &= expect_true(sima_examples::load_labels(config.string("model.labels")).size() == 15,
                    "model.labels lists the 15 DOTA classes");
  ok &= expect_true(config.string_list("input.source").size() == 2, "the config lists two streams");
  ok &= expect_true(codec == "h264" || codec == "h265" || codec == "mjpeg",
                    "the shipped codec is supported");
  ok &= expect_true(config.integer("input.width") == 0 && config.integer("input.height") == 0,
                    "RTSP streams report their own frame size");
  ok &= expect_true(config.integer("input.fps") >= 0 &&
                        config.integer("input.rtsp.latency_ms") >= 0 &&
                        config.integer("inference.frames") >= 0,
                    "frame rate, latency, and frame limit are not negative");
  ok &= expect_true(config.integer("inference.max_detections") > 0 &&
                        config.number("inference.min_score") >= 0.0 &&
                        config.number("inference.min_score") <= 1.0,
                    "the detection limits are usable");
  for (const char* key : {"output.insight.video_port_base", "output.insight.metadata_port_base"}) {
    ok &= expect_true(config.integer(key) >= 1 && config.integer(key) <= 65535,
                      std::string(key) + " is a UDP port");
  }
  // Core sends to 127.0.0.1 when the host is empty.
  ok &=
      expect_true(!config.string("output.insight.host").empty() && config.boolean("input.rtsp.tcp"),
                  "the Insight host is set and RTSP uses TCP");
  return ok;
}

bool test_objects_match_the_shared_fixture() {
  std::ifstream input(kExampleDir / "tests" / "fixtures" / "insight_objects.json");
  if (!input.good()) {
    std::cerr << "[FAIL] tests/fixtures/insight_objects.json not found\n";
    return false;
  }
  const auto fixture = nlohmann::json::parse(input);
  std::vector<obb::OrientedBox> boxes;
  for (const auto& box : fixture.at("boxes")) {
    // decode_obb_tensor returns float32 values, as the Python test feeds them.
    const auto f = [&](const char* key) { return static_cast<double>(box.at(key).get<float>()); };
    boxes.push_back({f("cx"), f("cy"), f("width"), f("height"), f("angle_radians"), f("score"),
                     box.at("class_id").get<int>()});
  }
  const auto objects = nlohmann::json::parse(
      obb::detections_json(boxes, fixture.at("labels").get<std::vector<std::string>>()))["objects"];
  const bool ok = expect_true(objects == fixture.at("objects"),
                              "Insight objects match the shared fixture: " + objects.dump());
  if (ok) {
    std::cout << "[OK] Insight objects match the shared fixture\n";
  }
  return ok;
}

// Insight clears the previous frame's boxes only when it receives the empty list.
bool test_a_frame_without_detections_sends_an_empty_list() {
  return expect_true(nlohmann::json::parse(obb::detections_json({}, {"plane"})) ==
                         nlohmann::json{{"objects", nlohmann::json::array()}},
                     "a frame without detections sends an empty list");
}

} // namespace

int main() {
  bool ok = true;
  ok &= test_shipped_config_names_a_supported_model_input_and_labels();
  ok &= test_objects_match_the_shared_fixture();
  ok &= test_a_frame_without_detections_sends_an_empty_list();
  return ok ? 0 : 1;
}
