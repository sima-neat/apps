// Unit tests for the single-stream object detector: the checks the application leaves to its
// config. tests/python/test_unit.py runs the same checks against main.py's tables. This example
// also holds the test of the shared C++ Config::choice, since support/ has no C++ test program.

#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_input_model_and_labels() {
  const fs::path common_dir = "examples/object-detection/single-stream-object-detector/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const std::string type = config.string("input.type");
  const std::string codec = config.string("input.codec");
  bool ok = expect_true(type == "rtsp" || type == "http" || type == "usb",
                        "the shipped input type is supported");
  ok &= expect_true(codec == "h264" || codec == "h265" || codec == "mjpeg",
                    "the shipped codec is supported");
  ok &= expect_true(fs::path(config.string("model.path")).filename() == "yolo26m_w16a16_mpk.tar.gz",
                    "model.path names the YOLO26m package");
  // YOLO26's class IDs index the 80 COCO labels.
  ok &= expect_true(sima_examples::load_labels(config.string("model.labels")).size() == 80U,
                    "model.labels lists the 80 COCO classes");
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

// An unknown table value names the key, the value, and the choices, as in Python.
bool test_choice_names_the_key_and_the_value() {
  const fs::path path = fs::temp_directory_path() / "single-stream-choice-test.yaml";
  std::ofstream(path) << "input:\n  codec: h265\n  type: vp8\n";
  const auto config = sima_examples::Config::load(path);
  const std::map<std::string, int> codecs = {{"h264", 1}, {"h265", 2}, {"mjpeg", 3}};
  bool ok = expect_true(config.choice("input.codec", codecs) == 2, "choice returns the entry");
  std::string error;
  try {
    (void)config.choice("input.type", codecs);
  } catch (const std::runtime_error& thrown) {
    error = thrown.what();
  }
  fs::remove(path);
  ok &= expect_true(error == "input.type: unknown value 'vp8' (expected h264, h265, mjpeg)",
                    "an unknown value names the key, the value, and the choices: " + error);
  return ok;
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_input_model_and_labels();
  ok &= test_choice_names_the_key_and_the_value();
  return ok ? 0 : 1;
}
