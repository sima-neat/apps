// Unit tests for the single-stream object detector: the checks the application leaves to its
// config. tests/python/test_unit.py runs the same checks against main.py's tables.

#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <filesystem>
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

} // namespace

int main() {
  return test_shipped_config_names_a_supported_input_model_and_labels() ? 0 : 1;
}
