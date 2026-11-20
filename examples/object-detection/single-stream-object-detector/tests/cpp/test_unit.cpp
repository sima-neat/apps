// Unit tests for the single-stream object detector: the shipped config and each input type's
// documented keys load. tests/python/test_unit.py runs the same cases against main.py.

#include "sima_examples/testing/test_checks.h"
#include "sima_examples/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_scratch_config;

namespace {

constexpr const char* kExampleName = "single-stream-object-detector";

using Lines = std::vector<std::pair<std::string, std::string>>;

// The input keys each input type reads, with test values, as YAML lines under `input:`.
Lines input_lines(const std::string& type) {
  if (type == "rtsp") {
    return {{"type", "rtsp"},
            {"source", "rtsp://127.0.0.1:8554/src1"},
            {"codec", "h264"},
            {"width", "0"},
            {"height", "0"},
            {"fps", "0"},
            {"rtsp", "\n    tcp: true\n    latency_ms: 100"}};
  }
  if (type == "http") {
    return {{"type", "http"},
            {"source", "http://127.0.0.1:8080/stream.mjpg"},
            {"width", "1280"},
            {"height", "720"},
            {"fps", "30"}};
  }
  return {{"type", "usb"},
          {"source", "usb-0000:00:14.0-1"},
          {"width", "1280"},
          {"height", "720"},
          {"fps", "30"}};
}

struct Rejected {
  std::string key; // input.type, input.codec, output.insight.host, or inference.frames.
  std::string value;
  std::string message;
};

// The values the example rejects before it runs, the same as tests/python/test_unit.py.
const std::vector<Rejected> kRejected = {
    {"input.type", "rtp", "input.type must be rtsp, http, or usb"},
    {"input.codec", "vp8", "input.codec must be h264, h265, or mjpeg"},
    {"output.insight.host", "\"\"", "output.insight.host must not be empty"},
    {"inference.frames", "-1", "inference.frames must be 0 or more"},
};

// A config of `type` with the keys the config comments list, and `change` applied.
std::string config_body(const std::string& type, const Rejected* change = nullptr) {
  const auto value_of = [change](const std::string& key, const std::string& value) {
    return change != nullptr && change->key == key ? change->value : value;
  };
  std::string body = "model:\n  path: models/yolo26m_w16a16_mpk.tar.gz\n  labels: coco_label.txt\n"
                     "input:\n";
  for (const auto& [key, value] : input_lines(type)) {
    body += "  " + key + ": " + value_of("input." + key, value) + "\n";
  }
  body += "inference:\n  frames: " + value_of("inference.frames", "0") +
          "\n  min_score: 0.3\n  nms_iou: 0.6\n  max_detections: 50\n"
          "output:\n  insight:\n    host: " +
          value_of("output.insight.host", "127.0.0.1") +
          "\n    video_port_base: 9000\n"
          "    metadata_port_base: 9100\n";
  if (type == "http") {
    body += "app:\n  ssl_strict: false\n";
  }
  return body;
}

sima_examples::testing::ProcessResult validate(const std::string& binary, const std::string& name,
                                               const std::string& body) {
  const fs::path config_path = write_scratch_config(kExampleName, name, body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  return result;
}

bool test_shipped_config_loads(const std::string& binary) {
  // A scratch copy, so the process logs the harness writes next to the config stay out of
  // src/common.
  std::ifstream shipped("examples/object-detection/single-stream-object-detector/src/common/"
                        "config.yaml");
  const std::string body{std::istreambuf_iterator<char>(shipped), {}};
  const auto result = validate(binary, "shipped_config", body);
  return expect_true(!body.empty() && result.exit_code == 0, "config.yaml loads") &&
         expect_contains(result.stdout_text, "input=rtsp, codec=h264, 0x0, fps=0",
                         "config.yaml reads its values");
}

bool test_each_input_type_loads_with_its_documented_keys(const std::string& binary) {
  bool ok = true;
  for (const std::string type : {"rtsp", "http", "usb"}) {
    const auto result = validate(binary, "own_keys_" + type, config_body(type));
    ok &= expect_true(result.exit_code == 0, type + " config loads") &&
          expect_contains(result.stdout_text, "input=" + type, type + " config reads as " + type);
  }
  return ok;
}

bool test_rejected_values_fail_with_their_message(const std::string& binary) {
  bool ok = true;
  for (const auto& rejected : kRejected) {
    const auto result =
        validate(binary, "rejected_" + rejected.key, config_body("rtsp", &rejected));
    ok &= expect_true(result.exit_code != 0, rejected.key + " " + rejected.value + " fails") &&
          expect_contains(result.stderr_text, rejected.message, rejected.message);
  }
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_shipped_config_loads(binary);
  ok &= test_each_input_type_loads_with_its_documented_keys(binary);
  ok &= test_rejected_values_fail_with_their_message(binary);
  return ok ? 0 : 1;
}
