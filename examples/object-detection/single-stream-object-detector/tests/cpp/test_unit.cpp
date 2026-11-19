// Unit tests for the single-stream object detector's own config rules: input.type selects the
// input, and the example reads only the keys that type takes, each with the codecs and frame
// sizes it can carry. tests/python/test_unit.py runs the same cases against main.py.

#include "sima_examples/testing/test_checks.h"
#include "sima_examples/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
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
            {"width", "1280"},
            {"height", "720"},
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

// A config of `type`. `inputs` replaces or adds input keys; `app` is the app: section, which an
// http config has by default.
std::string config_body(const std::string& type, const std::map<std::string, std::string>& inputs,
                        const std::string& app = "") {
  std::string body = "model:\n  path: models/yolo26m_w16a16_mpk.tar.gz\n  labels: coco_label.txt\n"
                     "input:\n";
  auto values = inputs;
  for (const auto& [key, value] : input_lines(type)) {
    const auto replaced = values.find(key);
    body += "  " + key + ": " + (replaced == values.end() ? value : replaced->second) + "\n";
    if (replaced != values.end()) {
      values.erase(replaced);
    }
  }
  for (const auto& [key, value] : values) {
    body += "  " + key + ": " + value + "\n";
  }
  body += "inference:\n  frames: 0\n  min_score: 0.3\n  nms_iou: 0.6\n  max_detections: 50\n"
          "output:\n  insight:\n    host: 127.0.0.1\n    video_port_base: 9000\n"
          "    metadata_port_base: 9100\n";
  const std::string app_section = app.empty() && type == "http" ? "ssl_strict: false" : app;
  if (!app_section.empty()) {
    body += "app:\n  " + app_section + "\n";
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

bool expect_rejected(const std::string& binary, const std::string& name, const std::string& body,
                     const std::string& message) {
  const auto result = validate(binary, name, body);
  return expect_true(result.exit_code == 1, name + " is rejected") &&
         expect_contains(result.stderr_text, message, name + " names the cause");
}

bool test_shipped_config_loads(const std::string& binary) {
  // A scratch copy, so the process logs the harness writes next to the config stay out of
  // src/common.
  std::ifstream shipped("examples/object-detection/single-stream-object-detector/src/common/"
                        "config.yaml");
  const std::string body{std::istreambuf_iterator<char>(shipped), {}};
  const auto result = validate(binary, "shipped_config", body);
  return expect_true(!body.empty() && result.exit_code == 0, "config.yaml loads") &&
         expect_contains(result.stdout_text, "source=rtsp", "config.yaml reads an RTSP source");
}

bool test_each_input_type_reads_its_own_keys(const std::string& binary) {
  bool ok = true;
  for (const std::string type : {"rtsp", "http", "usb"}) {
    const auto result = validate(binary, "own_keys_" + type, config_body(type, {}));
    ok &= expect_true(result.exit_code == 0, type + " config loads") &&
          expect_contains(result.stdout_text, "source=" + type, type + " config reads as " + type);
  }
  return ok;
}

bool test_a_key_of_another_input_type_is_rejected(const std::string& binary) {
  bool ok = true;
  ok &= expect_rejected(binary, "http_with_codec", config_body("http", {{"codec", "mjpeg"}}),
                        "unknown keys: input.codec");
  ok &= expect_rejected(binary, "usb_with_rtsp", config_body("usb", {{"rtsp", "\n    tcp: true"}}),
                        "unknown keys: input.rtsp.tcp");
  ok &= expect_rejected(binary, "rtsp_with_ssl_strict", config_body("rtsp", {}, "ssl_strict: true"),
                        "unknown keys: app.ssl_strict");
  return ok;
}

bool test_unknown_input_type_is_rejected(const std::string& binary) {
  return expect_rejected(binary, "unknown_input_type", config_body("rtsp", {{"type", "rtsps"}}),
                         "input.type must be rtsp, http, or usb");
}

bool test_codec_must_be_h264_h265_or_mjpeg(const std::string& binary) {
  return expect_rejected(binary, "unsupported_codec", config_body("rtsp", {{"codec", "vp9"}}),
                         "input.codec must be h264, h265, or mjpeg");
}

bool test_http_and_usb_sources_need_a_frame_rate(const std::string& binary) {
  bool ok = true;
  for (const std::string type : {"http", "usb"}) {
    ok &= expect_rejected(binary, "no_fps_" + type, config_body(type, {{"fps", "0"}}),
                          "input.fps must be above 0 for an HTTP or USB source");
  }
  return ok;
}

bool test_usb_frame_size_must_fit_rtp_jpeg(const std::string& binary) {
  bool ok = true;
  for (const auto& [width, height] :
       std::vector<std::pair<std::string, std::string>>{{"1284", "720"}, {"3840", "2160"}}) {
    ok &= expect_rejected(binary, "usb_size_" + width,
                          config_body("usb", {{"width", width}, {"height", height}}),
                          "multiples of 8 and at most 2040");
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
  ok &= test_each_input_type_reads_its_own_keys(binary);
  ok &= test_a_key_of_another_input_type_is_rejected(binary);
  ok &= test_unknown_input_type_is_rejected(binary);
  ok &= test_codec_must_be_h264_h265_or_mjpeg(binary);
  ok &= test_http_and_usb_sources_need_a_frame_rate(binary);
  ok &= test_usb_frame_size_must_fit_rtp_jpeg(binary);
  return ok ? 0 : 1;
}
