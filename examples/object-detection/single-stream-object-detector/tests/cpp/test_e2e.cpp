#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/source_cases.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "single-stream-object-detector";
constexpr int kFrames = 100;
// Above the shipped 0.30, so the scores show that the example applies inference.min_score.
constexpr double kMinScore = 0.5;
// The USB case asks the camera for this MJPEG mode.
constexpr int kUsbWidth = 1280;
constexpr int kUsbHeight = 720;

// The test stream's frame size, which the config must state, read over TCP as the example
// reads RTSP; zero when it cannot be read.
std::pair<int, int> frame_size(const std::string& url) {
  ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
  cv::VideoCapture capture(url, cv::CAP_FFMPEG);
  const auto width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const auto height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  return {width, height};
}

// The labels of the label file, one per line.
std::vector<std::string> label_file() {
  std::ifstream input(example_common_config_path(kExampleName).parent_path() / "coco_label.txt");
  std::vector<std::string> labels;
  for (std::string line; std::getline(input, line);) {
    labels.push_back(line);
  }
  return labels;
}

// Every box carries a label from the label file and a score of at least kMinScore.
bool valid_detections(const std::vector<MetadataJsonMessage>& messages,
                      const std::vector<std::string>& labels) {
  for (const auto& message : messages) {
    // A named value: a range-for over a member of a temporary would outlive the temporary.
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& entry : payload.at("data").at("objects")) {
      const auto label = entry.at("label").get<std::string>();
      if (std::find(labels.begin(), labels.end(), label) == labels.end() ||
          entry.at("confidence").get<double>() < kMinScore) {
        std::cerr << "[FAIL] unexpected detection: " << entry.dump() << "\n";
        return false;
      }
    }
  }
  return true;
}

int run_source_case(const std::string& binary, const std::string& model_path,
                    const StreamSourceCase& source_case, const std::string& source,
                    const std::string& label) {
  const bool usb = source_case.type == "usb";
  const auto [width, height] = usb ? std::pair{kUsbWidth, kUsbHeight} : frame_size(source);
  if (width <= 0 || height <= 0) {
    std::cerr << "[FAIL] " << label << " cannot read the frame size of " << source << "\n";
    return 1;
  }
  const std::string output_dir = create_test_output_dir(kExampleName, "test_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  // This test is the Insight receiver, so the example publishes to loopback.
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  // Only the keys of the case's input type, as the example reads them.
  std::ofstream config(config_path);
  config << "model:\n  path: " << model_path << "\n  labels: "
         << fs::absolute(example_common_config_path(kExampleName).parent_path() / "coco_label.txt")
                .string()
         << "\ninput:\n  type: " << source_case.type << "\n  source: " << source
         << "\n  width: " << width << "\n  height: " << height << "\n  fps: " << source_case.fps
         << "\n";
  if (source_case.type == "rtsp") {
    config << "  codec: " << source_case.codec << "\n  rtsp:\n    tcp: true\n    latency_ms: 100\n";
  }
  config << "inference:\n  frames: " << kFrames << "\n  min_score: " << kMinScore
         << "\n  nms_iou: 0.60\n  max_detections: 50\noutput:\n  insight:\n    host: 127.0.0.1"
         << "\n    video_port_base: " << video_port << "\n    metadata_port_base: " << metadata_port
         << "\n";
  if (source_case.type == "http") {
    config << "app:\n  ssl_strict: " << (source_case.ssl_strict ? "true" : "false") << "\n";
  }
  config.close();

  MetadataJsonListenerOptions options;
  options.base_port = metadata_port;
  options.timeout_ms = 5000;
  options.require_all_ports = true;
  options.min_object_count = 1;
  MetadataJsonListener listener(options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
    return 1;
  }
  const int video_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in video_address{};
  video_address.sin_family = AF_INET;
  video_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  video_address.sin_port = htons(static_cast<uint16_t>(video_port));
  if (video_socket < 0 || ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address),
                                 sizeof(video_address)) != 0) {
    if (video_socket >= 0) {
      ::close(video_socket);
    }
    std::cerr << "[FAIL] could not bind Insight video port\n";
    return 1;
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);
  std::array<unsigned char, 65536> video_packet{};
  const auto video_bytes =
      ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
  ::close(video_socket);
  remove_dir(output_dir);

  if (result.exit_code != 0) {
    std::cerr << "[FAIL] " << label << " exited with " << result.exit_code << "\nstdout:\n"
              << result.stdout_text << "\nstderr:\n"
              << result.stderr_text << "\n";
    return 1;
  }
  const auto metadata = listener.wait_for_messages();
  const auto following = listener.wait_for_messages();
  // The rest of the run's messages, so the checks below see every frame it sent.
  const auto rest = listener.drain_pending();
  if (!metadata.success || !following.success) {
    std::cerr << "[FAIL] " << label << " sent no metadata with a detection: " << metadata.error
              << following.error << "\n";
    return 1;
  }
  const auto& first = metadata.messages.back();
  const auto& last = following.messages.back();
  if (std::stoll(last.frame_id) <= std::stoll(first.frame_id) ||
      !valid_detections(metadata.messages, label_file()) ||
      !valid_detections(following.messages, label_file()) ||
      !valid_detections(rest.messages, label_file()) ||
      result.stdout_text.find("processed=" + std::to_string(kFrames) + " ") == std::string::npos) {
    std::cerr << "[FAIL] " << label << " did not send " << kFrames
              << " frames of valid, progressing detections\nstdout:\n"
              << result.stdout_text << "\n";
    return 1;
  }
  // H.265 goes to Insight as received (RTP payload type 98), and so does a USB camera's JPEG
  // (26). H.264 is passed through and decoded MJPEG and HTTP frames are encoded as H.264 (96).
  const int payload_type = usb ? 26 : source_case.codec == "h265" ? 98 : 96;
  if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 ||
      (video_packet[1] & 0x7F) != payload_type) {
    std::cerr << "[FAIL] " << label << " did not publish RTP video with payload type "
              << payload_type << " to Insight\n";
    return 1;
  }
  std::cout << "[OK] " << label << " sent " << kFrames
            << " frames of detection metadata and Insight video\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  // Absolute, because the example resolves a relative model.path from its config's folder.
  const std::string models_dir =
      fs::absolute(models_dir_raw != nullptr ? models_dir_raw : "models").string();
  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("configured single-stream detector model not found under "
                        "SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const auto run_case = [&](const StreamSourceCase& source_case, const std::string& source) {
    int case_rc = 0;
    for (const std::string& model_path : model_paths) {
      const std::string label = model_case_label(source_case.name, model_path, model_paths.size());
      if (run_source_case(binary, model_path, source_case, source, label) != 0) {
        case_rc = 1;
      }
    }
    return case_rc;
  };
  // HTTP MJPEG reports no frame rate, and the test server uses a self-signed certificate.
  const int cases_rc = run_single_stream_source_cases(
      "single-stream object detector",
      {
          {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
          {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
          {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 0, true},
          {"http_mjpeg", "SIMANEAT_TEST_HTTP_MJPEG_URL", "http", "mjpeg", 30, false},
      },
      run_case);
  // USB capture needs a camera on the DevKit, which CI runners do not have, so this case runs
  // only when SIMANEAT_TEST_USB_CAMERA names one (see usb_camera_ids()), even under --strict.
  int usb_rc = 0;
  bool usb_ran = false;
  if (const char* camera = env_or_null("SIMANEAT_TEST_USB_CAMERA")) {
    usb_ran = true;
    usb_rc = run_case({"usb_camera", "SIMANEAT_TEST_USB_CAMERA", "usb", "mjpeg", 30, true}, camera);
  } else {
    std::cerr << "[SKIP] set SIMANEAT_TEST_USB_CAMERA to a USB camera ID to run usb_camera e2e\n";
  }
  if (rc != 0 || usb_rc != 0) {
    return 1;
  }
  // A camera-only run on a DevKit is a pass, not "no stream URLs configured".
  return usb_ran && cases_rc == 77 ? 0 : cases_rc;
}
