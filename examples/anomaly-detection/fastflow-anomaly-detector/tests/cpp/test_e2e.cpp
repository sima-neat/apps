// End-to-end tests: each RTSP codec reaches Insight as video and defect-region metadata.
// tests/python/test_e2e.py runs the same cases against main.py.

#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/source_cases.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "fastflow-anomaly-detector";
constexpr int kFrames = 60;
// Every map pixel reaches a threshold of 0, so every frame sends one segment, whatever the test
// stream shows.
constexpr double kThreshold = 0.0;
constexpr int kMinRegionPx = 1;

// Every message has a segment, and every segment is an anomaly at or above the threshold, with its
// box and outline in the frame.
bool valid_segments(const std::vector<MetadataJsonMessage>& messages, int width, int height) {
  for (const auto& message : messages) {
    // A named value: a range-for over a member of a temporary would outlive the temporary.
    const auto payload = nlohmann::json::parse(message.payload);
    if (payload.at("data").at("segments").empty()) {
      std::cerr << "[FAIL] a frame sent no segment: " << message.payload << "\n";
      return false;
    }
    for (const auto& segment : payload.at("data").at("segments")) {
      const auto bbox = segment.at("bbox").get<std::array<int, 4>>();
      bool inside =
          bbox[0] >= 0 && bbox[1] >= 0 && bbox[0] + bbox[2] <= width && bbox[1] + bbox[3] <= height;
      for (const auto& point : segment.at("mask")) {
        const int x = point.at(0).get<int>();
        const int y = point.at(1).get<int>();
        inside &= x >= 0 && x < width && y >= 0 && y < height;
      }
      const double confidence = segment.at("confidence").get<double>();
      if (segment.at("label") != "anomaly" || confidence < kThreshold || confidence > 1.0 ||
          segment.at("mask_format") != "polygon" || !inside) {
        std::cerr << "[FAIL] unexpected segment: " << segment.dump() << "\n";
        return false;
      }
    }
  }
  return true;
}

// The first video packet's RTP header, or zeros when none arrives within 5 seconds.
std::array<unsigned char, 12> first_video_header(int socket) {
  const timeval timeout{5, 0};
  ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  std::array<unsigned char, 65536> packet{};
  std::array<unsigned char, 12> header{};
  if (::recv(socket, packet.data(), packet.size(), 0) > 12) {
    std::copy_n(packet.begin(), 12, header.begin());
  }
  return header;
}

int run_codec_case(const std::string& binary, const std::string& model_path,
                   const StreamSourceCase& source_case, const std::string& url,
                   const std::string& label) {
  const std::string output_dir = create_test_output_dir(kExampleName, "test_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  // This test is the Insight receiver, so the example publishes to loopback.
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  std::ofstream(config_path) << "model:\n  id: fastflow_demo\n  path: " << model_path
                             << "\ninput:\n  source: " << url << "\n  codec: " << source_case.codec
                             << "\n  width: 0\n  height: 0\n  fps: 0\n  rtsp:\n    tcp: true"
                             << "\n    latency_ms: 100\ninference:\n  frames: " << kFrames
                             << "\n  max_detections: 10\noutput:\n  insight:\n"
                             << "    host: 127.0.0.1\n    video_port_base: " << video_port
                             << "\n    metadata_port_base: " << metadata_port
                             << "\napp:\n  anomaly_threshold: " << kThreshold
                             << "\n  min_region_px: " << kMinRegionPx << "\n";

  MetadataJsonListenerOptions options;
  options.base_port = metadata_port;
  options.timeout_ms = 5000;
  options.require_all_ports = true;
  options.metadata_type = "segmentation";
  options.data_array_key = "segments";
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
  const auto header = first_video_header(video_socket);
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
  std::smatch stream;
  const std::regex stream_size(R"(stream=(\d+)x(\d+)@)");
  const std::regex summary("processed=" + std::to_string(kFrames) +
                           " flagged=" + std::to_string(kFrames) + " ");
  if (!metadata.success || !following.success ||
      !std::regex_search(result.stdout_text, stream, stream_size)) {
    std::cerr << "[FAIL] " << label << " sent no segmentation metadata: " << metadata.error
              << following.error << "\nstdout:\n"
              << result.stdout_text << "\n";
    return 1;
  }
  const int width = std::stoi(stream[1]);
  const int height = std::stoi(stream[2]);
  if (std::stoll(following.messages.back().frame_id) <=
          std::stoll(metadata.messages.back().frame_id) ||
      !valid_segments(metadata.messages, width, height) ||
      !valid_segments(following.messages, width, height) ||
      !valid_segments(rest.messages, width, height) ||
      !std::regex_search(result.stdout_text, summary)) {
    std::cerr << "[FAIL] " << label << " did not send " << kFrames
              << " frames of valid, progressing regions\nstdout:\n"
              << result.stdout_text << "\n";
    return 1;
  }
  // VideoSender forwards the stream as received: RTP H.264 (96), H.265 (98), or JPEG (26).
  const int payload_type = source_case.codec == "h265"    ? 98
                           : source_case.codec == "mjpeg" ? 26
                                                          : 96;
  if ((header[0] >> 6) != 2 || (header[1] & 0x7F) != payload_type) {
    std::cerr << "[FAIL] " << label << " did not publish RTP video with payload type "
              << payload_type << " to Insight\n";
    return 1;
  }
  std::cout << "[OK] " << label << " sent " << kFrames << " frames of regions and Insight video\n";
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
    return skip_or_fail("configured FastFlow model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const int cases_rc = run_single_stream_source_cases(
      "FastFlow anomaly detector",
      {
          {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
          {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
          {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 0, true},
      },
      [&](const StreamSourceCase& source_case, const std::string& url) {
        int case_rc = 0;
        for (const std::string& model_path : model_paths) {
          const std::string label =
              model_case_label(source_case.name, model_path, model_paths.size());
          if (run_codec_case(binary, model_path, source_case, url, label) != 0) {
            case_rc = 1;
          }
        }
        return case_rc;
      });
  return rc != 0 ? 1 : cases_rc;
}
