// E2E test for efficientsam3-promptable-segmenter: an RTSP H.264 stream in, H.264 video and
// person segments out to Insight.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "efficientsam3-promptable-segmenter";
constexpr int kFrames = 40;

bool valid_segments(const MetadataJsonListenerResult& result) {
  for (const auto& message : result.messages) {
    if (message.frame_id.empty() ||
        message.frame_id.find_first_not_of("0123456789") != std::string::npos ||
        message.timestamp_ms < 0) {
      return false;
    }
    for (const auto& segment : nlohmann::json::parse(message.payload).at("data").at("segments")) {
      const double confidence = segment.at("confidence").get<double>();
      if (segment.at("label") != "person" || confidence <= 0.3 || confidence > 1.0 ||
          segment.at("bbox").size() != 4U || segment.at("mask_format") != "polygon" ||
          segment.at("mask").size() < 3U) {
        return false;
      }
      for (const auto& value : segment.at("bbox")) {
        if (value.get<int>() < 0) {
          return false;
        }
      }
      for (const auto& point : segment.at("mask")) {
        if (point.size() != 2U || point[0].get<int>() < 0 || point[1].get<int>() < 0) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const char* models_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path models_dir = models_raw != nullptr ? models_raw : "models";
  const fs::path model = models_dir / "efficientsam3_rv_e48xt_flash_mpk.tar.gz";
  const fs::path text_encoder = models_dir / "efficientsam3_text_encoder_mpk.tar.gz";
  if (!fs::exists(model) || !fs::exists(text_encoder)) {
    return skip_or_fail("missing EfficientSAM3 artifacts under SIMANEAT_APPS_TEST_MODELS_DIR");
  }
  const char* rtsp_url = env_or_skip("SIMANEAT_TEST_RTSP_H264_URL", "RTSP H.264 stream URL");
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  const std::string output_dir = create_test_output_dir(kExampleName, "test_rtsp_h264_to_insight");
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  write_e2e_config(kExampleName, config_path,
                   {{"model.path", fs::absolute(model).string()},
                    {"model.text_encoder", fs::absolute(text_encoder).string()},
                    {"prompt.text", "person"},
                    {"source.rtsp_url", rtsp_url},
                    {"inference.frames", std::to_string(kFrames)},
                    {"output.insight.host", "127.0.0.1"},
                    {"output.insight.video_port", std::to_string(video_port)},
                    {"output.insight.metadata_port", std::to_string(metadata_port)}});

  MetadataJsonListenerOptions options;
  options.host = "127.0.0.1";
  options.base_port = metadata_port;
  options.num_ports = 1;
  options.timeout_ms = 5000;
  options.metadata_type = "segmentation";
  options.data_array_key = "segments";
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
  video_address.sin_port = htons(video_port);
  if (video_socket < 0 || ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address),
                                 sizeof(video_address)) != 0) {
    if (video_socket >= 0) {
      ::close(video_socket);
    }
    std::cerr << "[FAIL] could not bind Insight video port\n";
    return 1;
  }

  const ProcessResult process =
      spawn_and_wait(argv[1], {"--config", config_path.string()}, timeout_ms);
  std::array<unsigned char, 65536> video_packet{};
  const auto video_bytes =
      ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
  ::close(video_socket);
  if (process.exit_code != 0) {
    std::cerr << "[FAIL] exited with " << process.exit_code << "\nstdout:\n"
              << process.stdout_text << "\nstderr:\n"
              << process.stderr_text << "\n";
    return 1;
  }
  const auto metadata = listener.wait_for_messages();
  const auto following = listener.wait_for_messages();
  if (!metadata.success || !following.success || !valid_segments(metadata) ||
      !valid_segments(following)) {
    std::cerr << "[FAIL] invalid segmentation metadata: " << metadata.error << following.error
              << "\n";
    return 1;
  }
  const auto& first = metadata.messages.back();
  const auto& last = following.messages.back();
  if (std::stoll(last.frame_id) <= std::stoll(first.frame_id) ||
      last.timestamp_ms <= first.timestamp_ms ||
      process.stdout_text.find("processed=" + std::to_string(kFrames) + " ") == std::string::npos) {
    std::cerr << "[FAIL] did not complete with progressing frame identities\n";
    return 1;
  }
  if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 || (video_packet[1] & 0x7f) != 96) {
    std::cerr << "[FAIL] did not publish H.264 RTP video\n";
    return 1;
  }
  remove_dir(output_dir);
  std::cout << "[OK] published H.264 video and person segments\n";
  return 0;
}
