// End-to-end test: an H.264 RTSP stream reaches Insight as video and feature-point metadata.
// tests/python/test_e2e.py runs the same checks against main.py.

#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/rtp_video.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "superpoint-feature-extractor";
constexpr int kFrames = 100;
// SuperPoint's input frame, which the decoder reports its keypoints in.
constexpr int kModelWidth = 640;
constexpr int kModelHeight = 480;
// The decoder keeps at most 600 points per frame (the A65V1 profile's top_k).
constexpr double kMaxPoints = 600.0;

// The completion summary: kFrames frames, an average from 0 (exclusive) to kMaxPoints, 256-value
// descriptors, and the configured video sender.
bool has_valid_summary(const std::string& text, int video_port) {
  const std::regex summary(
      "frames=" + std::to_string(kFrames) +
      R"( average_points=([0-9.]+) descriptor_dim=256 video_sender=127\.0\.0\.1:)" +
      std::to_string(video_port) + "\n");
  std::smatch match;
  return std::regex_search(text, match, summary) && std::stod(match[1].str()) > 0.0 &&
         std::stod(match[1].str()) <= kMaxPoints;
}

// Every point lies on the stream's frame, and some point reaches past the model frame: model-pixel
// points never would. The model input is the whole frame stretched, so points reach the frame's top
// and bottom tenth; with a letterbox those rows would be padding, and the stretch mapping would
// leave them empty. This needs a scene with texture near the top and bottom edges, as the test
// stream has.
bool valid_points(const std::vector<const std::vector<MetadataJsonMessage>*>& batches, int width,
                  int height) {
  bool past_model_frame = false;
  bool top = false;
  bool bottom = false;
  for (const auto* batch : batches) {
    for (const auto& message : *batch) {
      // A named value: a range-for over a member of a temporary would outlive the temporary.
      const auto payload = nlohmann::json::parse(message.payload);
      for (const auto& pose : payload.at("data").at("poses")) {
        for (const auto& point : pose.at("keypoints")) {
          const int x = point.at("x").get<int>();
          const int y = point.at("y").get<int>();
          if (x < 0 || y < 0 || x >= width || y >= height) {
            std::cerr << "[FAIL] unexpected feature point: " << point.dump() << "\n";
            return false;
          }
          past_model_frame |= x >= kModelWidth || y >= kModelHeight;
          top |= y < height / 10;
          bottom |= y >= height - height / 10;
        }
      }
    }
  }
  if (!past_model_frame) {
    std::cerr << "[FAIL] no feature point lies past the " << kModelWidth << "x" << kModelHeight
              << " model frame on the " << width << "x" << height << " stream\n";
    return false;
  }
  if (!top || !bottom) {
    std::cerr << "[FAIL] no feature point in the top or bottom tenth of the frame: the model "
                 "input is not the whole frame stretched\n";
    return false;
  }
  return true;
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
  const fs::path model = configured_model_path(kExampleName, models_dir);
  if (!fs::is_regular_file(model)) {
    return skip_or_fail("SuperPoint model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }
  const char* source_url = env_or_skip("SIMANEAT_TEST_RTSP_H264_URL", "RTSP H.264 stream URL");

  // This test is the Insight receiver, so the example publishes to loopback.
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const fs::path test_dir = create_test_output_dir(kExampleName, "rtsp_h264");
  const fs::path config_path = test_dir / "config.yaml";
  write_e2e_config(kExampleName, config_path,
                   {{"model.path", model.string()},
                    {"input.source", source_url},
                    {"inference.frames", std::to_string(kFrames)},
                    {"output.insight.host", "127.0.0.1"},
                    {"output.insight.video_port_base", std::to_string(video_port)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port)}});

  MetadataJsonListenerOptions options;
  options.base_port = metadata_port;
  options.timeout_ms = 5000;
  options.metadata_type = "pose-estimation";
  options.data_array_key = "poses";
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
  std::atomic<bool> stop{false};
  std::vector<RtpHeader> headers;
  std::thread drain([&] { headers = drain_video(video_socket, stop); });
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);
  stop = true;
  drain.join();
  ::close(video_socket);
  remove_dir(test_dir.string());

  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exited with " << result.exit_code << "\nstdout:\n"
              << result.stdout_text << "\nstderr:\n"
              << result.stderr_text << "\n";
    return 1;
  }
  if (!has_valid_summary(result.stdout_text, video_port)) {
    std::cerr << "[FAIL] missing or invalid feature summary\nstdout:\n"
              << result.stdout_text << "\nstderr:\n"
              << result.stderr_text << "\n";
    return 1;
  }
  std::smatch stream;
  const std::regex stream_size(R"(stream=(\d+)x(\d+)@)");
  if (!std::regex_search(result.stdout_text, stream, stream_size)) {
    std::cerr << "[FAIL] missing stream size\nstdout:\n" << result.stdout_text << "\n";
    return 1;
  }
  const int width = std::stoi(stream[1].str());
  const int height = std::stoi(stream[2].str());
  // The mapping from model-input pixels shows only on a stream larger than the model input.
  if (width <= kModelWidth && height <= kModelHeight) {
    std::cerr << "[FAIL] the " << width << "x" << height << " test stream must be larger than the "
              << kModelWidth << "x" << kModelHeight << " model input\n";
    return 1;
  }

  const auto metadata = listener.wait_for_messages();
  const auto following = listener.wait_for_messages();
  // The rest of the run's messages, so the checks below see every frame it sent.
  const auto rest = listener.drain_pending();
  if (!metadata.success || !following.success) {
    std::cerr << "[FAIL] sent no metadata with a feature point: " << metadata.error
              << following.error << "\n";
    return 1;
  }
  if (metadata.messages.back().frame_id.empty() || following.messages.back().frame_id.empty()) {
    std::cerr << "[FAIL] metadata without a frame_id\n";
    return 1;
  }
  if (std::stoll(following.messages.back().frame_id) <=
          std::stoll(metadata.messages.back().frame_id) ||
      !valid_points({&metadata.messages, &following.messages, &rest.messages}, width, height)) {
    std::cerr << "[FAIL] did not send progressing feature points on the stream's frame\n";
    return 1;
  }
  // The H.264 stream goes to Insight as received (RTP payload type 96).
  if (headers.empty() || (headers.front()[0] >> 6) != 2 || (headers.front()[1] & 0x7F) != 96) {
    std::cerr << "[FAIL] did not publish RTP H.264 video to Insight\n";
    return 1;
  }
  // Insight draws each message's points on the video frame its timestamp names.
  const std::set<uint32_t> timestamps = rtp_timestamps(headers);
  for (const auto* batch : {&metadata.messages, &following.messages, &rest.messages}) {
    for (const auto& message : *batch) {
      if (!pairs(message.timestamp_ms, timestamps)) {
        std::cerr << "[FAIL] sent metadata timestamp " << message.timestamp_ms
                  << " that no video frame carries\n";
        return 1;
      }
    }
  }
  std::cout << "[OK] sent " << kFrames << " frames of feature points and Insight video\n";
  return 0;
}
