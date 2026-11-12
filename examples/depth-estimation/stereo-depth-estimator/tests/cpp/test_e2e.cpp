// End-to-end test for stereo-depth-estimator: on a rectified stereo pair of RTSP H.264 streams,
// the example must save disparity heatmaps and publish RTP H.264 video on three Insight channels.
// SIMANEAT_TEST_STEREO_RTSP_H264_URLS names the left stream, then the right.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "stereo-depth-estimator";
constexpr const char* kStreamsKey = "SIMANEAT_TEST_STEREO_RTSP_H264_URLS";
constexpr int kSaveEvery = 10;
constexpr int kSavedFrames = 3;
constexpr int kChannels = 3;

// The first saved frame, in name order.
cv::Mat first_saved_frame(const std::string& dir) {
  fs::path first;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".jpg" && (first.empty() || entry.path() < first)) {
      first = entry.path();
    }
  }
  return first.empty() ? cv::Mat() : cv::imread(first.string());
}

// A stereo pair has near and far points, so its disparity heatmap is not one flat color.
bool heatmap_is_flat(const cv::Mat& heatmap) {
  cv::Scalar mean;
  cv::Scalar deviation;
  cv::meanStdDev(heatmap, mean, deviation);
  return deviation[0] + deviation[1] + deviation[2] < 15.0;
}

// A UDP socket bound to a loopback port, or -1.
int bind_loopback_port(int port) {
  const int receiver = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(static_cast<uint16_t>(port));
  if (receiver >= 0 &&
      ::bind(receiver, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    ::close(receiver);
    return -1;
  }
  return receiver;
}

// Whether an RTP H.264 packet waits on the socket.
bool received_h264_rtp(int receiver) {
  std::array<unsigned char, 65536> packet{};
  const auto bytes = ::recv(receiver, packet.data(), packet.size(), MSG_DONTWAIT);
  return bytes > 12 && (packet[0] >> 6) == 2 && (packet[1] & 0x7F) == 96;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const char* models_dir = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path model =
      fs::path(models_dir ? models_dir : "models") /
      fs::path(example_common_config(kExampleName).string_or("model.path", "")).filename();
  if (!fs::is_regular_file(model)) {
    return skip_or_fail("model not found: " + model.string());
  }
  const char* streams = env_or_null(kStreamsKey);
  const std::string urls = streams ? streams : "";
  const auto comma = urls.find(',');
  if (comma == std::string::npos || comma == 0 || comma + 1 == urls.size()) {
    return skip_or_fail(std::string("set ") + kStreamsKey +
                        " to the left and the right stream of a stereo pair");
  }

  const std::string output_dir = create_test_output_dir(kExampleName, "rtsp_h264");
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config = fs::path(output_dir).parent_path() / "config.yaml";
  // This test is the Insight video receiver, so it publishes to its own loopback address.
  const std::string insight_host = "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  write_e2e_config(kExampleName, config,
                   {{"model.path", fs::absolute(model).string()},
                    {"source.type", "rtsp"},
                    {"source.rtsp.left_url", urls.substr(0, comma)},
                    {"source.rtsp.right_url", urls.substr(comma + 1)},
                    {"source.rtsp.codec", "h264"},
                    {"inference.frames", "0"},
                    {"output.save_dir", output_dir},
                    {"output.save_every", std::to_string(kSaveEvery)},
                    {"output.insight.host", insight_host},
                    {"output.insight.video_port", std::to_string(video_port)}});

  // The left camera, the right camera and the depth map each have a channel.
  std::array<int, kChannels> channels{};
  for (int channel = 0; channel < kChannels; ++channel) {
    channels[channel] = bind_loopback_port(video_port + channel);
    if (channels[channel] < 0) {
      std::cerr << "[FAIL] could not bind the Insight video port " << video_port + channel << "\n";
      remove_dir(output_dir);
      return 1;
    }
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(argv[1], {"--config", config.string()},
                                                        output_dir, kSavedFrames, timeout_ms);
  int channels_with_video = 0;
  for (const int channel : channels) {
    channels_with_video += received_h264_rtp(channel) ? 1 : 0;
    ::close(channel);
  }

  const std::regex start_line(R"(source=rtsp cameras=[0-9]+x[0-9]+@[0-9]+)");
  const std::regex summary(R"(frames=[0-9]+ depth_fps=[0-9.]+ insight=)" + insight_host + ":" +
                           std::to_string(video_port) + "-" +
                           std::to_string(video_port + kChannels - 1));
  const std::string frames_problem = saved_frames_problem(output_dir, kSavedFrames);
  const cv::Mat saved = first_saved_frame(output_dir);

  int rc = 1;
  const std::string exit_text = exit_problem(result);
  if (!exit_text.empty()) {
    std::cerr << "[FAIL] " << exit_text << "\n" << result.stderr_text << "\n";
  } else if (!std::regex_search(result.stdout_text, start_line) ||
             !std::regex_search(result.stdout_text, summary)) {
    std::cerr << "[FAIL] missing start or summary line\n" << result.stdout_text << "\n";
  } else if (!frames_problem.empty()) {
    std::cerr << "[FAIL] " << frames_problem << "\n";
  } else if (heatmap_is_flat(saved)) {
    std::cerr << "[FAIL] the saved depth maps are one flat color\n";
  } else if (channels_with_video != kChannels) {
    std::cerr << "[FAIL] RTP H.264 video reached " << channels_with_video << " of " << kChannels
              << " Insight ports\n";
  } else {
    std::cout << "[OK] saved " << saved.cols << "x" << saved.rows
              << " depth maps and sent video to " << kChannels << " Insight channels\n";
    rc = 0;
  }
  remove_dir(output_dir);
  return rc;
}
