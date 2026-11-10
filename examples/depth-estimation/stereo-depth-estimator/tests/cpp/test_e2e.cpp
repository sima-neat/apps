// End-to-end test for stereo-depth-estimator: on a rectified stereo pair of RTSP H.264 streams,
// the example must save frames whose right half is a disparity heatmap and publish RTP H.264 video
// to Insight. SIMANEAT_TEST_STEREO_RTSP_H264_URLS names the left stream, then the right.
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

// A stereo pair has near and far points, so its disparity heatmap, the frame's right half, is not
// one flat color.
bool heatmap_is_flat(const cv::Mat& canvas) {
  cv::Scalar mean;
  cv::Scalar deviation;
  cv::meanStdDev(canvas.colRange(canvas.cols / 2, canvas.cols), mean, deviation);
  return deviation[0] + deviation[1] + deviation[2] < 15.0;
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
    std::cerr << "[FAIL] could not bind the Insight video port " << video_port << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(argv[1], {"--config", config.string()},
                                                        output_dir, kSavedFrames, timeout_ms);
  std::array<unsigned char, 65536> video_packet{};
  const auto video_bytes =
      ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
  ::close(video_socket);

  const std::regex start_line(R"(source=rtsp cameras=[0-9]+x[0-9]+@[0-9]+)");
  const std::regex summary(R"(frames=[0-9]+ output_fps=[0-9.]+ video_sender=)" + insight_host +
                           ":" + std::to_string(video_port));
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
  } else if (saved.cols % 2 != 0 || heatmap_is_flat(saved)) {
    std::cerr << "[FAIL] the saved frames hold no disparity heatmap beside the left view\n";
  } else if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 || (video_packet[1] & 0x7F) != 96) {
    std::cerr << "[FAIL] no RTP H.264 video reached the Insight port\n";
  } else {
    std::cout << "[OK] saved " << saved.cols << "x" << saved.rows
              << " frames with a disparity heatmap and sent Insight video\n";
    rc = 0;
  }
  remove_dir(output_dir);
  return rc;
}
