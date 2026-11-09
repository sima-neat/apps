// End-to-end test for superpoint-feature-matcher: each matcher on an RTSP H.264 stream must match
// live frames to the reference, save side-by-side frames twice the stream's width, and publish
// RTP H.264 video to Insight.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "superpoint-feature-matcher";

// The model the shipped config names under `key`, in the test model directory.
fs::path test_model(const std::string& models_dir, const std::string& key) {
  return fs::path(models_dir) /
         fs::path(example_common_config(kExampleName).string_or(key, "")).filename();
}

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

struct MatcherCase {
  std::string type;  // the config's matcher.type
  std::string label; // the matcher's name in the app's start line
  fs::path model;
};

// Runs the example with one matcher; 0 when every check passes.
int run_matcher(const std::string& binary, const fs::path& superpoint, const MatcherCase& matcher,
                const std::string& source_url) {
  const std::string output_dir = create_test_output_dir(kExampleName, matcher.type + "_rtsp_h264");
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config = fs::path(output_dir).parent_path() / ("config-" + matcher.type + ".yaml");
  // This test is the Insight video receiver, so it publishes to its own loopback address.
  const std::string insight_host = "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int save_every = e2e_int(kExampleName, "testing.e2e.output", "save_every");
  const int total_saved_frames = e2e_int(kExampleName, "testing.e2e.output", "total_saved_frames");
  write_e2e_config(kExampleName, config,
                   {{"superpoint.path", fs::absolute(superpoint).string()},
                    {"matcher.type", matcher.type},
                    {"matcher." + matcher.type + ".path", fs::absolute(matcher.model).string()},
                    {"source.type", "rtsp"},
                    {"source.rtsp.url", source_url},
                    {"source.rtsp.codec", "h264"},
                    {"inference.frames", "0"},
                    {"output.save_dir", output_dir},
                    {"output.save_every", std::to_string(save_every)},
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
    std::cerr << "[FAIL] " << matcher.label << ": could not bind the Insight video port "
              << video_port << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(binary, {"--config", config.string()},
                                                        output_dir, total_saved_frames, timeout_ms);
  std::array<unsigned char, 65536> video_packet{};
  const auto video_bytes =
      ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
  ::close(video_socket);

  const std::regex start_line(R"(source=rtsp stream=([0-9]+)x([0-9]+)@[0-9]+ matcher=)" +
                              matcher.label + " ");
  const std::regex summary(R"(frames=([0-9]+) matched=([0-9]+) skipped=[0-9]+ )"
                           R"(average_matches=([0-9.]+) inlier_ratio=([0-9.]+) output_fps=[0-9.]+ )"
                           R"(video_sender=)" +
                           insight_host + ":" + std::to_string(video_port));
  std::smatch start;
  std::smatch totals;
  const bool started = std::regex_search(result.stdout_text, start, start_line);
  const bool summarized = std::regex_search(result.stdout_text, totals, summary);
  const std::string problem = streamed_frames_problem(output_dir, total_saved_frames);
  const cv::Mat saved = first_saved_frame(output_dir);

  int rc = 0;
  const std::string exit_text = exit_problem(result);
  if (!exit_text.empty()) {
    std::cerr << "[FAIL] " << matcher.label << ": " << exit_text << "\n"
              << result.stderr_text << "\n";
    rc = 1;
  } else if (!started || !summarized) {
    std::cerr << "[FAIL] " << matcher.label << ": missing start or summary line\n"
              << result.stdout_text << "\n";
    rc = 1;
  } else if (std::stoi(totals[2].str()) == 0 || std::stod(totals[3].str()) <= 0.0) {
    std::cerr << "[FAIL] " << matcher.label << ": no live frame matched the reference\n"
              << result.stdout_text << "\n";
    rc = 1;
  } else if (!problem.empty()) {
    std::cerr << "[FAIL] " << matcher.label << ": " << problem << "\n";
    rc = 1;
  } else if (saved.cols != 2 * std::stoi(start[1].str()) ||
             saved.rows != std::stoi(start[2].str())) {
    std::cerr << "[FAIL] " << matcher.label << ": saved frames are " << saved.cols << "x"
              << saved.rows << ", not the reference beside the live frame\n";
    rc = 1;
  } else if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 || (video_packet[1] & 0x7F) != 96) {
    std::cerr << "[FAIL] " << matcher.label << ": no RTP H.264 video reached the Insight port\n";
    rc = 1;
  } else {
    std::cout << "[OK] " << matcher.label << " matched " << totals[2] << " of " << totals[1]
              << " frames, average_matches=" << totals[3] << " inlier_ratio=" << totals[4]
              << ", saved side-by-side frames and sent Insight video\n";
  }
  remove_dir(output_dir);
  return rc;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  const fs::path superpoint = test_model(models_dir, "superpoint.path");
  const std::vector<MatcherCase> matchers = {
      {"lightglue", "LightGlue", test_model(models_dir, "matcher.lightglue.path")},
      {"superglue", "SuperGlue", test_model(models_dir, "matcher.superglue.path")},
  };
  if (!fs::is_regular_file(superpoint)) {
    return skip_or_fail("SuperPoint model not found under " + models_dir);
  }
  for (const MatcherCase& matcher : matchers) {
    if (!fs::is_regular_file(matcher.model)) {
      return skip_or_fail(matcher.label + " model not found under " + models_dir);
    }
  }
  const char* source_url = env_or_skip("SIMANEAT_TEST_RTSP_H264_URL", "RTSP H.264 stream URL");

  int rc = 0;
  for (const MatcherCase& matcher : matchers) {
    rc |= run_matcher(argv[1], superpoint, matcher, source_url);
  }
  return rc;
}
