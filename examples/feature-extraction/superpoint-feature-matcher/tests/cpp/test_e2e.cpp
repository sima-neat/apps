// End-to-end test for superpoint-feature-matcher: on each source, the example must match live
// frames to the reference, save side-by-side frames twice the stream's width, and publish RTP H.264
// video to Insight. Every source runs with LightGlue, and SuperGlue runs on RTSP H.264: the source
// graph does not depend on the matcher, and the matcher does not depend on the source.
#include "support/testing/source_cases.h"
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

// Runs the example with one matcher on one source; 0 when every check passes. `source_value` is
// the source's URL, or the camera's ID or name for a USB or MIPI camera.
int run_case(const std::string& binary, const fs::path& superpoint, const MatcherCase& matcher,
             const StreamSourceCase& source, const std::string& source_value) {
  const std::string name = matcher.type + "_" + source.name;
  const std::string output_dir = create_test_output_dir(kExampleName, name);
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config = fs::path(output_dir).parent_path() / ("config-" + name + ".yaml");
  // This test is the Insight video receiver, so it publishes to its own loopback address.
  const std::string insight_host = "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int save_every = e2e_int(kExampleName, "testing.e2e.output", "save_every");
  const int total_saved_frames = e2e_int(kExampleName, "testing.e2e.output", "total_saved_frames");
  const bool camera = source.type == "usb" || source.type == "mipi";
  const std::string section = "source." + source.type + ".";
  ConfigScalars overrides{
      {"superpoint.path", fs::absolute(superpoint).string()},
      {"matcher.type", matcher.type},
      {"matcher." + matcher.type + ".path", fs::absolute(matcher.model).string()},
      {"source.type", source.type},
      {section + (camera ? "camera" : "url"), source_value},
      {"inference.frames", "0"},
      {"output.save_dir", output_dir},
      {"output.save_every", std::to_string(save_every)},
      {"output.insight.host", insight_host},
      {"output.insight.video_port", std::to_string(video_port)}};
  if (source.type == "rtsp") {
    overrides[section + "codec"] = source.codec;
  }
  if (source.type == "http") {
    overrides[section + "ssl_strict"] = source.ssl_strict ? "true" : "false";
  }
  if (source.fps > 0) {
    overrides[section + "fps"] = std::to_string(source.fps);
  }
  write_e2e_config(kExampleName, config, overrides);

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
    std::cerr << "[FAIL] " << name << ": could not bind the Insight video port " << video_port
              << "\n";
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

  const std::regex start_line("source=" + source.type +
                              R"( stream=([0-9]+)x([0-9]+)@[0-9]+ matcher=)" + matcher.label + " ");
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
    std::cerr << "[FAIL] " << name << ": " << exit_text << "\n" << result.stderr_text << "\n";
    rc = 1;
  } else if (!started || !summarized) {
    std::cerr << "[FAIL] " << name << ": missing start or summary line\n"
              << result.stdout_text << "\n";
    rc = 1;
  } else if (std::stoi(totals[2].str()) == 0 || std::stod(totals[3].str()) <= 0.0) {
    std::cerr << "[FAIL] " << name << ": no live frame matched the reference\n"
              << result.stdout_text << "\n";
    rc = 1;
  } else if (!problem.empty()) {
    std::cerr << "[FAIL] " << name << ": " << problem << "\n";
    rc = 1;
  } else if (saved.cols != 2 * std::stoi(start[1].str()) ||
             saved.rows != std::stoi(start[2].str())) {
    std::cerr << "[FAIL] " << name << ": saved frames are " << saved.cols << "x" << saved.rows
              << ", not the reference beside the live frame\n";
    rc = 1;
  } else if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 || (video_packet[1] & 0x7F) != 96) {
    std::cerr << "[FAIL] " << name << ": no RTP H.264 video reached the Insight port\n";
    rc = 1;
  } else {
    std::cout << "[OK] " << name << ": matched " << totals[2] << " of " << totals[1]
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
  const MatcherCase& lightglue = matchers[0];
  const MatcherCase& superglue = matchers[1];
  const std::vector<StreamSourceCase> sources = {
      {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
      {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
      {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 30, true},
      {"http_mjpeg", "SIMANEAT_TEST_HTTP_MJPEG_URL", "http", "mjpeg", 30, false},
  };
  const int streams_rc = run_single_stream_source_cases(
      kExampleName, sources, [&](const StreamSourceCase& source, const std::string& url) {
        int rc = run_case(argv[1], superpoint, lightglue, source, url);
        if (source.name == "rtsp_h264") {
          rc |= run_case(argv[1], superpoint, superglue, source, url);
        }
        return rc;
      });

  // CI runners have no cameras, so these cases run only when the environment names one, even
  // under strict e2e mode.
  int camera_rc = 0;
  bool camera_ran = false;
  if (const char* usb = env_or_null("SIMANEAT_TEST_USB_CAMERA")) {
    camera_ran = true;
    camera_rc |=
        run_case(argv[1], superpoint, lightglue,
                 {"usb_camera", "SIMANEAT_TEST_USB_CAMERA", "usb", "mjpeg", 30, true}, usb);
  } else {
    std::cerr << "[SKIP] set SIMANEAT_TEST_USB_CAMERA to a USB camera ID to run usb_camera e2e\n";
  }
  if (const char* mipi = env_or_null("SIMA_NEAT_TEST_MIPI_CAMERA");
      mipi != nullptr && std::string(mipi) == "1") {
    camera_ran = true;
    const char* name = env_or_null("SIMA_NEAT_MIPI_CAMERA_NAME");
    camera_rc |= run_case(argv[1], superpoint, lightglue,
                          {"mipi_camera", "SIMA_NEAT_TEST_MIPI_CAMERA", "mipi", "", 0, true},
                          name != nullptr ? name : "");
  } else {
    std::cerr << "[SKIP] set SIMA_NEAT_TEST_MIPI_CAMERA=1 on a DevKit with a camera to run "
                 "mipi_camera e2e\n";
  }
  if (camera_rc != 0) {
    return 1;
  }
  // A camera-only run on a DevKit is a pass, not "no stream URLs configured".
  return camera_ran && streams_rc == 77 ? 0 : streams_rc;
}
