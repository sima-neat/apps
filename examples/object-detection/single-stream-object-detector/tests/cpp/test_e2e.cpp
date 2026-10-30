#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "single-stream-object-detector";

int run_source_case(const std::string& binary, const std::string& model_path,
                    const StreamSourceCase& source_case, const std::string& source_url,
                    const std::string& label) {
  const std::string output_dir =
      create_test_output_dir(kExampleName, "test_full_pipeline_" + label);
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  // This test is the Insight video receiver, so it publishes to its own loopback address.
  const std::string insight_host = "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int total_saved_frames = e2e_int(kExampleName, "testing.e2e.output", "total_saved_frames");
  // The case fills its source type's section; a USB case's "URL" is the camera's stable ID.
  const bool usb = source_case.type == "usb";
  const std::string src = "source." + source_case.type + ".";
  ConfigScalars overrides{{"source.type", source_case.type},
                          {src + (usb ? "camera" : "url"), source_url},
                          {"model.path", model_path},
                          {"output.save_dir", output_dir},
                          {"output.insight.host", insight_host},
                          {"output.insight.video_port", std::to_string(video_port)},
                          {"output.insight.metadata_port", std::to_string(metadata_port)}};
  if (source_case.type == "rtsp") {
    overrides[src + "codec"] = source_case.codec;
  }
  if (source_case.type == "http") {
    overrides[src + "ssl_strict"] = source_case.ssl_strict ? "true" : "false";
  }
  if (source_case.fps > 0) {
    overrides[src + "fps"] = std::to_string(source_case.fps);
  }
  write_e2e_config(kExampleName, config_path, overrides);

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
    remove_dir(output_dir);
    return 1;
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(binary, {"--config", config_path.string()},
                                                        output_dir, total_saved_frames, timeout_ms);
  std::array<unsigned char, 65536> video_packet{};
  const auto video_bytes =
      ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
  ::close(video_socket);

  int rc = 0;
  const std::string exit_problem_text = exit_problem(result);
  if (!exit_problem_text.empty()) {
    std::cerr << "[FAIL] " << label << " " << exit_problem_text << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const int files = count_output_files(output_dir);
    const std::string problem = streamed_frames_problem(output_dir, total_saved_frames);
    if (!problem.empty()) {
      std::cerr << "[FAIL] " << label << " " << problem << "\n";
      rc = 1;
    } else if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 ||
               (video_packet[1] & 0x7F) != (usb ? 26 : 96)) {
      // Streams are re-encoded to RTP H.264 (payload type 96); a USB camera's own JPEG is
      // forwarded as RTP/JPEG (payload type 26).
      std::cerr << "[FAIL] " << label << " did not publish RTP " << (usb ? "JPEG" : "H.264")
                << " video to Insight\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << label << " produced " << files
                << " sampled output files and Insight video\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("configured single-stream detector model not found under "
                        "SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::vector<StreamSourceCase> source_cases = {
      {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
      {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
      {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 0, true},
      {"http_mjpeg", "SIMANEAT_TEST_HTTP_MJPEG_URL", "http", "mjpeg", 30, false},
  };

  const int cases_rc = run_single_stream_source_cases(
      "single-stream object detector", source_cases,
      [&](const StreamSourceCase& source_case, const std::string& source_url) {
        int case_rc = 0;
        for (const std::string& model_path : model_paths) {
          const std::string label =
              model_case_label(source_case.name, model_path, model_paths.size());
          if (run_source_case(binary, model_path, source_case, source_url, label) != 0) {
            case_rc = 1;
          }
        }
        return case_rc;
      });
  // USB capture needs a camera on the DevKit, which CI runners do not have, so this case runs
  // only when SIMANEAT_TEST_USB_CAMERA names one (see usb_camera_ids()), even under --strict.
  int usb_rc = 0;
  bool usb_ran = false;
  const StreamSourceCase usb_case{"usb_camera", "SIMANEAT_TEST_USB_CAMERA", "usb", "mjpeg", 30,
                                  true};
  if (const char* camera = env_or_null(usb_case.env_key.c_str())) {
    usb_ran = true;
    for (const std::string& model_path : model_paths) {
      const std::string label = model_case_label(usb_case.name, model_path, model_paths.size());
      if (run_source_case(binary, model_path, usb_case, camera, label) != 0) {
        usb_rc = 1;
      }
    }
  } else {
    std::cerr << "[SKIP] set SIMANEAT_TEST_USB_CAMERA to a USB camera ID to run usb_camera e2e\n";
  }
  if (rc != 0 || usb_rc != 0) {
    return 1;
  }
  // A camera-only run on a DevKit is a pass, not "no stream URLs configured".
  return usb_ran && cases_rc == 77 ? 0 : cases_rc;
}
