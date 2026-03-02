// E2E test for live-rtsp-depth-estimation.
// Runs the binary with a real model and RTSP stream, verifies output video.
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* rtsp_url = env_or_skip("SIMANEAT_APPS_TEST_RTSP_URL", "RTSP stream URL");

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("midas_v21_small_256") != std::string::npos &&
          name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }
  if (model_path.empty()) {
    model_path = env_or_skip("SIMANEAT_APPS_TEST_MPK",
        "depth model (.tar.gz) in SIMANEAT_APPS_TEST_MODELS_DIR or SIMANEAT_APPS_TEST_MPK");
  }

  auto out_dir = create_temp_dir("live-rtsp-depth-estimation_e2e_");
  if (out_dir.empty()) return 1;
  std::string out_file = out_dir + "/depth_test.mp4";

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {
    "--url", rtsp_url,
    "--model", model_path,
    "--output-file", out_file,
    "--frames", "5",
    "--fps", "30",
    "--tcp"
  }, timeout);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (!fs::exists(out_file) || fs::file_size(out_file) == 0) {
    std::cerr << "[FAIL] output video not created or empty\n";
    rc = 1;
  } else {
    std::cout << "[OK] live RTSP depth estimation produced output video ("
              << fs::file_size(out_file) << " bytes)\n";
  }

  remove_dir(out_dir);
  return rc;
}
