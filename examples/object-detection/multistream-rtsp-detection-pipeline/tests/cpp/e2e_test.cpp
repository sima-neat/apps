// E2E test for multistream-rtsp-detection-pipeline.
// Runs the binary with real model and RTSP streams, verifies per-stream output.
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  // Get RTSP URLs (comma-separated for multistream, fall back to single URL)
  const char* urls_raw = env_or_null("SIMANEAT_APPS_TEST_RTSP_URLS");
  const char* url_raw = env_or_null("SIMANEAT_APPS_TEST_RTSP_URL");

  std::vector<std::string> rtsp_urls;
  if (urls_raw) {
    std::istringstream ss(urls_raw);
    std::string token;
    while (std::getline(ss, token, ',')) {
      auto trimmed = token;
      while (!trimmed.empty() && trimmed.front() == ' ') trimmed.erase(trimmed.begin());
      while (!trimmed.empty() && trimmed.back() == ' ') trimmed.pop_back();
      if (!trimmed.empty()) rtsp_urls.push_back(trimmed);
    }
  } else if (url_raw) {
    rtsp_urls.push_back(url_raw);
  }

  if (rtsp_urls.empty()) {
    env_or_skip("SIMANEAT_APPS_TEST_RTSP_URLS",
        "comma-separated RTSP URLs (or SIMANEAT_APPS_TEST_RTSP_URL for single stream)");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("yolo_v8m") != std::string::npos && name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }
  if (model_path.empty()) {
    model_path = env_or_skip("SIMANEAT_APPS_TEST_MPK",
        "YOLO model (.tar.gz) in SIMANEAT_APPS_TEST_MODELS_DIR or SIMANEAT_APPS_TEST_MPK");
  }

  auto out_dir = create_temp_dir("multistream_e2e_");
  if (out_dir.empty()) return 1;

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  std::vector<std::string> args = {
    "--model", model_path,
    "--output", out_dir,
    "--frames", "10",
    "--fps", "30",
    "--width", "640",
    "--height", "360",
    "--save-every", "1",
    "--tcp"
  };
  for (auto& url : rtsp_urls) {
    args.push_back("--rtsp");
    args.push_back(url);
  }

  auto r = spawn_and_wait(binary, args, timeout);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (count_files(out_dir) == 0) {
    // Check subdirectories too (per-stream output)
    int total = 0;
    for (auto& entry : fs::directory_iterator(out_dir)) {
      if (entry.is_directory()) {
        total += count_files(entry.path().string());
      }
    }
    if (total == 0) {
      std::cerr << "[FAIL] no output files in any stream subdirectory\n";
      rc = 1;
    } else {
      std::cout << "[OK] multistream detection produced " << total
                << " output files across stream directories\n";
    }
  } else {
    std::cout << "[OK] multistream detection produced "
              << count_files(out_dir) << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
