#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace sima_examples::testing;

namespace {

std::vector<std::string> split_csv(const std::string& raw) {
  std::vector<std::string> out;
  std::istringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::size_t start = token.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    std::size_t end = token.find_last_not_of(" \t\r\n");
    out.push_back(token.substr(start, end - start + 1));
  }
  return out;
}

std::string find_model(const fs::path& models_dir) {
  if (!fs::exists(models_dir)) {
    return {};
  }
  for (const auto& entry : fs::directory_iterator(models_dir)) {
    const std::string name = entry.path().filename().string();
    if (name.find("yolo_v8") != std::string::npos &&
        name.find("seg") == std::string::npos &&
        name.size() >= 7 &&
        name.substr(name.size() - 7) == ".tar.gz") {
      return entry.path().string();
    }
  }
  return {};
}

int count_files_recursive(const fs::path& root) {
  int total = 0;
  std::error_code ec;
  if (!fs::exists(root, ec)) {
    return 0;
  }
  for (fs::recursive_directory_iterator it(root, ec), end; it != end && !ec; it.increment(ec)) {
    if (it->is_regular_file()) {
      ++total;
    }
  }
  return total;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* urls_raw = env_or_null("SIMANEAT_APPS_TEST_RTSP_URLS");
  const char* url_raw = env_or_null("SIMANEAT_APPS_TEST_RTSP_URL");

  std::vector<std::string> rtsp_urls;
  if (urls_raw) {
    rtsp_urls = split_csv(urls_raw);
  } else if (url_raw) {
    rtsp_urls.push_back(url_raw);
  }
  if (rtsp_urls.size() < 2) {
    return skip_or_fail("need at least two RTSP URLs for multistream e2e");
  }

  const std::string models_dir = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR")
                                     ? env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR")
                                     : "assets/models";
  const std::string model_path = find_model(models_dir);
  if (model_path.empty()) {
    return skip_or_fail("yolo detector model not found in SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir = create_temp_dir("multi_camera_people_tracking_e2e_");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir) / "config.yaml";
  {
    std::ofstream out(config_path);
    out << "model: " << model_path << "\n"
           "\n"
           "input:\n"
           "  tcp: true\n"
           "  latency_ms: 200\n"
           "\n"
           "inference:\n"
           "  frames: 10\n"
           "  fps: 0\n"
           "  bitrate_kbps: 2500\n"
           "  profile: false\n"
           "  person_class_id: 0\n"
           "  detection_threshold: null\n"
           "  nms_iou_threshold: null\n"
           "  top_k: null\n"
           "\n"
           "tracking:\n"
           "  iou_threshold: 0.3\n"
           "  max_missing_frames: 15\n"
           "\n"
           "output:\n"
           "  optiview:\n"
           "    host: 127.0.0.1\n"
           "    video_port_base: 9000\n"
           "    json_port_base: 9100\n"
           "  debug_dir: " << output_dir << "\n"
           "  save_every: 2\n"
           "\n"
           "streams:\n";
    for (std::size_t index = 0; index < 2; ++index) {
      out << "  - " << rtsp_urls[index] << "\n";
    }
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const int files = count_files_recursive(output_dir);
    if (files <= 0) {
      std::cerr << "[FAIL] expected sampled output files but output directory is empty\n";
      std::cerr << "stdout:\n" << result.stdout_text << "\n";
      std::cerr << "stderr:\n" << result.stderr_text << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] multi-camera tracking produced " << files << " sampled output files\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
