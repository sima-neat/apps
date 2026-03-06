// E2E test for simple-detess-segmentation-mask-overlay.
// Runs the binary with a real model and test images, verifies mask overlay outputs.
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

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("yolov5") != std::string::npos && name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }
  if (model_path.empty()) {
    model_path = env_or_skip("SIMANEAT_APPS_TEST_MPK",
        "YOLOv5 model (.tar.gz) in SIMANEAT_APPS_TEST_MODELS_DIR or SIMANEAT_APPS_TEST_MPK");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
        "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_temp_dir("simple-detess-segmentation-mask-overlay_e2e_");
  if (out_dir.empty()) return 1;

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {model_path, input_dir, out_dir}, timeout);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (count_files(out_dir) == 0) {
    std::cerr << "[FAIL] no overlay output files produced\n";
    rc = 1;
  } else if (!all_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else {
    bool bad_name = false;
    for (const auto& entry : fs::directory_iterator(out_dir)) {
      const auto name = entry.path().filename().string();
      if (name.size() < std::string("_overlay.jpg").size() ||
          name.rfind("_overlay.jpg") != name.size() - std::string("_overlay.jpg").size()) {
        bad_name = true;
        std::cerr << "[FAIL] unexpected output file name: " << name << "\n";
        break;
      }
    }
    if (bad_name) {
      rc = 1;
    }
  }

  if (rc == 0) {
    std::cout << "[OK] detess instance segmentation overlay produced "
              << count_files(out_dir) << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
