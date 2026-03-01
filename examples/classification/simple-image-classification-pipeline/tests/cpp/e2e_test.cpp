// E2E test for simple-image-classification-pipeline.
// Runs the binary with a real model and local test image, verifies it exits successfully.
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

  // Resolve model directory
  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  // Find a resnet model in the models directory
  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if ((name.find("resnet_50") != std::string::npos || name.find("resnet50") != std::string::npos) &&
          name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }

  if (model_path.empty()) {
    model_path = env_or_skip("SIMANEAT_APPS_TEST_MPK",
        "ResNet model (.tar.gz) in SIMANEAT_APPS_TEST_MODELS_DIR or SIMANEAT_APPS_TEST_MPK");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
        "directory with test images (assets/test_images is empty or missing)");
  }

  std::string image_path = (fs::path(input_dir) / "goldfish.jpeg").string();
  if (!fs::exists(image_path)) {
    image_path = (fs::path(input_dir) / "goldfish.jpg").string();
  }
  if (!fs::exists(image_path)) {
    image_path = (fs::path(input_dir) / "n01443537_goldfish.JPEG").string();
  }
  if (!fs::exists(image_path)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
        "directory must contain a goldfish image (goldfish.jpeg or n01443537_goldfish.JPEG)");
  }

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  auto r =
      spawn_and_wait(binary, {"--model", model_path, "--image", image_path, "--min-prob", "0.0"}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    return 1;
  }

  std::cout << "[OK] classification pipeline completed successfully\n";
  return 0;
}
