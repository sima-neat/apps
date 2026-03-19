// E2E test for retinaface-face-detection.
// Runs the binary with a real RetinaFace model and a local face image, and verifies it exits successfully
// and produces an annotated output image.
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

  // Resolve model directory (same env var used by Python tests).
  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  // Find a RetinaFace model in the models directory.
  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      const auto name = entry.path().filename().string();
      if (name.find("retinaface_mobilenet25") != std::string::npos &&
          name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }

  if (model_path.empty()) {
    return skip_or_fail("RetinaFace model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  // Use test images directory for input (faces image if available).
  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/test_images";

  std::string image_path;
  if (fs::exists(input_dir)) {
    for (auto& entry : fs::directory_iterator(input_dir)) {
      const auto name = entry.path().filename().string();
      if (name.find("face") != std::string::npos &&
          (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" ||
           entry.path().extension() == ".jpeg")) {
        image_path = entry.path().string();
        break;
      }
    }
  }

  if (image_path.empty()) {
    env_or_skip(
        "SIMANEAT_APPS_TEST_INPUT_DIR",
        "directory containing a face test image (e.g. assets/test_images/faces.png)");
    return kSkipCode; // not reached if env_or_skip exits, but keeps compiler happy
  }

  // Resolve output root.
  const char* output_root_raw = env_or_null("SIMANEAT_APPS_TEST_OUTPUT_DIR");
  const std::string output_root = output_root_raw ? output_root_raw : "/tmp";

  fs::path out_dir = fs::path(output_root) / "cpp" / "retinaface-face-detection";
  std::error_code ec;
  fs::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "[FAIL] could not create output directory: " << out_dir << " (" << ec.message() << ")\n";
    return 1;
  }
  fs::path out_image = out_dir / "retinaface_output.png";

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  auto r = spawn_and_wait(
      binary,
      {
          image_path,
          "--model",
          model_path,
          "--conf",
          "0.4",
          "--nms",
          "0.9",
          "--output",
          out_image.string(),
      },
      timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    return 1;
  }

  if (!fs::exists(out_image)) {
    std::cerr << "[FAIL] expected annotated output image not found at " << out_image << "\n";
    return 1;
  }

  std::cout << "[OK] retinaface-face-detection pipeline completed successfully: " << out_image << "\n";
  return 0;
}

