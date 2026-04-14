// E2E test for detr-object-detection.
// Runs the binary with a real DETR model and a local image, and verifies it exits successfully
// and produces an annotated output image.
#include "support/testing/test_process.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
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

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (const auto& entry : fs::directory_iterator(models_dir)) {
      const std::string name = entry.path().filename().string();
      if (name.find("detr_resnet50_modified_class_embed_bbox_embed") != std::string::npos &&
          name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }

  if (model_path.empty()) {
    return skip_or_fail("DETR model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/test_images";

  std::vector<fs::path> image_candidates;
  if (fs::exists(input_dir)) {
    for (const auto& entry : fs::directory_iterator(input_dir)) {
      const auto ext = entry.path().extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        image_candidates.push_back(entry.path());
      }
    }
  }
  std::sort(image_candidates.begin(), image_candidates.end());
  if (image_candidates.empty()) {
    env_or_skip(
        "SIMANEAT_APPS_TEST_INPUT_DIR",
        "directory containing test images (defaults to assets/test_images)");
    return kSkipCode;
  }

  const char* output_root_raw = env_or_null("SIMANEAT_APPS_TEST_OUTPUT_DIR");
  const std::string output_root = output_root_raw ? output_root_raw : "/tmp";
  fs::path out_dir = fs::path(output_root) / "cpp" / "detr-object-detection";

  std::error_code ec;
  fs::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "[FAIL] could not create output directory: " << out_dir << " (" << ec.message() << ")\n";
    return 1;
  }

  fs::path out_image = out_dir / "detr_output.png";
  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  auto r = spawn_and_wait(
      binary,
      {
          image_candidates.front().string(),
          "--model",
          model_path,
          "--conf",
          "0.5",
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

  std::cout << "[OK] detr-object-detection pipeline completed successfully: " << out_image << "\n";
  return 0;
}
