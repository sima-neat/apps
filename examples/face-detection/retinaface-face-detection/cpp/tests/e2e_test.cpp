// E2E test for retinaface-face-detection.
// Runs the binary with a real RetinaFace model and a local face image, and verifies it exits
// successfully and produces an annotated output image.
#include "support/testing/test_process.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
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

  // Follow the shared e2e convention: default to assets/test_images.
  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/test_images";

  std::string image_path;
  if (fs::exists(input_dir)) {
    std::vector<fs::path> image_candidates;
    for (auto& entry : fs::directory_iterator(input_dir)) {
      const auto ext = entry.path().extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        image_candidates.push_back(entry.path());
      }
    }

    std::sort(image_candidates.begin(), image_candidates.end());

    for (const auto& candidate : image_candidates) {
      std::string name = candidate.filename().string();
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (name.find("face") != std::string::npos) {
        image_path = candidate.string();
        break;
      }
    }

    if (image_path.empty() && !image_candidates.empty()) {
      image_path = image_candidates.front().string();
    }
  }

  if (image_path.empty()) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory containing test images (defaults to assets/test_images)");
    return kSkipCode; // not reached if env_or_skip exits, but keeps compiler happy
  }

  const std::string out_dir =
      create_test_output_dir("retinaface-face-detection", "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }
  fs::path out_image = fs::path(out_dir) / "retinaface_output.png";
  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "model:\n"
                << "  path: " << model_path << "\n"
                << "io:\n"
                << "  image: " << image_path << "\n"
                << "  output: " << out_image.string() << "\n"
                << "decode:\n"
                << "  confidence_threshold: 0.40\n"
                << "  nms_iou: 0.90\n"
                << "  top_k: 5000\n"
                << "  keep_top_k: 750\n"
                << "  max_draw: 50\n"
                << "  landmarks: true\n"
                << "runtime:\n"
                << "  timeout_ms: 5000\n"
                << "  profile: false\n"
                << "  num_runs: 1\n";
  }

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  if (!fs::exists(out_image)) {
    std::cerr << "[FAIL] expected annotated output image not found at " << out_image << "\n";
    remove_dir(out_dir);
    return 1;
  }

  remove_dir(out_dir);
  std::cout << "[OK] retinaface-face-detection pipeline completed successfully: " << out_image
            << "\n";
  return 0;
}
