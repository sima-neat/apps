// E2E test for detr-object-detector.
// Runs the binary with a real DETR model and a local image, and verifies it exits successfully
// and produces an annotated output image.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <algorithm>
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

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  const std::string model_path = configured_model_path("detr-object-detector", models_dir);

  if (model_path.empty() || !fs::exists(model_path)) {
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
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory containing test images (defaults to assets/test_images)");
    return kSkipCode;
  }

  const std::string out_dir = create_test_output_dir("detr-object-detector", "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  fs::path out_image = fs::path(out_dir) / "detr_output.png";
  const double confidence_threshold =
      e2e_double("detr-object-detector", "decode", "confidence_threshold");
  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "model:\n"
                << "  path: " << model_path << "\n"
                << "io:\n"
                << "  image: " << image_candidates.front().string() << "\n"
                << "  output: " << out_image.string() << "\n"
                << "decode:\n"
                << "  confidence_threshold: " << confidence_threshold << "\n"
                << "  max_draw: 50\n"
                << "  person_only: false\n"
                << "runtime:\n"
                << "  timeout_ms: 20000\n"
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
  if (fs::is_empty(out_image)) {
    std::cerr << "[FAIL] annotated output image is empty at " << out_image << "\n";
    remove_dir(out_dir);
    return 1;
  }

  remove_dir(out_dir);
  std::cout << "[OK] detr-object-detector pipeline completed successfully: " << out_image << "\n";
  return 0;
}
