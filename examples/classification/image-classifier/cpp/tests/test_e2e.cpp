// E2E test for image-classifier.
// Runs the binary with a real model and local test image, verifies it exits successfully.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <filesystem>
#include <fstream>
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

  const std::string model_path = configured_model_path("image-classifier", models_dir);

  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("ResNet model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  std::string image_path;
  if (const char* image_env = env_or_null("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE")) {
    image_path = image_env;
  } else {
    image_path = "assets/test_images_classification/goldfish.jpeg";
  }
  if (!fs::exists(image_path)) {
    env_or_skip("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
                "path to goldfish image for classification e2e (e.g. "
                "assets/test_images_classification/goldfish.jpeg)");
  }

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  auto out_dir = create_test_output_dir("image-classifier", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const int expected_class_id = e2e_int("image-classifier", "validation", "expected_class_id");
  const double min_probability = e2e_double("image-classifier", "validation", "min_probability");
  const std::string config_path = (fs::path(out_dir).parent_path() / "config.yaml").string();
  {
    std::ofstream config(config_path);
    config << "model:\n"
           << "  path: " << model_path << "\n"
           << "io:\n"
           << "  image: " << image_path << "\n"
           << "  fallback_image_url: null\n"
           << "runtime:\n"
           << "  input_width: 224\n"
           << "  input_height: 224\n"
           << "  timeout_ms: 20000\n"
           << "validation:\n"
           << "  expected_class_id: " << expected_class_id << "\n"
           << "  min_probability: " << min_probability << "\n";
  }

  auto r = spawn_and_wait(binary, {"--config", config_path}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  std::cout << "[OK] classification pipeline completed successfully\n";
  remove_dir(out_dir);
  return 0;
}
