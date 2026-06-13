// E2E test for yolov8-instance-segmenter.
// Runs the binary with a real model and test images, verifies overlay outputs.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

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

  const std::string model_path = configured_model_path("yolov8-instance-segmenter", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured segmentation model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_test_output_dir("yolov8-instance-segmenter", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  write_e2e_config(
      "yolov8-instance-segmenter", config_path,
      {{"model.path", model_path}, {"io.input_dir", input_dir}, {"io.output_dir", out_dir}});

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  const int output_files = count_output_files(out_dir);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (output_files == 0) {
    std::cerr << "[FAIL] expected output files but output directory is empty\n";
    rc = 1;
  } else if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else {
    std::cout << "[OK] YOLOv8 instance segmentation overlay produced " << output_files
              << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
