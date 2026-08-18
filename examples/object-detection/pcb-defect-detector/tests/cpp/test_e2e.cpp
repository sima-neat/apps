// E2E test for pcb-defect-detector.
// Runs the binary with a real model pack and the bundled PCB test images,
// then verifies that one annotated image is written per input image.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr char kExample[] = "pcb-defect-detector";

int count_input_images(const std::string& input_dir) {
  int count = 0;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      ++count;
    }
  }
  return count;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  const std::string model_path = configured_model_path(kExample, models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured PCB defect model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  std::string labels_file;
  if (const char* labels_env = env_or_null("SIMANEAT_APPS_TEST_LABELS_FILE")) {
    labels_file = labels_env;
  }

  const std::string example_dir = fs::path(binary).parent_path().string();
  const std::vector<std::string> label_search = {
      "examples/object-detection/pcb-defect-detector/src/common/pcb_label.txt",
      example_dir + "/src/common/pcb_label.txt",
  };
  for (const auto& candidate : label_search) {
    if (!labels_file.empty()) {
      break;
    }
    if (fs::exists(candidate)) {
      labels_file = candidate;
    }
  }
  if (labels_file.empty()) {
    return skip_or_fail("src/common/pcb_label.txt not found; set SIMANEAT_APPS_TEST_LABELS_FILE "
                        "or ensure the example label file is available");
  }

  // PCB defects are not present in the shared COCO fixtures, so this example
  // uses its own test images instead of SIMANEAT_APPS_TEST_INPUT_DIR.
  const std::string input_dir = "assets/datasets-test/pcb";
  if (!fs::is_directory(input_dir) || fs::is_empty(input_dir)) {
    return skip_or_fail("PCB test images are missing or empty: " + input_dir);
  }

  const auto out_dir = create_test_output_dir(kExample, "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  write_e2e_config(kExample, config_path,
                   {{"model.path", model_path},
                    {"model.labels", labels_file},
                    {"io.input_dir", input_dir},
                    {"io.output_dir", out_dir}});

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  const int expected_files = count_input_images(input_dir);
  const int output_files = count_output_files(out_dir);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else if (output_files != expected_files) {
    std::cerr << "[FAIL] expected " << expected_files << " annotated images, found " << output_files
              << "\n";
    rc = 1;
  } else if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else if (result.stdout_text.find("Per-class totals:") == std::string::npos) {
    std::cerr << "[FAIL] run did not report per-class defect totals\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    rc = 1;
  } else {
    std::cout << "[OK] PCB defect detection produced " << output_files << " annotated images\n";
  }

  remove_dir(out_dir);
  return rc;
}
