// E2E test for mae-medical-anomaly-detection.
// Runs the binary with real models and test .npy files, verifies it exits successfully.
#include "support/testing/test_process.h"

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

  // Resolve model directory
  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  // Find MAE models in the models directory
  std::string mae_model_path;
  std::string cls_model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("mae_brats_deterministic_grid_masking_simplified_3channels_mpk") != std::string::npos &&
          name.find(".tar.gz") != std::string::npos) {
        mae_model_path = entry.path().string();
      }
      if (name.find("mae_classifier_brats_split_3channels_mpk") != std::string::npos &&
          name.find(".tar.gz") != std::string::npos) {
        cls_model_path = entry.path().string();
      }
    }
  }

  if (mae_model_path.empty()) {
    return skip_or_fail("MAE model (*mae_brats_deterministic_grid_masking_simplified_3channels_mpk*.tar.gz) "
                        "not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  if (cls_model_path.empty()) {
    return skip_or_fail("Classifier model (*mae_classifier_brats_split_3channels_mpk*.tar.gz) "
                        "not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  // Find test .npy files
  std::string test_images_dir = "assets/test_images_anomaly_detection";
  if (const char* input_dir_env = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR")) {
    // Try anomaly_detection subdirectory first
    test_images_dir = std::string(input_dir_env) + "/anomaly_detection";
    if (!fs::exists(test_images_dir)) {
      test_images_dir = input_dir_env;
    }
  }

  std::vector<std::string> npy_files;
  if (fs::exists(test_images_dir)) {
    for (auto& entry : fs::directory_iterator(test_images_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".npy") {
        npy_files.push_back(entry.path().string());
      }
    }
  }

  if (npy_files.empty()) {
    return skip_or_fail("No .npy test files found under " + test_images_dir +
                        " (set SIMANEAT_APPS_TEST_INPUT_DIR to override)");
  }

  // Use first few .npy files (limit to 2 for faster tests)
  if (npy_files.size() > 2) {
    npy_files.resize(2);
  }

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  // Create temp output directory
  std::string output_dir = create_temp_dir("mae_ad_e2e_");

  // Build command args
  std::vector<std::string> args = {mae_model_path, cls_model_path};
  args.insert(args.end(), npy_files.begin(), npy_files.end());
  args.push_back("--output-dir");
  args.push_back(output_dir);

  auto r = spawn_and_wait(binary, args, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stdout:\n" << r.stdout_text << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(output_dir);
    return 1;
  }

  // Verify output files were created
  int viz_count = count_files(output_dir);
  if (viz_count == 0) {
    std::cerr << "[WARN] no visualization files found in output directory\n";
  } else {
    std::cout << "[OK] found " << viz_count << " visualization file(s)\n";
  }

  // Cleanup
  if (const char* keep_output = env_or_null("SIMANEAT_APPS_TEST_KEEP_OUTPUT")) {
    if (std::string(keep_output) == "1") {
      std::cout << "[INFO] keeping output directory: " << output_dir << "\n";
    } else {
      remove_dir(output_dir);
    }
  } else {
    remove_dir(output_dir);
  }

  std::cout << "[OK] MAE anomaly detection pipeline completed successfully\n";
  return 0;
}
