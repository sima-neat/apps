// E2E test for human-pose-estimator.
// Runs the binary with a real model and test images, verifies outputs.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <fstream>
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

  const std::string model_path = configured_model_path("human-pose-estimator", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured pose model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_test_output_dir("human-pose-estimator", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  const double keypoint_score = e2e_double("human-pose-estimator", "decode", "keypoint_score");
  const int nms_radius = e2e_int("human-pose-estimator", "decode", "nms_radius");
  const double paf_score = e2e_double("human-pose-estimator", "decode", "paf_score");
  const double paf_success_ratio =
      e2e_double("human-pose-estimator", "decode", "paf_success_ratio");
  const int paf_samples = e2e_int("human-pose-estimator", "decode", "paf_samples");
  const int min_valid_joints = e2e_int("human-pose-estimator", "decode", "min_valid_joints");
  const double min_avg_person_score =
      e2e_double("human-pose-estimator", "decode", "min_avg_person_score");
  {
    std::ofstream config_file(config_path);
    config_file << "model:\n";
    config_file << "  path: " << model_path << "\n";
    config_file << "io:\n";
    config_file << "  input_dir: " << input_dir << "\n";
    config_file << "  output_dir: " << out_dir << "\n";
    config_file << "runtime:\n";
    config_file << "  infer_size: 640\n";
    config_file << "  timeout_ms: 20000\n";
    config_file << "  upsample_factor: 4.0\n";
    config_file << "decode:\n";
    config_file << "  keypoint_score: " << keypoint_score << "\n";
    config_file << "  nms_radius: " << nms_radius << "\n";
    config_file << "  paf_score: " << paf_score << "\n";
    config_file << "  paf_success_ratio: " << paf_success_ratio << "\n";
    config_file << "  paf_samples: " << paf_samples << "\n";
    config_file << "  min_valid_joints: " << min_valid_joints << "\n";
    config_file << "  min_avg_person_score: " << min_avg_person_score << "\n";
  }

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
    std::cout << "[OK] pose estimation overlay produced " << output_files << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
