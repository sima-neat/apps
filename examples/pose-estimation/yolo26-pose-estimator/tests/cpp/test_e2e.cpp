// E2E test for yolo26-pose-estimator.
// Runs the binary with a real model and test images, then verifies pose overlays.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

bool has_nonzero_pose_count(const std::string& text) {
  const std::regex pattern(R"(\(([0-9]+) poses\))");
  auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    if (std::stoi((*it)[1].str()) > 0) {
      return true;
    }
  }
  return false;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  const std::string model_path = configured_model_path("yolo26-pose-estimator", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured YOLO26 pose model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_test_output_dir("yolo26-pose-estimator", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const double score_threshold = e2e_double("yolo26-pose-estimator", "decode", "score_threshold");
  const double nms_iou = e2e_double("yolo26-pose-estimator", "decode", "nms_iou");
  const int max_detections = e2e_int("yolo26-pose-estimator", "decode", "max_detections");
  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "model:\n"
                << "  path: " << model_path << "\n"
                << "io:\n"
                << "  input_dir: " << input_dir << "\n"
                << "  output_dir: " << out_dir << "\n"
                << "decode:\n"
                << "  score_threshold: " << score_threshold << "\n"
                << "  nms_iou: " << nms_iou << "\n"
                << "  max_detections: " << max_detections << "\n"
                << "runtime:\n"
                << "  timeout_ms: 20000\n"
                << "  num_runs: 1\n"
                << "  profile: false\n"
                << "output:\n"
                << "  overlay: true\n";
  }

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  const int output_files = count_output_files(out_dir);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (!has_nonzero_pose_count(r.stdout_text)) {
    std::cerr << "[FAIL] expected at least one decoded pose\n";
    rc = 1;
  } else if (output_files == 0) {
    std::cerr << "[FAIL] expected output files but output directory is empty\n";
    rc = 1;
  } else if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else {
    std::cout << "[OK] YOLO26 pose overlay produced " << output_files << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
