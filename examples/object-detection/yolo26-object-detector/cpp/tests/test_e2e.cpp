// E2E test for yolo26-object-detector.
// Runs the binary with a real model, labels, and test images, verifies outputs.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

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

  const std::string model_path = configured_model_path("yolo26-object-detector", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured YOLO26 model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  std::string labels_file;
  if (const char* labels_env = env_or_null("SIMANEAT_APPS_TEST_LABELS_FILE")) {
    labels_file = labels_env;
  }

  std::string example_dir = fs::path(binary).parent_path().string();
  std::vector<std::string> label_search = {
      "examples/object-detection/yolo26-object-detector/"
      "common/coco_label.txt",
      example_dir + "/common/coco_label.txt",
  };
  for (const auto& p : label_search) {
    if (!labels_file.empty()) {
      break;
    }
    if (fs::exists(p)) {
      labels_file = p;
    }
  }
  if (labels_file.empty()) {
    return skip_or_fail("common/coco_label.txt not found; set SIMANEAT_APPS_TEST_LABELS_FILE "
                        "or ensure example label file is available");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_test_output_dir("yolo26-object-detector", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const double score_threshold = e2e_double("yolo26-object-detector", "decode", "score_threshold");
  const double nms_iou = e2e_double("yolo26-object-detector", "decode", "nms_iou");
  const int max_detections = e2e_int("yolo26-object-detector", "decode", "max_detections");
  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  {
    std::ofstream config_file(config_path);
    config_file << "model:\n"
                << "  path: " << model_path << "\n"
                << "  labels: " << labels_file << "\n"
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
    std::cout << "[OK] yolo26m object detection overlay produced " << output_files
              << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
