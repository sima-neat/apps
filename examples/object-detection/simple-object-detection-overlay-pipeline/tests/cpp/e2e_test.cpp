// E2E test for simple-object-detection-overlay-pipeline.
// Runs the binary with a real model, labels, and test images, verifies outputs.
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

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("yolo_v8n") != std::string::npos && name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }
  if (model_path.empty()) {
    model_path = env_or_skip("SIMANEAT_APPS_TEST_MPK",
        "YOLOv8 model (.tar.gz) in SIMANEAT_APPS_TEST_MODELS_DIR or SIMANEAT_APPS_TEST_MPK");
  }

  std::string labels_file;
  if (const char* labels_env = env_or_null("SIMANEAT_APPS_TEST_LABELS_FILE")) {
    labels_file = labels_env;
  }

  std::string example_dir = fs::path(binary).parent_path().string();
  std::vector<std::string> label_search = {
    "examples/object-detection/simple-object-detection-overlay-pipeline/coco_label.txt",
    example_dir + "/coco_label.txt",
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
    return skip_or_fail(
        "coco_label.txt not found; set SIMANEAT_APPS_TEST_LABELS_FILE or ensure example label file is available");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
        "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_temp_dir("objdet_e2e_");
  if (out_dir.empty()) return 1;

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {model_path, labels_file, input_dir, out_dir}, timeout);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (count_files(out_dir) == 0) {
    std::cerr << "[FAIL] no annotated output files produced\n";
    rc = 1;
  } else if (!all_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else {
    std::cout << "[OK] object detection overlay produced "
              << count_files(out_dir) << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
