// E2E test for instance-overlay.
// Runs the binary with a real model and test images, verifies overlay outputs.
#include "support/testing/test_process.h"

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

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";

  std::string model_path;
  if (fs::exists(models_dir)) {
    for (auto& entry : fs::directory_iterator(models_dir)) {
      auto name = entry.path().filename().string();
      if (name.find("yolo_v8n_seg") != std::string::npos &&
          name.find(".tar.gz") != std::string::npos) {
        model_path = entry.path().string();
        break;
      }
    }
  }
  if (model_path.empty()) {
    return skip_or_fail("YOLOv8-seg model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/test_images is empty or missing)");
  }

  auto out_dir = create_test_output_dir("instance-overlay", "test_full_pipeline");
  if (out_dir.empty())
    return 1;

  const std::string config_path = out_dir + "/config.yaml";
  {
    std::ofstream config(config_path);
    config << "model:\n"
           << "  path: " << model_path << "\n"
           << "io:\n"
           << "  input_dir: " << input_dir << "\n"
           << "  output_dir: " << out_dir << "\n"
           << "runtime:\n"
           << "  infer_size: 640\n"
           << "  timeout_ms: 3000\n"
           << "  queue_depth: 8\n"
           << "decode:\n"
           << "  score_threshold: 0.60\n"
           << "  nms_iou: 0.45\n"
           << "  max_detections: 200\n"
           << "visualization:\n"
           << "  mask_alpha: 0.65\n";
  }

  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {"--config", config_path}, timeout);

  int rc = 0;
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    rc = 1;
  } else if (count_output_files(out_dir) == 0) {
    std::cerr << "[FAIL] no overlay output files produced\n";
    rc = 1;
  } else if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else {
    std::cout << "[OK] instance segmentation overlay produced " << count_output_files(out_dir)
              << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
