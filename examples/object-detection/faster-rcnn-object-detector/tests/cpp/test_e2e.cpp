// E2E test for faster-rcnn-object-detector.
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
  const fs::path models_dir = models_dir_raw ? models_dir_raw : "assets/models";
  const fs::path backbone = models_dir / "backbone_rpn_head_640_640_mpk.tar.gz";
  const fs::path head = models_dir / "box_head_predictor_640_640_mpk.tar.gz";
  if (!fs::exists(backbone) || !fs::exists(head)) {
    return skip_or_fail("Faster R-CNN model packages not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/test_images";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR", "directory containing test images (defaults to assets/test_images)");
    return kSkipCode;
  }

  const std::string out_dir = create_test_output_dir("faster-rcnn-object-detector", "test_full_pipeline_cpp");
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  write_e2e_config("faster-rcnn-object-detector", config_path,
                   {{"models.backbone_rpn.path", backbone.string()},
                    {"models.head_predictor.path", head.string()},
                    {"io.input_dir", input_dir},
                    {"io.output_dir", out_dir},
                    {"runtime.num_runs", "1"}});
  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 60000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const int output_files = count_output_files(out_dir);
  if (output_files == 0) {
    std::cerr << "[FAIL] expected annotated output images but output directory is empty\n";
    remove_dir(out_dir);
    return 1;
  }
  if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some annotated output images are empty\n";
    remove_dir(out_dir);
    return 1;
  }

  remove_dir(out_dir);
  std::cout << "[OK] faster-rcnn-object-detector C++ pipeline produced " << output_files << " output files\n";
  return 0;
}
