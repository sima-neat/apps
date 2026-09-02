// E2E test for patchcore (C++).
// Runs --calibrate against the bundled test image set to produce a fresh memory
// bank, then scores the same images and verifies annotated overlay output. The
// bundled COCO test images are not a real defect-free inspection set -- this
// only proves the pipeline runs end to end on real hardware, not that the
// resulting scores are meaningful; see the README for calibrating against a
// real nominal image set.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

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
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  const std::string model_path = configured_model_path("patchcore", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("patchcore model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/datasets-test/coco";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/datasets-test/coco is empty or missing)");
  }

  auto out_dir = create_test_output_dir("patchcore", "test_calibrate_then_score");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  write_e2e_config(
      "patchcore", config_path,
      {
          {"model.path", model_path},
          {"source.type", "image_dir"},
          {"source.image_dir", input_dir},
          {"calibration.nominal_images_dir", input_dir},
          {"calibration.threshold_images_dir", input_dir},
          {"memory_bank.path", bank_path.string()},
          {"memory_bank.meta_path", meta_path.string()},
          {"output.dir", out_dir},
      });

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto calibrate = spawn_and_wait(binary, {"--calibrate", "--config", config_path.string()}, timeout);
  if (calibrate.exit_code != 0) {
    std::cerr << "[FAIL] --calibrate exited with code " << calibrate.exit_code << "\n";
    std::cerr << "stderr:\n" << calibrate.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }
  if (!fs::exists(bank_path) || fs::file_size(bank_path) == 0 || !fs::exists(meta_path) ||
      fs::file_size(meta_path) == 0) {
    std::cerr << "[FAIL] --calibrate did not produce " << bank_path << " and " << meta_path << "\n";
    remove_dir(out_dir);
    return 1;
  }

  auto score = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  const int output_files = count_output_files(out_dir);

  int rc = 0;
  if (score.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << score.exit_code << "\n";
    std::cerr << "stderr:\n" << score.stderr_text << "\n";
    rc = 1;
  } else if (output_files == 0) {
    std::cerr << "[FAIL] expected overlay output files but output directory is empty\n";
    rc = 1;
  } else if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some output files are empty\n";
    rc = 1;
  } else {
    std::cout << "[OK] calibrate+score produced " << output_files << " output files\n";
  }

  remove_dir(out_dir);
  return rc;
}
