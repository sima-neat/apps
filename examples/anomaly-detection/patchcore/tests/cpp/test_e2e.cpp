// E2E test for patchcore (C++).
//
// Test 1 runs --calibrate against the bundled COCO test image set to produce a
// fresh memory bank, then scores the same images and verifies annotated
// overlay output -- this only proves the pipeline runs end to end on real
// hardware, not that the resulting scores are meaningful (COCO has no
// nominal/defect labels), which is what tests 2 and 3 cover.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

int run_calibrate_then_score(const std::string& binary, const std::string& model_path) {
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

// A bank_meta.json pinned to a different model hash must fail at load, not
// silently score against a mismatched bank.
int run_bank_model_mismatch_fails_at_load(const std::string& binary) {
  auto out_dir = create_test_output_dir("patchcore", "test_bank_model_mismatch_fails_at_load");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  write_e2e_config(
      "patchcore", config_path,
      {
          {"memory_bank.path", bank_path.string()},
          {"memory_bank.meta_path", meta_path.string()},
          {"output.dir", out_dir},
      });

  // A minimal but structurally valid .npy file: patchcore::MemoryBank::load
  // must get far enough to be rejected by the hash check, not fail earlier
  // trying to parse the array.
  {
    std::ofstream bank(bank_path, std::ios::binary);
    static const unsigned char header[] = {
        0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00, 0x76, 0x00, '{', '\'', 'd', 'e', 's',
        'c', 'r', '\'', ':', ' ', '\'', '<', 'f', '4', '\'', ',', ' ', '\'', 'f', 'o',
        'r', 't', 'r', 'a', 'n', '_', 'o', 'r', 'd', 'e', 'r', '\'', ':', ' ', 'F', 'a',
        'l', 's', 'e', ',', ' ', '\'', 's', 'h', 'a', 'p', 'e', '\'', ':', ' ', '(', '4',
        ',', ' ', '1', '5', '3', '6', ')', ',', ' ', '}'};
    bank.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (int i = 0; i < 61; ++i) {
      bank.put('\n');
    }
    std::vector<float> zeros(4 * 1536, 0.0f);
    bank.write(reinterpret_cast<const char*>(zeros.data()),
               static_cast<std::streamsize>(zeros.size() * sizeof(float)));
  }
  {
    std::ofstream meta(meta_path);
    meta << "{\"model_sha256\": \"" << std::string(64, '0') << "\", \"threshold\": {\"value\": 1.0}}";
  }

  auto result = spawn_and_wait(binary, {"--config", config_path.string()}, 30000);
  remove_dir(out_dir);

  if (result.exit_code == 0) {
    std::cerr << "[FAIL] bank/model mismatch: expected a nonzero exit code\n";
    return 1;
  }
  if (result.stderr_text.find("different model package") == std::string::npos) {
    std::cerr << "[FAIL] bank/model mismatch: stderr does not explain the failure\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    return 1;
  }
  std::cout << "[OK] bank/model mismatch correctly rejected\n";
  return 0;
}

// Calibrates against a single known-good reference image, then scores that
// same image plus a real defect image -- proves the pipeline produces a
// meaningful anomaly signal end to end, not just that it runs and writes
// files (the bundled COCO set has no nominal/defect labels to assert against).
int run_defect_image_scores_higher_than_nominal(const std::string& binary,
                                                const std::string& model_path) {
  const fs::path real_images_dir = "assets/datasets/patchcore/images";
  const fs::path nominal_src = real_images_dir / "plain_0.png";
  const fs::path defect_src = real_images_dir / "scratch_0.png";
  if (!fs::exists(nominal_src) || !fs::exists(defect_src)) {
    return skip_or_fail("real nominal/defect images missing under " + real_images_dir.string());
  }

  auto out_dir = create_test_output_dir("patchcore", "test_defect_image_scores_higher_than_nominal");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  // Calibration must see only the nominal image -- including the defect image
  // here would poison the memory bank with anomalous patches (exactly the
  // mistake that corrupted a real bank during development).
  const fs::path nominal_dir = run_dir / "nominal";
  fs::create_directories(nominal_dir);
  fs::copy_file(nominal_src, nominal_dir / "plain_0.png", fs::copy_options::overwrite_existing);

  const fs::path score_dir = run_dir / "score_inputs";
  fs::create_directories(score_dir);
  fs::copy_file(nominal_src, score_dir / "plain_0.png", fs::copy_options::overwrite_existing);
  fs::copy_file(defect_src, score_dir / "scratch_0.png", fs::copy_options::overwrite_existing);

  write_e2e_config(
      "patchcore", config_path,
      {
          {"model.path", model_path},
          {"source.type", "image_dir"},
          {"source.image_dir", score_dir.string()},
          {"calibration.nominal_images_dir", nominal_dir.string()},
          {"calibration.threshold_images_dir", nominal_dir.string()},
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

  auto score = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  if (score.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << score.exit_code << "\n";
    std::cerr << "stderr:\n" << score.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  static const std::regex kScoreRe(R"(([^\s:]+):\s*score=([-\d.]+))");
  double nominal_score = -1.0;
  double defect_score = -1.0;
  for (std::sregex_iterator it(score.stdout_text.begin(), score.stdout_text.end(), kScoreRe), end;
       it != end; ++it) {
    const std::string path = (*it)[1].str();
    const double value = std::stod((*it)[2].str());
    if (path.find("plain_0.png") != std::string::npos) {
      nominal_score = value;
    } else if (path.find("scratch_0.png") != std::string::npos) {
      defect_score = value;
    }
  }

  remove_dir(out_dir);

  if (nominal_score < 0.0 || defect_score < 0.0) {
    std::cerr << "[FAIL] could not find both images' scores in stdout:\n" << score.stdout_text << "\n";
    return 1;
  }
  if (!(defect_score > nominal_score)) {
    std::cerr << "[FAIL] expected the defect image to score higher than the nominal image it was "
              << "calibrated on, got nominal=" << nominal_score << " defect=" << defect_score << "\n";
    return 1;
  }
  std::cout << "[OK] defect image (" << defect_score << ") scores higher than nominal ("
            << nominal_score << ")\n";
  return 0;
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
  const std::string model_path = configured_model_path("patchcore", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("patchcore model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  int failures = 0;
  failures += run_calibrate_then_score(binary, model_path) != 0;
  failures += run_bank_model_mismatch_fails_at_load(binary) != 0;
  failures += run_defect_image_scores_higher_than_nominal(binary, model_path) != 0;

  return failures > 0 ? 1 : 0;
}
