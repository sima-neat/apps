// E2E test for patchcore (C++). Test 1 runs --calibrate against the bundled
// nominal set, then scores the bundled test images and verifies verdicts
// and overlay output.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

// spawn_and_wait uses execv, not execvp, so a plain "python3" isn't resolved
// against PATH the way a shell would -- this does that resolution up front.
std::string resolve_python_bin() {
  if (const char* env = env_or_null("SIMANEAT_APPS_TEST_PYTHON_BIN")) {
    return env;
  }
  if (const char* path_env = std::getenv("PATH")) {
    std::stringstream ss(path_env);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
      const fs::path candidate = fs::path(dir) / "python3";
      if (fs::exists(candidate)) {
        return candidate.string();
      }
    }
  }
  return "";
}

int run_calibrate_then_score(const std::string& binary, const std::string& model_path) {
  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/datasets-test/coco";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/datasets-test/coco is empty or missing)");
  }

  auto out_dir = create_test_output_dir("patchcore-anomaly-detector", "test_calibrate_then_score");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  write_e2e_config(
      "patchcore-anomaly-detector", config_path,
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

// A run that writes some overlays and fails to write others must exit
// nonzero -- succeeding as long as at least one write went through would
// silently under-report incomplete output.
int run_partial_write_failure_fails_the_run(const std::string& binary, const std::string& model_path) {
  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/datasets-test/coco";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/datasets-test/coco is empty or missing)");
  }
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      images.push_back(entry.path());
    }
  }
  if (images.size() < 2) {
    return skip_or_fail("test input directory needs at least 2 images");
  }
  std::sort(images.begin(), images.end());

  auto out_dir =
      create_test_output_dir("patchcore-anomaly-detector", "test_partial_write_failure_fails_the_run");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  write_e2e_config(
      "patchcore-anomaly-detector", config_path,
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
    remove_dir(out_dir);
    return 1;
  }

  // Pre-create one output path as read-only so its overlay write fails while
  // every other image's write should still succeed.
  const std::string blocked_name = images.front().filename().string();
  const fs::path blocked_path = fs::path(out_dir) / blocked_name;
  {
    std::ofstream placeholder(blocked_path, std::ios::binary | std::ios::trunc);
  }
  fs::permissions(blocked_path, fs::perms::owner_read | fs::perms::group_read |
                                    fs::perms::others_read);

  auto score = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  fs::permissions(blocked_path, fs::perms::owner_all);

  int rc = 0;
  if (score.exit_code == 0) {
    std::cerr << "[FAIL] expected a nonzero exit code for a partial write failure, got 0\n";
    rc = 1;
  } else if (score.stderr_text.find(blocked_name) == std::string::npos) {
    std::cerr << "[FAIL] expected the failed path (" << blocked_name << ") named in stderr\n"
              << "stderr:\n" << score.stderr_text << "\n";
    rc = 1;
  } else {
    int other_outputs = 0;
    for (const auto& entry : fs::directory_iterator(out_dir)) {
      if (entry.is_regular_file() && entry.path().filename() != "config.yaml" &&
          entry.path().filename().string() != blocked_name) {
        ++other_outputs;
      }
    }
    if (other_outputs == 0) {
      std::cerr << "[FAIL] the one blocked write should not have stopped the rest from running\n";
      rc = 1;
    } else {
      std::cout << "[OK] partial write failure correctly fails the run\n";
    }
  }

  remove_dir(out_dir);
  return rc;
}

// A bank_meta.json pinned to a different model hash must fail at load, not
// silently score against a mismatched bank.
int run_bank_model_mismatch_fails_at_load(const std::string& binary) {
  auto out_dir = create_test_output_dir("patchcore-anomaly-detector", "test_bank_model_mismatch_fails_at_load");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  write_e2e_config(
      "patchcore-anomaly-detector", config_path,
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

// Scores every held-out normal image and every defect image, and asserts
// the actual pass/fail verdict on each -- not just relative ordering, and
// not just one cherry-picked passing image.
int run_held_out_normal_passes_and_defect_fails(const std::string& binary,
                                                const std::string& model_path) {
  const fs::path nominal_dir = "assets/datasets/patchcore/nominal";
  const fs::path held_out_dir = "assets/datasets/patchcore/held_out_normal";
  const fs::path images_dir = "assets/datasets/patchcore/images";
  if (!fs::exists(nominal_dir) || fs::is_empty(nominal_dir)) {
    return skip_or_fail("nominal calibration set missing under " + nominal_dir.string());
  }

  std::vector<std::string> held_out_names;
  if (fs::exists(held_out_dir)) {
    for (const auto& entry : fs::directory_iterator(held_out_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".png") {
        held_out_names.push_back(entry.path().filename().string());
      }
    }
  }
  std::vector<std::string> defect_names;
  if (fs::exists(images_dir)) {
    for (const auto& entry : fs::directory_iterator(images_dir)) {
      if (entry.is_regular_file() && entry.path().filename().string().rfind("scratch_", 0) == 0) {
        defect_names.push_back(entry.path().filename().string());
      }
    }
  }
  std::sort(held_out_names.begin(), held_out_names.end());
  std::sort(defect_names.begin(), defect_names.end());
  if (held_out_names.empty() || defect_names.empty()) {
    return skip_or_fail("held-out normal or defect images missing (" + held_out_dir.string() + ", " +
                        images_dir.string() + ")");
  }

  auto out_dir = create_test_output_dir("patchcore-anomaly-detector", "test_held_out_normal_passes_and_defect_fails");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  const fs::path score_dir = run_dir / "score_inputs";
  fs::create_directories(score_dir);
  for (const auto& name : held_out_names) {
    fs::copy_file(held_out_dir / name, score_dir / name, fs::copy_options::overwrite_existing);
  }
  for (const auto& name : defect_names) {
    fs::copy_file(images_dir / name, score_dir / name, fs::copy_options::overwrite_existing);
  }

  write_e2e_config(
      "patchcore-anomaly-detector", config_path,
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

  static const std::regex kVerdictRe(
      R"(([^\s:]+):\s*score=([-\d.]+)\s+threshold=[-\d.]+\s+verdict=(\w+))");
  std::map<std::string, std::pair<std::string, std::string>> verdicts; // name -> (verdict, score)
  for (std::sregex_iterator it(score.stdout_text.begin(), score.stdout_text.end(), kVerdictRe), end;
       it != end; ++it) {
    const fs::path path((*it)[1].str());
    verdicts[path.filename().string()] = {(*it)[3].str(), (*it)[2].str()};
  }

  remove_dir(out_dir);

  int rc = 0;
  std::vector<std::string> held_out_failures;
  for (const auto& name : held_out_names) {
    auto it = verdicts.find(name);
    if (it == verdicts.end()) {
      std::cerr << "[FAIL] no verdict for " << name << " in stdout:\n" << score.stdout_text << "\n";
      rc = 1;
      continue;
    }
    if (it->second.first != "normal") {
      held_out_failures.push_back(name + " scored " + it->second.second + " (ANOMALOUS)");
    }
  }
  std::vector<std::string> defect_failures;
  for (const auto& name : defect_names) {
    auto it = verdicts.find(name);
    if (it == verdicts.end()) {
      std::cerr << "[FAIL] no verdict for " << name << " in stdout:\n" << score.stdout_text << "\n";
      rc = 1;
      continue;
    }
    if (it->second.first != "ANOMALOUS") {
      defect_failures.push_back(name + " scored " + it->second.second + " (normal)");
    }
  }
  if (!held_out_failures.empty()) {
    std::cerr << "[FAIL] " << held_out_failures.size() << "/" << held_out_names.size()
              << " held-out normal images were flagged anomalous:\n";
    for (const auto& f : held_out_failures) std::cerr << "  " << f << "\n";
    rc = 1;
  }
  if (!defect_failures.empty()) {
    std::cerr << "[FAIL] " << defect_failures.size() << "/" << defect_names.size()
              << " defect images were not flagged anomalous:\n";
    for (const auto& f : defect_failures) std::cerr << "  " << f << "\n";
    rc = 1;
  }
  if (rc == 0) {
    std::cout << "[OK] all " << held_out_names.size() << " held-out normal images pass, all "
              << defect_names.size() << " defect images fail\n";
  }
  return rc;
}

// Cross-language regression: a memory bank calibrated by Python must load
// and separate defect from nominal correctly when scored by C++ (see
// squared_diff_sum in patchcore_memory_bank.cpp for the distance formula
// both languages share).
int run_python_built_bank_scores_correctly_in_cpp(const std::string& binary,
                                                  const std::string& model_path) {
  const fs::path real_images_dir = "assets/datasets/patchcore/images";
  const fs::path nominal_src = real_images_dir / "plain_0.png";
  const fs::path defect_src = real_images_dir / "scratch_0.png";
  if (!fs::exists(nominal_src) || !fs::exists(defect_src)) {
    return skip_or_fail("real nominal/defect images missing under " + real_images_dir.string());
  }
  const fs::path python_main =
      "examples/anomaly-detection/patchcore-anomaly-detector/src/python/main.py";
  if (!fs::exists(python_main)) {
    return skip_or_fail("Python main.py not found at " + python_main.string());
  }
  const std::string python_bin = resolve_python_bin();
  if (python_bin.empty()) {
    return skip_or_fail("python3 not found (set SIMANEAT_APPS_TEST_PYTHON_BIN)");
  }

  auto out_dir = create_test_output_dir("patchcore-anomaly-detector", "test_python_built_bank_scores_correctly_in_cpp");
  if (out_dir.empty())
    return 1;

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path bank_path = run_dir / "memory_bank.npy";
  const fs::path meta_path = run_dir / "bank_meta.json";

  const fs::path nominal_dir = run_dir / "nominal";
  fs::create_directories(nominal_dir);
  fs::copy_file(nominal_src, nominal_dir / "plain_0.png", fs::copy_options::overwrite_existing);

  const fs::path score_dir = run_dir / "score_inputs";
  fs::create_directories(score_dir);
  fs::copy_file(nominal_src, score_dir / "plain_0.png", fs::copy_options::overwrite_existing);
  fs::copy_file(defect_src, score_dir / "scratch_0.png", fs::copy_options::overwrite_existing);

  write_e2e_config(
      "patchcore-anomaly-detector", config_path,
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

  auto calibrate = spawn_and_wait(
      python_bin, {python_main.string(), "--calibrate", "--config", config_path.string()}, timeout);
  if (calibrate.exit_code != 0) {
    std::cerr << "[FAIL] Python --calibrate exited with code " << calibrate.exit_code << "\n";
    std::cerr << "stderr:\n" << calibrate.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  auto score = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  if (score.exit_code != 0) {
    std::cerr << "[FAIL] C++ scoring against a Python-built bank exited with code " << score.exit_code
              << "\n";
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
    std::cerr << "[FAIL] C++ scoring a Python-built bank must still separate defect from nominal: "
              << "nominal=" << nominal_score << " defect=" << defect_score << "\n";
    return 1;
  }
  std::cout << "[OK] C++ scores a Python-built bank correctly: defect (" << defect_score
            << ") > nominal (" << nominal_score << ")\n";
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
  const std::string model_path = configured_model_path("patchcore-anomaly-detector", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("patchcore model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  int failures = 0;
  failures += run_calibrate_then_score(binary, model_path) != 0;
  failures += run_partial_write_failure_fails_the_run(binary, model_path) != 0;
  failures += run_bank_model_mismatch_fails_at_load(binary) != 0;
  failures += run_held_out_normal_passes_and_defect_fails(binary, model_path) != 0;
  failures += run_python_built_bank_scores_correctly_in_cpp(binary, model_path) != 0;

  return failures > 0 ? 1 : 0;
}
