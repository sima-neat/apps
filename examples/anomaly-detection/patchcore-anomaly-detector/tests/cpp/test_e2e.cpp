// E2E test for patchcore (C++). Test 1 runs --calibrate against the bundled
// nominal set, then scores the images in SIMANEAT_APPS_TEST_INPUT_DIR and
// verifies the overlay output; the held-out test checks verdicts and overlays
// on the bundled held-out normal and defect images.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
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

  // One decodable overlay per input image.
  const int expected_images = supported_image_count(input_dir);
  const int output_files = count_output_files(out_dir);
  const std::string output_problem = saved_frames_problem(out_dir, expected_images);

  int rc = 0;
  if (score.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << score.exit_code << "\n";
    std::cerr << "stderr:\n" << score.stderr_text << "\n";
    rc = 1;
  } else if (!output_problem.empty()) {
    std::cerr << "[FAIL] overlay output: " << output_problem << "\n";
    rc = 1;
  } else {
    std::cout << "[OK] calibrate+score produced " << output_files << " decodable overlays for "
              << expected_images << " input images\n";
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

// Share of pixels a defect overlay may highlight: above zero (the defect is
// shown) and well below the whole image (it is localized).
constexpr double kMinDefectHighlight = 0.005;
constexpr double kMaxDefectHighlight = 0.60;

// Fraction of pixels the overlay changed relative to the input image; -1 if
// either image can't be read or their sizes differ.
double highlighted_fraction(const fs::path& input_path, const fs::path& overlay_path) {
  const cv::Mat src = cv::imread(input_path.string(), cv::IMREAD_COLOR);
  const cv::Mat out = cv::imread(overlay_path.string(), cv::IMREAD_COLOR);
  if (src.empty() || out.empty() || src.size() != out.size()) {
    return -1.0;
  }
  cv::Mat diff;
  cv::absdiff(src, out, diff);
  cv::Mat gray;
  cv::transform(diff, gray, cv::Matx13f(1.f, 1.f, 1.f));
  return static_cast<double>(cv::countNonZero(gray)) / static_cast<double>(gray.total());
}

// Scores every held-out normal image and every defect image, and asserts
// the actual pass/fail verdict on each -- not just relative ordering, and
// not just one cherry-picked passing image. Also checks the overlays:
// held-out normal overlays must be pixel-identical to their inputs, and
// defect overlays must highlight a localized region.
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

  std::map<std::string, double> highlighted;
  for (const auto& name : held_out_names) {
    highlighted[name] = highlighted_fraction(score_dir / name, fs::path(out_dir) / name);
  }
  for (const auto& name : defect_names) {
    highlighted[name] = highlighted_fraction(score_dir / name, fs::path(out_dir) / name);
  }
  remove_dir(out_dir);

  int rc = 0;
  for (const auto& name : held_out_names) {
    if (highlighted[name] != 0.0) {
      std::cerr << "[FAIL] held-out normal overlay " << name << " highlighted fraction "
                << highlighted[name] << " (expected 0)\n";
      rc = 1;
    }
  }
  for (const auto& name : defect_names) {
    const double f = highlighted[name];
    if (!(f > kMinDefectHighlight && f < kMaxDefectHighlight)) {
      std::cerr << "[FAIL] defect overlay " << name << " highlighted fraction " << f
                << " (expected " << kMinDefectHighlight << "-" << kMaxDefectHighlight << ")\n";
      rc = 1;
    }
  }
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
    std::cout << "[OK] all " << held_out_names.size() << " held-out normal images pass with "
              << "unhighlighted overlays, all " << defect_names.size()
              << " defect images fail with localized highlights\n";
  }
  return rc;
}

// Cross-language regression: a memory bank calibrated by Python must load
// and separate defect from nominal correctly when scored by C++ (both use
// OpenCV's batchDistance for the nearest-neighbor search and must agree on the
// same bank).
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

// A folder whose images differ in size must calibrate and score every image,
// with each overlay at its input image's size.
int run_mixed_image_sizes_calibrate_and_score(const std::string& binary,
                                              const std::string& model_path) {
  const fs::path nominal_dir = "assets/datasets/patchcore/nominal";
  std::vector<fs::path> nominal;
  if (fs::exists(nominal_dir)) {
    for (const auto& entry : fs::directory_iterator(nominal_dir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".png") {
        nominal.push_back(entry.path());
      }
    }
  }
  std::sort(nominal.begin(), nominal.end());
  if (nominal.size() < 4) {
    return skip_or_fail("need 4 nominal images under " + nominal_dir.string());
  }

  auto out_dir = create_test_output_dir("patchcore-anomaly-detector",
                                        "test_mixed_image_sizes_calibrate_and_score");
  if (out_dir.empty())
    return 1;
  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path mixed_dir = run_dir / "mixed";
  fs::create_directories(mixed_dir);
  const std::vector<cv::Size> sizes = {{0, 0}, {448, 448}, {320, 240}, {0, 0}}; // 0 keeps 224x224
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    cv::Mat image = cv::imread(nominal[i].string(), cv::IMREAD_COLOR);
    if (sizes[i].width > 0) {
      cv::resize(image, image, sizes[i], 0, 0, cv::INTER_CUBIC);
    }
    cv::imwrite((mixed_dir / nominal[i].filename()).string(), image);
  }

  const fs::path config_path = run_dir / "config.yaml";
  write_e2e_config("patchcore-anomaly-detector", config_path,
                   {
                       {"model.path", model_path},
                       {"source.type", "image_dir"},
                       {"source.image_dir", mixed_dir.string()},
                       {"calibration.nominal_images_dir", mixed_dir.string()},
                       {"calibration.threshold_images_dir", mixed_dir.string()},
                       {"memory_bank.path", (run_dir / "memory_bank.npy").string()},
                       {"memory_bank.meta_path", (run_dir / "bank_meta.json").string()},
                       {"output.dir", out_dir},
                   });
  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  int rc = 0;
  for (const auto& args : std::vector<std::vector<std::string>>{
           {"--calibrate", "--config", config_path.string()}, {"--config", config_path.string()}}) {
    auto result = spawn_and_wait(binary, args, timeout);
    if (result.exit_code != 0) {
      std::cerr << "[FAIL] mixed sizes: " << args[0] << " exited with code " << result.exit_code
                << "\nstderr:\n"
                << result.stderr_text << "\n";
      rc = 1;
      break;
    }
  }
  for (std::size_t i = 0; rc == 0 && i < sizes.size(); ++i) {
    const auto name = nominal[i].filename();
    const cv::Mat src = cv::imread((mixed_dir / name).string(), cv::IMREAD_COLOR);
    const cv::Mat out = cv::imread((fs::path(out_dir) / name).string(), cv::IMREAD_COLOR);
    if (out.empty() || out.size() != src.size()) {
      std::cerr << "[FAIL] mixed sizes: overlay for " << name << " missing or not at input size\n";
      rc = 1;
    }
  }
  if (rc == 0) {
    std::cout << "[OK] mixed-size folder calibrates and scores with overlays at input size\n";
  }
  remove_dir(out_dir);
  return rc;
}

// Runs --calibrate on the bundled nominal set, then one live run with the given source
// settings. Returns the live run's stdout, or nullopt after printing a failure.
std::optional<std::string>
calibrate_and_run_live(const std::string& binary, const std::string& model_path,
                       const std::string& test_name,
                       const std::vector<std::pair<std::string, std::string>>& source, int frames,
                       int save_every, std::string& out_dir) {
  out_dir = create_test_output_dir("patchcore-anomaly-detector", test_name);
  if (out_dir.empty()) {
    return std::nullopt;
  }
  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const char* port = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT");
  ConfigScalars values = {
      {"model.path", model_path},
      {"calibration.nominal_images_dir", "assets/datasets/patchcore/nominal"},
      {"calibration.threshold_images_dir", ""},
      {"memory_bank.path", (run_dir / "memory_bank.npy").string()},
      {"memory_bank.meta_path", (run_dir / "bank_meta.json").string()},
      {"output.dir", out_dir},
      {"output.save_every", std::to_string(save_every)},
      {"output.insight.host", "127.0.0.1"},
      {"output.insight.video_port", port && *port ? port : "9000"},
      {"runtime.frames", std::to_string(frames)},
  };
  values.insert(source.begin(), source.end());
  write_e2e_config("patchcore-anomaly-detector", config_path, values);

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  std::string live_stdout;
  for (const auto& args : std::vector<std::vector<std::string>>{
           {"--calibrate", "--config", config_path.string()}, {"--config", config_path.string()}}) {
    auto result = spawn_and_wait(binary, args, timeout);
    if (result.exit_code != 0) {
      std::cerr << "[FAIL] " << test_name << ": " << args[0] << " exited with code "
                << result.exit_code << "\nstderr:\n"
                << result.stderr_text << "\n";
      remove_dir(out_dir);
      return std::nullopt;
    }
    live_stdout = result.stdout_text;
  }
  return live_stdout;
}

// Frame number -> verdict for every "frame=N: score=... verdict=..." line.
std::map<int, std::string> frame_verdicts(const std::string& text) {
  static const std::regex kFrameRe(R"(frame=(\d+): score=[-\d.]+ threshold=[-\d.]+ verdict=(\w+))");
  std::map<int, std::string> verdicts;
  for (std::sregex_iterator it(text.begin(), text.end(), kFrameRe), end; it != end; ++it) {
    verdicts[std::stoi((*it)[1].str())] = (*it)[2].str();
  }
  return verdicts;
}

// Empty when out_dir holds `expected` decodable frame_*.jpg snapshots of one size
// (`size` if non-empty); otherwise the problem.
std::string snapshots_problem(const std::string& out_dir, int expected, cv::Size size) {
  std::vector<fs::path> snapshots;
  for (const auto& entry : fs::directory_iterator(out_dir)) {
    const auto name = entry.path().filename().string();
    if (name.rfind("frame_", 0) == 0 && entry.path().extension() == ".jpg") {
      snapshots.push_back(entry.path());
    }
  }
  if (static_cast<int>(snapshots.size()) != expected) {
    return "expected " + std::to_string(expected) + " snapshots, found " +
           std::to_string(snapshots.size());
  }
  for (const auto& path : snapshots) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      return path.filename().string() + " does not decode";
    }
    if (size.empty()) {
      size = image.size();
    }
    if (image.size() != size) {
      return path.filename().string() + " has an unexpected size";
    }
  }
  return "";
}

// video_file streams to Insight and saves snapshots, with the verdicts the bundled clip
// should get: its first 60 frames are plain texture, the rest carry defects.
int run_video_file_live_run(const std::string& binary, const std::string& model_path) {
  const fs::path video = "assets/datasets/patchcore/video/test.mp4";
  if (!fs::exists(video)) {
    return skip_or_fail("demo video missing: " + video.string());
  }
  std::string out_dir;
  const auto stdout_text = calibrate_and_run_live(
      binary, model_path, "test_video_file_live_run",
      {{"source.type", "video_file"}, {"source.video_path", video.string()}}, 120, 40, out_dir);
  if (!stdout_text) {
    return 1;
  }
  const auto verdicts = frame_verdicts(*stdout_text);
  std::string problem;
  if (verdicts.size() != 120 || verdicts.begin()->first != 1 || verdicts.rbegin()->first != 120) {
    problem = "expected verdicts for frames 1-120, got " + std::to_string(verdicts.size());
  }
  for (int f = 1; problem.empty() && f <= 50; ++f) {
    if (verdicts.at(f) != "normal") {
      problem = "plain frame " + std::to_string(f) + " flagged ANOMALOUS";
    }
  }
  for (int f = 71; problem.empty() && f <= 120; ++f) {
    if (verdicts.at(f) != "ANOMALOUS") {
      problem = "defect frame " + std::to_string(f) + " not flagged";
    }
  }
  if (problem.empty()) {
    problem = snapshots_problem(out_dir, 3, cv::Size(224, 224));
  }
  remove_dir(out_dir);
  if (!problem.empty()) {
    std::cerr << "[FAIL] video_file live run: " << problem << "\n";
    return 1;
  }
  std::cout << "[OK] video_file: plain frames normal, defect frames ANOMALOUS, 3 snapshots\n";
  return 0;
}

// rtsp runs the model inside the graph, scores every joined frame, streams to Insight,
// and saves snapshots at the stream's size.
int run_rtsp_live_run(const std::string& binary, const std::string& model_path) {
  const auto urls = rtsp_h264_urls_from_env();
  if (urls.empty()) {
    return skip_or_fail("SIMANEAT_TEST_RTSP_H264_URL is not set");
  }
  std::string out_dir;
  const auto stdout_text = calibrate_and_run_live(
      binary, model_path, "test_rtsp_live_run",
      {{"source.type", "rtsp"}, {"source.rtsp.url", urls.front()}}, 60, 20, out_dir);
  if (!stdout_text) {
    return 1;
  }
  const auto verdicts = frame_verdicts(*stdout_text);
  std::string problem;
  if (verdicts.size() != 60 || verdicts.begin()->first != 1 || verdicts.rbegin()->first != 60) {
    problem = "expected verdicts for frames 1-60, got " + std::to_string(verdicts.size());
  }
  if (problem.empty()) {
    problem = snapshots_problem(out_dir, 3, cv::Size());
  }
  remove_dir(out_dir);
  if (!problem.empty()) {
    std::cerr << "[FAIL] rtsp live run: " << problem << "\n";
    return 1;
  }
  std::cout << "[OK] rtsp: 60 frames scored in the graph, 3 snapshots\n";
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
  failures += run_mixed_image_sizes_calibrate_and_score(binary, model_path) != 0;
  failures += run_video_file_live_run(binary, model_path) != 0;
  failures += run_rtsp_live_run(binary, model_path) != 0;

  return failures > 0 ? 1 : 0;
}
