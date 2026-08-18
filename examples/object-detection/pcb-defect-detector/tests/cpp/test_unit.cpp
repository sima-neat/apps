// Unit test for pcb-defect-detector: CLI handling and configuration validation.
// Runs without a model, a board, or input images.
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr char kShippedConfig[] =
    "examples/object-detection/pcb-defect-detector/src/common/config.yaml";
constexpr char kShippedLabels[] =
    "examples/object-detection/pcb-defect-detector/src/common/pcb_label.txt";

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir("pcb-defect-detector", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

// A minimal valid config with one overridden line appended.
std::string config_body(const std::string& extra_decode = "", const std::string& extra_runtime = "",
                        const std::string& labels = kShippedLabels) {
  return std::string("model:\n")
      .append("  path: models/yolo26n_plc_mpk.tar.gz\n")
      .append("  labels: " + labels + "\n")
      .append("io:\n")
      .append("  input_dir: assets/datasets/pcb\n")
      .append("  output_dir: sandbox/pcb-defect-detector\n")
      .append("decode:\n")
      .append("  score_threshold: 0.25\n")
      .append("  nms_iou: 0.45\n")
      .append("  max_detections: 300\n")
      .append(extra_decode)
      .append("runtime:\n")
      .append("  timeout_ms: 8000\n")
      .append("  queue_depth: 8\n")
      .append(extra_runtime);
}

bool validate_rejects(const std::string& binary, const std::string& test_name,
                      const std::string& body, const std::string& expected_error,
                      const std::string& message) {
  const fs::path config_path = write_config(test_name, body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, message + " is rejected") &&
      expect_contains(result.stderr_text, expected_error, message + " names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "Usage", "help prints usage") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_unknown_flag_is_rejected(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(result.exit_code != 0, "unknown flag exits nonzero") &&
         expect_contains(result.stderr_text, "unknown argument",
                         "unknown flag error names the argument");
}

bool test_missing_config_value_is_rejected(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config"}, 20000);
  return expect_true(result.exit_code == 1, "bare --config exits with code 1") &&
         expect_contains(result.stderr_text, "--config requires a path",
                         "bare --config explains the missing value");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "/nonexistent/pcb-config.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_non_numeric_score_is_rejected(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--score", "high"}, 20000);
  return expect_true(result.exit_code == 1, "non-numeric --score exits with code 1") &&
         expect_contains(result.stderr_text, "--score requires a number",
                         "non-numeric --score explains the expected value");
}

// The harness writes process logs next to the --config file, so the shipped config is
// validated through a scratch copy rather than in place.
bool test_shipped_config_validates(const std::string& binary) {
  std::ifstream shipped(kShippedConfig);
  if (!shipped.good()) {
    std::cerr << "[FAIL] shipped config is missing: " << kShippedConfig << "\n";
    return false;
  }
  const std::string body((std::istreambuf_iterator<char>(shipped)),
                         std::istreambuf_iterator<char>());

  const fs::path config_path = write_config("test_shipped_config_validates", body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "shipped config validates") &&
                  expect_contains(result.stdout_text, "classes=6",
                                  "validate output reports the six defect classes") &&
                  expect_contains(result.stdout_text, "input_size=640",
                                  "validate output reports the model input size") &&
                  expect_contains(result.stdout_text, "max_detections=300",
                                  "validate output reports the decode cap") &&
                  expect_contains(result.stdout_text, "configuration OK",
                                  "validate output confirms the configuration");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_cli_overrides_apply(const std::string& binary) {
  const fs::path config_path = write_config("test_cli_overrides_apply", config_body());
  const auto result = spawn_and_wait(binary,
                                     {"--config", config_path.string(), "--score", "0.40", "--nms",
                                      "0.55", "--validate-config-only"},
                                     20000);
  const bool ok = expect_true(result.exit_code == 0, "config with CLI overrides validates") &&
                  expect_contains(result.stdout_text, "score_threshold=0.40",
                                  "--score override reaches the resolved config") &&
                  expect_contains(result.stdout_text, "nms_iou=0.55",
                                  "--nms override reaches the resolved config");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_out_of_range_score_override_is_rejected(const std::string& binary) {
  const fs::path config_path =
      write_config("test_out_of_range_score_override_is_rejected", config_body());
  const auto result = spawn_and_wait(
      binary, {"--config", config_path.string(), "--score", "1.5", "--validate-config-only"},
      20000);
  const bool ok = expect_true(result.exit_code == 1, "out-of-range --score override is rejected") &&
                  expect_contains(result.stderr_text, "decode.score_threshold",
                                  "override error names decode.score_threshold");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_zero_input_size_is_rejected(const std::string& binary) {
  const fs::path config_path =
      write_config("test_zero_input_size_is_rejected",
                   std::string("model:\n")
                       .append("  path: models/yolo26n_plc_mpk.tar.gz\n")
                       .append("  labels: " + std::string(kShippedLabels) + "\n")
                       .append("  input_size: 0\n")
                       .append("io:\n")
                       .append("  input_dir: assets/datasets/pcb\n"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "zero model.input_size is rejected") &&
                  expect_contains(result.stderr_text, "model.input_size",
                                  "input size error names model.input_size");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_missing_model_path_is_rejected(const std::string& binary) {
  const fs::path config_path =
      write_config("test_missing_model_path_is_rejected",
                   std::string("model:\n")
                       .append("  labels: " + std::string(kShippedLabels) + "\n")
                       .append("io:\n")
                       .append("  input_dir: assets/datasets/pcb\n"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "config without model.path is rejected") &&
      expect_contains(result.stderr_text, "model.path", "missing model error names model.path");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_missing_labels_file_is_rejected(const std::string& binary) {
  const fs::path config_path = write_config("test_missing_labels_file_is_rejected",
                                            config_body("", "", "/nonexistent/pcb_label.txt"));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "config with a missing labels file is rejected") &&
      expect_contains(result.stderr_text, "labels file does not exist",
                      "missing labels error names the labels file");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_empty_labels_file_is_rejected(const std::string& binary) {
  const std::string temp_dir =
      create_test_scratch_dir("pcb-defect-detector", "test_empty_labels_file_is_rejected");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] failed to create temp directory\n";
    return false;
  }
  const fs::path labels_path = fs::path(temp_dir) / "empty_label.txt";
  std::ofstream(labels_path) << "\n \n";

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream(config_path) << config_body("", "", labels_path.string());

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "config with an empty labels file is rejected") &&
      expect_contains(result.stderr_text, "labels file is empty",
                      "empty labels error names the labels file");
  remove_dir(temp_dir);
  return ok;
}

bool test_missing_input_dir_is_reported(const std::string& binary) {
  const fs::path config_path =
      write_config("test_missing_input_dir_is_reported",
                   std::string("model:\n")
                       .append("  path: models/yolo26n_plc_mpk.tar.gz\n")
                       .append("  labels: " + std::string(kShippedLabels) + "\n")
                       .append("io:\n")
                       .append("  input_dir: /nonexistent/pcb-images\n"));
  const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
  const bool ok = expect_true(result.exit_code == 2, "missing input directory exits with code 2") &&
                  expect_contains(result.stderr_text, "Input directory does not exist",
                                  "missing input directory is reported before model load");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_empty_input_dir_is_reported(const std::string& binary) {
  const std::string temp_dir =
      create_test_scratch_dir("pcb-defect-detector", "test_empty_input_dir_is_reported");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] failed to create temp directory\n";
    return false;
  }
  const fs::path empty_input = fs::path(temp_dir) / "images";
  fs::create_directories(empty_input);

  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream(config_path) << std::string("model:\n")
                                    .append("  path: models/yolo26n_plc_mpk.tar.gz\n")
                                    .append("  labels: " + std::string(kShippedLabels) + "\n")
                                    .append("io:\n")
                                    .append("  input_dir: " + empty_input.string() + "\n");

  const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
  const bool ok = expect_true(result.exit_code == 3, "empty input directory exits with code 3") &&
                  expect_contains(result.stderr_text, "No images found",
                                  "empty input directory is reported before model load");
  remove_dir(temp_dir);
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_unknown_flag_is_rejected(binary);
  ok &= test_missing_config_value_is_rejected(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_non_numeric_score_is_rejected(binary);
  ok &= test_shipped_config_validates(binary);
  ok &= test_cli_overrides_apply(binary);
  ok &= test_out_of_range_score_override_is_rejected(binary);
  ok &= test_zero_input_size_is_rejected(binary);
  ok &= test_missing_model_path_is_rejected(binary);
  ok &= test_missing_labels_file_is_rejected(binary);
  ok &= test_empty_labels_file_is_rejected(binary);
  ok &= test_missing_input_dir_is_reported(binary);
  ok &= test_empty_input_dir_is_reported(binary);

  ok &= validate_rejects(binary, "test_rejects_out_of_range_nms", config_body("  nms_iou: 1.20\n"),
                         "decode.nms_iou", "out-of-range decode.nms_iou");
  ok &= validate_rejects(binary, "test_rejects_zero_max_detections",
                         config_body("  max_detections: 0\n"), "decode.max_detections",
                         "zero decode.max_detections");
  ok &= validate_rejects(binary, "test_rejects_zero_timeout", config_body("", "  timeout_ms: 0\n"),
                         "runtime.timeout_ms", "zero runtime.timeout_ms");
  ok &= validate_rejects(binary, "test_rejects_zero_queue_depth",
                         config_body("", "  queue_depth: 0\n"), "runtime.queue_depth",
                         "zero runtime.queue_depth");

  return ok ? 0 : 1;
}
