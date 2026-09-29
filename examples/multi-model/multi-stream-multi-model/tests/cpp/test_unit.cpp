#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr const char* kFourStreams = "streams:\n"
                                     "  - url: rtsp://127.0.0.1:8554/src1\n"
                                     "    task: detection\n"
                                     "    decode: yolov8\n"
                                     "    model: models/yolo_11s_mpk.tar.gz\n"
                                     "  - url: rtsp://127.0.0.1:8554/src2\n"
                                     "    task: segmentation\n"
                                     "    decode: yolov8\n"
                                     "    model: models/yolo_11s_seg_mpk.tar.gz\n"
                                     "  - url: rtsp://127.0.0.1:8554/src3\n"
                                     "    task: pose\n"
                                     "    decode: yolo26\n"
                                     "    model: models/yolo26m-pose-int8-b1.tar.gz\n"
                                     "  - url: rtsp://127.0.0.1:8554/src4\n"
                                     "    task: detection\n"
                                     "    decode: yolo26\n"
                                     "    model: models/yolo26m-det-int8-b1.tar.gz\n";

constexpr const char* kOneStream = "streams:\n"
                                   "  - url: rtsp://127.0.0.1:8554/src1\n"
                                   "    task: detection\n"
                                   "    decode: yolo26\n"
                                   "    model: models/yolo26m-det-int8-b1.tar.gz\n";

constexpr const char* kInsightOutput = "output:\n"
                                       "  insight:\n"
                                       "    host: 127.0.0.1\n";

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
  const std::string temp_dir = create_test_scratch_dir("multi-stream-multi-model", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

sima_examples::testing::ProcessResult validate(const std::string& binary,
                                               const fs::path& config_path) {
  return spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"},
                        20000);
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_config_only_reports_every_stream_task(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_reports_every_stream_task",
                                            std::string(kFourStreams) +
                                                "input:\n"
                                                "  codec: avc\n"
                                                "inference:\n"
                                                "  max_inflight_per_stream: 3\n" +
                                                kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok =
      expect_true(result.exit_code == 0, "four-stream multi-model config validates") &&
      expect_contains(result.stdout_text, "streams=4", "validate output reports stream count") &&
      expect_contains(result.stdout_text, "tasks=detection,segmentation,pose,detection",
                      "validate output reports the per-stream tasks in order") &&
      expect_contains(result.stdout_text, "max_inflight_per_stream=3",
                      "validate output reports per-stream inflight limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_too_many_streams",
                                            std::string(kFourStreams) +
                                                "  - url: rtsp://127.0.0.1:8554/src5\n"
                                                "    task: detection\n"
                                                "    decode: yolo26\n"
                                                "    model: models/yolo26m-det-int8-b1.tar.gz\n" +
                                                kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok = expect_true(result.exit_code == 1, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to four streams",
                                  "too-many-stream error mentions the four-stream limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_empty_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_empty_streams",
                                            std::string("streams: []\n") + kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok =
      expect_true(result.exit_code == 1, "empty stream config is rejected") &&
      expect_contains(result.stderr_text, "streams", "empty-stream error mentions streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_unknown_task(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_unknown_task",
                   std::string("streams:\n"
                               "  - url: rtsp://127.0.0.1:8554/src1\n"
                               "    task: tracking\n"
                               "    model: models/yolo26m-det-int8-b1.tar.gz\n") +
                       kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok = expect_true(result.exit_code == 1, "unknown task is rejected") &&
                  expect_contains(result.stderr_text, "task must be one of",
                                  "unknown-task error lists the supported tasks");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_unknown_decode_family(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_unknown_decode_family",
                   std::string("streams:\n"
                               "  - url: rtsp://127.0.0.1:8554/src1\n"
                               "    task: detection\n"
                               "    decode: yolov5\n"
                               "    model: models/yolo26m-det-int8-b1.tar.gz\n") +
                       kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok = expect_true(result.exit_code == 1, "unknown decode family is rejected") &&
                  expect_contains(result.stderr_text, "decode must be one of",
                                  "unknown-decode error lists the supported head layouts");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_stream_without_model(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_stream_without_model",
                   std::string("streams:\n"
                               "  - url: rtsp://127.0.0.1:8554/src1\n"
                               "    task: detection\n") +
                       kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok = expect_true(result.exit_code == 1, "stream without a model is rejected") &&
                  expect_contains(result.stderr_text, "streams[0].model must be set",
                                  "missing-model error names the offending stream");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_inflight_limit(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_inflight_limit",
                   std::string(kOneStream) +
                       "inference:\n"
                       "  max_inflight_per_stream: 0\n" +
                       kInsightOutput);

  const auto result = validate(binary, config_path);
  const bool ok = expect_true(result.exit_code == 1, "invalid inflight limit is rejected") &&
                  expect_contains(result.stderr_text, "max_inflight_per_stream must be -1 or > 0",
                                  "invalid inflight error names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_mask_threshold(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_mask_threshold",
                   std::string(kOneStream) + "output:\n"
                                             "  mask_threshold: 1.5\n"
                                             "  insight:\n"
                                             "    host: 127.0.0.1\n");

  const auto result = validate(binary, config_path);
  const bool ok = expect_true(result.exit_code == 1, "out-of-range mask threshold is rejected") &&
                  expect_contains(result.stderr_text, "mask_threshold must be between 0 and 1",
                                  "invalid mask threshold error names the setting");
  remove_dir(config_path.parent_path().string());
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
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_config_only_reports_every_stream_task(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_validate_config_only_rejects_unknown_task(binary);
  ok &= test_validate_config_only_rejects_unknown_decode_family(binary);
  ok &= test_validate_config_only_rejects_stream_without_model(binary);
  ok &= test_validate_config_only_rejects_invalid_inflight_limit(binary);
  ok &= test_validate_config_only_rejects_invalid_mask_threshold(binary);
  return ok ? 0 : 1;
}
