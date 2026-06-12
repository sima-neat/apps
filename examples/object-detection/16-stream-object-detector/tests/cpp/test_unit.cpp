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

constexpr const char* kModelPath = "assets/models/yolo26m-det-int8-b1.tar.gz";

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
  const std::string temp_dir = create_test_scratch_dir("16-stream-object-detector", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

std::string stream_entries(int count) {
  std::string out;
  for (int index = 1; index <= count; ++index) {
    out += "  - rtsp://127.0.0.1:8554/src" + std::to_string(index) + "\n";
  }
  return out;
}

std::string model_header(const std::string& decode_type = "") {
  std::string out = std::string("model:\n  path: ") + kModelPath + "\n";
  if (!decode_type.empty())
    out += "  decode_type: " + decode_type + "\n";
  return out;
}

std::string valid_config(int streams, int workers, const std::string& decode_type = "") {
  return model_header(decode_type) + "streams:\n" + stream_entries(streams) +
         "input:\n"
         "  tcp: true\n"
         "  latency_ms: 100\n"
         "inference:\n"
         "  workers: " +
         std::to_string(workers) +
         "\n"
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
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

bool test_validate_config_only_accepts_sixteen_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_accepts_sixteen_streams", valid_config(16, 4));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "sixteen-stream config validates") &&
      expect_contains(result.stdout_text, "streams=16", "validate output reports stream count") &&
      expect_contains(result.stdout_text, "workers=4", "validate output reports worker count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_yolov8_decode_type(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_yolov8_decode_type",
                                            valid_config(4, 2, "yolov8"));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "yolov8 decode type validates");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_too_many_streams", valid_config(17, 4));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "seventeen-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to 16 streams",
                                  "too-many-stream error mentions sixteen-stream limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_worker_count(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_worker_count", valid_config(4, 5));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "worker count greater than stream count is rejected") &&
      expect_contains(result.stderr_text, "workers must be <= stream count",
                      "invalid-worker error mentions stream count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_empty_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_empty_streams",
                                            model_header() + "streams: []\n"
                                                             "output:\n"
                                                             "  insight:\n"
                                                             "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "empty streams config is rejected") &&
                  expect_contains(result.stderr_text, "streams must be a non-empty list",
                                  "empty-streams error mentions streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_fps_scheduler_knob(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_rejects_fps_scheduler_knob",
                                            model_header() + "streams:\n" + stream_entries(4) +
                                                "input:\n"
                                                "  tcp: true\n"
                                                "inference:\n"
                                                "  workers: 2\n"
                                                "  target_fps: 15\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "target_fps config is rejected") &&
                  expect_contains(result.stderr_text, "target_fps is not supported",
                                  "target_fps error tells user to set RTSP source FPS");
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
  ok &= test_validate_config_only_accepts_sixteen_streams(binary);
  ok &= test_validate_config_only_accepts_yolov8_decode_type(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_invalid_worker_count(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_validate_config_only_rejects_fps_scheduler_knob(binary);
  return ok ? 0 : 1;
}
