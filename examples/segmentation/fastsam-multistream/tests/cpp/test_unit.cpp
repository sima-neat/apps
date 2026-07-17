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
  const std::string temp_dir = create_test_scratch_dir("fastsam-multistream", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "failed to open config file",
                         "missing config error mentions failed to open config file");
}

bool test_rejects_missing_model_path(const std::string& binary) {
  const fs::path config_path = write_config("test_rejects_missing_model_path",
                                            "source:\n"
                                            "  rtsp_urls:\n"
                                            "    - rtsp://127.0.0.1:8554/src1\n");
  const auto result = spawn_and_wait(binary, {config_path.string()}, 20000);
  const bool ok = expect_true(result.exit_code == 2, "missing model.path is rejected") &&
                  expect_contains(result.stderr_text, "model.path must be set",
                                  "error names model.path");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_rejects_empty_rtsp_urls(const std::string& binary) {
  const fs::path config_path = write_config("test_rejects_empty_rtsp_urls",
                                            "model:\n"
                                            "  path: /models/fastsam.tar.gz\n");
  const auto result = spawn_and_wait(binary, {config_path.string()}, 20000);
  const bool ok =
      expect_true(result.exit_code == 2, "empty rtsp_urls is rejected") &&
      expect_contains(result.stderr_text, "source.rtsp_urls must be a non-empty list of RTSP URLs",
                      "error names source.rtsp_urls");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path = write_config("test_rejects_too_many_streams",
                                            "model:\n"
                                            "  path: /models/fastsam.tar.gz\n"
                                            "source:\n"
                                            "  rtsp_urls:\n"
                                            "    - rtsp://127.0.0.1:8554/src1\n"
                                            "    - rtsp://127.0.0.1:8554/src2\n"
                                            "    - rtsp://127.0.0.1:8554/src3\n"
                                            "    - rtsp://127.0.0.1:8554/src4\n"
                                            "    - rtsp://127.0.0.1:8554/src5\n");
  const auto result = spawn_and_wait(binary, {config_path.string()}, 20000);
  const bool ok = expect_true(result.exit_code == 2, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "this example supports up to four streams",
                                  "error mentions the four-stream limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_rejects_missing_insight_host(const std::string& binary) {
  const fs::path config_path = write_config("test_rejects_missing_insight_host",
                                            "model:\n"
                                            "  path: /models/fastsam.tar.gz\n"
                                            "source:\n"
                                            "  rtsp_urls:\n"
                                            "    - rtsp://127.0.0.1:8554/src1\n");
  const auto result = spawn_and_wait(binary, {config_path.string()}, 20000);
  const bool ok = expect_true(result.exit_code == 2, "missing insight host is rejected") &&
                  expect_contains(result.stderr_text, "output.insight.host must be set",
                                  "error names output.insight.host");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_rejects_missing_prompt_text(const std::string& binary) {
  const fs::path config_path = write_config("test_rejects_missing_prompt_text",
                                            "model:\n"
                                            "  path: /models/fastsam.tar.gz\n"
                                            "source:\n"
                                            "  rtsp_urls:\n"
                                            "    - rtsp://127.0.0.1:8554/src1\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");
  const auto result = spawn_and_wait(binary, {config_path.string()}, 20000);
  const bool ok = expect_true(result.exit_code == 2, "missing prompt text is rejected") &&
                  expect_contains(result.stderr_text, "prompt.text must be set",
                                  "error names prompt.text");
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
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_rejects_missing_model_path(binary);
  ok &= test_rejects_empty_rtsp_urls(binary);
  ok &= test_rejects_too_many_streams(binary);
  ok &= test_rejects_missing_insight_host(binary);
  ok &= test_rejects_missing_prompt_text(binary);
  return ok ? 0 : 1;
}
