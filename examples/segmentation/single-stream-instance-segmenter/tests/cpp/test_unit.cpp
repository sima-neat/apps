// Unit test for single-stream-instance-segmenter: CLI handling and the
// configuration rules --validate-config-only can reach without a model or a
// stream.
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr const char* kExampleName = "single-stream-instance-segmenter";

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

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& message) {
  return expect_true(haystack.find(needle) == std::string::npos, message);
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir(kExampleName, test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

// Runs --validate-config-only on `body` and removes the scratch directory.
ProcessResult validate(const std::string& binary, const std::string& test_name,
                       const std::string& body) {
  const fs::path config_path = write_config(test_name, body);
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  return result;
}

// The smallest config the validator accepts, with the source and model lines
// supplied by the caller so each test states exactly the keys it is about.
// Everything else takes its documented default.
std::string minimal_config(const std::string& source_lines, const std::string& model_lines = "") {
  return "model:\n"
         "  path: models/model.tar.gz\n" +
         model_lines + "source:\n" + source_lines +
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
}

bool test_help_runs(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(r.exit_code == 0, "help exits with code 0") &&
         expect_contains(r.stdout_text, "Usage", "help prints usage");
}

bool test_unknown_flag_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(r.exit_code != 0, "unknown flag exits non-zero");
}

bool test_missing_config_file_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
  return expect_true(r.exit_code != 0, "missing config exits non-zero");
}

// config.yaml ships source.url present and documents source.rtsp_url as
// "used when source.url is empty", so the empty value has to fall through.
bool test_empty_url_falls_back_to_the_legacy_key(const std::string& binary) {
  const auto r = validate(binary, "empty_url_falls_back",
                          minimal_config("  url: \"\"\n"
                                         "  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "empty url with a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=rtsp://127.0.0.1:8554/legacy",
                         "empty url selects the legacy rtsp_url");
}

bool test_absent_url_falls_back_to_the_legacy_key(const std::string& binary) {
  const auto r = validate(binary, "absent_url_falls_back",
                          minimal_config("  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "absent url with a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=rtsp://127.0.0.1:8554/legacy",
                         "absent url selects the legacy rtsp_url");
}

bool test_present_url_wins_over_the_legacy_key(const std::string& binary) {
  const auto r = validate(binary, "present_url_wins",
                          minimal_config("  url: rtsp://127.0.0.1:8554/src1\n"
                                         "  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "url beside a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=rtsp://127.0.0.1:8554/src1",
                         "present url is selected") &&
         expect_not_contains(r.stdout_text, "legacy", "legacy rtsp_url is not selected");
}

bool test_both_urls_empty_is_rejected(const std::string& binary) {
  const auto r = validate(binary, "both_urls_empty",
                          minimal_config("  url: \"\"\n"
                                         "  rtsp_url: \"\"\n"));
  return expect_true(r.exit_code != 0, "empty url and empty rtsp_url is rejected") &&
         expect_contains(r.stderr_text, "source.url or source.rtsp_url must be set",
                         "both-empty error names both keys");
}

// The Python loader once accepted this because Path("") is "."; the C++ side
// is held to the rule so the two implementations cannot drift apart.
bool test_empty_labels_is_rejected(const std::string& binary) {
  const auto r =
      validate(binary, "empty_labels",
               minimal_config("  url: rtsp://127.0.0.1:8554/src1\n", "  labels: \"\"\n"));
  return expect_true(r.exit_code != 0, "empty model.labels is rejected") &&
         expect_contains(r.stderr_text, "model.labels must be set",
                         "empty labels error names the key");
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
  ok &= test_unknown_flag_fails(binary);
  ok &= test_missing_config_file_fails(binary);
  ok &= test_empty_url_falls_back_to_the_legacy_key(binary);
  ok &= test_absent_url_falls_back_to_the_legacy_key(binary);
  ok &= test_present_url_wins_over_the_legacy_key(binary);
  ok &= test_both_urls_empty_is_rejected(binary);
  ok &= test_empty_labels_is_rejected(binary);
  return ok ? 0 : 1;
}
