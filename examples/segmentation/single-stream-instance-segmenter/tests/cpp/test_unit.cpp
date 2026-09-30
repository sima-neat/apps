// Unit test for single-stream-instance-segmenter: CLI handling and the
// configuration rules --validate-config-only can reach without a model or a
// stream.
#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_not_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::validate_config_body;

namespace {

constexpr const char* kExampleName = "single-stream-instance-segmenter";

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
  const auto r = validate_config_body(kExampleName, binary, "empty_url_falls_back",
                          minimal_config("  url: \"\"\n"
                                         "  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "empty url with a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=source.rtsp_url",
                         "empty url selects the legacy rtsp_url");
}

bool test_absent_url_falls_back_to_the_legacy_key(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "absent_url_falls_back",
                          minimal_config("  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "absent url with a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=source.rtsp_url",
                         "absent url selects the legacy rtsp_url");
}

bool test_present_url_wins_over_the_legacy_key(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "present_url_wins",
                          minimal_config("  url: rtsp://127.0.0.1:8554/src1\n"
                                         "  rtsp_url: rtsp://127.0.0.1:8554/legacy\n"));
  return expect_true(r.exit_code == 0, "url beside a legacy rtsp_url validates") &&
         expect_contains(r.stdout_text, "source=source.url", "present url is selected") &&
         expect_not_contains(r.stdout_text, "source.rtsp_url", "legacy rtsp_url is not selected") &&
         expect_not_contains(r.stdout_text, "rtsp://", "validated line does not echo the URL");
}

bool test_both_urls_empty_is_rejected(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "both_urls_empty",
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
      validate_config_body(kExampleName, binary, "empty_labels",
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
