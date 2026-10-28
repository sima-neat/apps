// Unit test for multi-stream-obb-detector: CLI handling, the configuration rules
// --validate-config-only can reach without a model or a stream, and the Insight objects
// built from oriented boxes, checked against the same tests/fixtures/insight_objects.json
// the Python unit test uses. Both implementations must publish the shared fixture identically.
#define main multi_stream_obb_detector_application_main
#include "../../src/cpp/main.cpp"
#undef main

#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::validate_config_body;

namespace {

constexpr const char* kExampleName = "multi-stream-obb-detector";

std::string config_body(const std::string& streams, const std::string& extra = "") {
  return "model:\n"
         "  path: models/model.tar.gz\n"
         "  labels: dota_label.txt\n"
         "streams:\n" +
         streams +
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n" +
         extra;
}

const std::string kTwoStreams = "  - rtsp://127.0.0.1:8554/src1\n"
                                "  - \"rtsp://127.0.0.1:8554/src2\"  # a quoted entry\n";

bool test_help_runs(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(r.exit_code == 0, "help exits with code 0") &&
         expect_contains(r.stdout_text, "Usage", "help prints usage");
}

bool test_unknown_flag_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(r.exit_code != 0, "unknown flag exits non-zero");
}

bool test_missing_config_file_exits_with_code_2(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
  return expect_true(r.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(r.stderr_text, "config file not found", "missing config is named");
}

bool test_validate_config_only_reports_the_stream_count(const std::string& binary) {
  const auto r =
      validate_config_body(kExampleName, binary, "two_streams", config_body(kTwoStreams));
  return expect_true(r.exit_code == 0, "two streams validate") &&
         expect_contains(r.stdout_text, "streams=2", "stream count is reported");
}

bool test_configuration_rules_are_enforced(const std::string& binary) {
  struct Case {
    const char* name;
    std::string body;
    const char* message;
  };
  const std::vector<Case> cases = {
      {"no_streams", config_body(""), "streams must be a non-empty list"},
      {"blank_stream", config_body("  - rtsp://host/a\n  - \"\"\n"),
       "streams[1] must be a non-empty string"},
      {"five_streams",
       config_body("  - rtsp://host/a\n  - rtsp://host/b\n  - rtsp://host/c\n"
                   "  - rtsp://host/d\n  - rtsp://host/e\n"),
       "streams supports up to four entries"},
      {"bad_codec", config_body(kTwoStreams, "input:\n  codec: vp9\n"),
       "input.codec must be h264/avc or h265/hevc"},
      {"zero_inflight", config_body(kTwoStreams, "inference:\n  max_inflight_per_stream: 0\n"),
       "inference.max_inflight_per_stream must be -1 or > 0"},
      {"min_score_above_one", config_body(kTwoStreams, "inference:\n  min_score: 1.5\n"),
       "inference.min_score must be between 0 and 1"},
      {"zero_max_detections", config_body(kTwoStreams, "inference:\n  max_detections: 0\n"),
       "inference.max_detections must be > 0"},
      {"negative_save_every", config_body(kTwoStreams, "  save_every: -1\n"),
       "output.save_every must be >= 0"},
  };
  bool ok = true;
  for (const auto& c : cases) {
    const auto r =
        validate_config_body(kExampleName, binary, std::string("rule_") + c.name, c.body);
    ok &= expect_true(r.exit_code != 0, std::string(c.name) + " is rejected") &&
          expect_contains(r.stderr_text, c.message, std::string(c.name) + " names its key");
  }
  return ok;
}

std::filesystem::path fixture_path() {
  const char* apps_root = std::getenv("APPS_ROOT");
  const std::filesystem::path root =
      (apps_root && *apps_root) ? std::filesystem::path(apps_root) : std::filesystem::path(".");
  return root / SIMANEAT_APPS_EXAMPLE_SOURCE_DIR / ".." / ".." / "tests" / "fixtures" /
         "insight_objects.json";
}

bool close_to(const nlohmann::json& actual, const nlohmann::json& expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(actual[i].get<double>() - expected[i].get<double>()) > 1e-9) {
      return false;
    }
  }
  return true;
}

bool test_objects_match_the_shared_fixture() {
  std::ifstream input(fixture_path());
  if (!input.good()) {
    std::cerr << "[FAIL] fixture not found: " << fixture_path() << "\n";
    return false;
  }
  const auto fixture = nlohmann::json::parse(input);
  const auto labels = fixture.at("labels").get<std::vector<std::string>>();
  const auto& expected = fixture.at("objects");
  bool ok = expect_true(fixture.at("boxes").size() == expected.size(), "fixture is consistent");
  for (std::size_t i = 0; ok && i < expected.size(); ++i) {
    const auto& spec = fixture.at("boxes")[i];
    const OrientedBox box{spec.at("cx").get<double>(),
                          spec.at("cy").get<double>(),
                          spec.at("width").get<double>(),
                          spec.at("height").get<double>(),
                          spec.at("angle_radians").get<double>(),
                          spec.at("score").get<double>(),
                          spec.at("class_id").get<int>()};
    const auto actual = insight_object(static_cast<int>(i) + 1, box, labels);
    const std::string name = expected[i].at("id").get<std::string>();
    ok &= expect_true(actual.at("id") == expected[i].at("id"), name + " id") &&
          expect_true(actual.at("label") == expected[i].at("label"), name + " label") &&
          expect_true(std::abs(actual.at("confidence").get<double>() -
                               expected[i].at("confidence").get<double>()) < 1e-9,
                      name + " confidence") &&
          expect_true(close_to(actual.at("bbox"), expected[i].at("bbox")), name + " bbox") &&
          expect_true(close_to(actual.at("oriented_bbox"), expected[i].at("oriented_bbox")),
                      name + " oriented_bbox");
  }
  if (ok) {
    std::cout << "[OK] Insight objects match the shared fixture\n";
  }
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  bool ok = test_objects_match_the_shared_fixture();
  ok &= test_help_runs(binary);
  ok &= test_unknown_flag_fails(binary);
  ok &= test_missing_config_file_exits_with_code_2(binary);
  ok &= test_validate_config_only_reports_the_stream_count(binary);
  ok &= test_configuration_rules_are_enforced(binary);
  return ok ? 0 : 1;
}
