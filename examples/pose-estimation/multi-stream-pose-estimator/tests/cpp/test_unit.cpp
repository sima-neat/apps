#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_scratch_config;

namespace {

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

bool test_validate_config_only_accepts_four_streams(const std::string& binary) {
  const fs::path config_path = write_scratch_config("multi-stream-pose-estimator", "test_validate_config_only_accepts_four_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-pose-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "input:\n"
                                            "  codec: avc\n"
                                            "  max_width: 2560\n"
                                            "  max_height: 1440\n"
                                            "inference:\n"
                                            "  max_inflight_per_stream: 3\n"
                                            "  max_inflight_total: 12\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "four-stream config validates") &&
      expect_contains(result.stdout_text, "streams=4", "validate output reports stream count") &&
      expect_contains(result.stdout_text, "max_inflight_per_stream=3",
                      "validate output reports per-stream inflight limit") &&
      expect_contains(result.stdout_text, "max_inflight_total=12",
                      "validate output reports total inflight limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path = write_scratch_config("multi-stream-pose-estimator", "test_validate_config_only_rejects_too_many_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-pose-int8-b1.tar.gz\n"
                                            "streams:\n"
                                            "  - rtsp://127.0.0.1:8554/src1\n"
                                            "  - rtsp://127.0.0.1:8554/src2\n"
                                            "  - rtsp://127.0.0.1:8554/src3\n"
                                            "  - rtsp://127.0.0.1:8554/src4\n"
                                            "  - rtsp://127.0.0.1:8554/src5\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to four streams",
                                  "too-many-stream error mentions four-stream phase limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_empty_streams(const std::string& binary) {
  const fs::path config_path = write_scratch_config("multi-stream-pose-estimator", "test_validate_config_only_rejects_empty_streams",
                                            "model:\n"
                                            "  path: models/yolo26m-pose-int8-b1.tar.gz\n"
                                            "streams: []\n"
                                            "output:\n"
                                            "  insight:\n"
                                            "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "empty stream config is rejected") &&
      expect_contains(result.stderr_text, "streams", "empty-stream error mentions streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_inflight_limit(const std::string& binary) {
  const fs::path config_path =
      write_scratch_config("multi-stream-pose-estimator", "test_validate_config_only_rejects_invalid_inflight_limit",
                   "model:\n"
                   "  path: models/yolo26m-pose-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "inference:\n"
                   "  max_inflight_per_stream: 0\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "invalid inflight limit is rejected") &&
                  expect_contains(result.stderr_text, "max_inflight_per_stream must be -1 or > 0",
                                  "invalid inflight error names the setting");
  remove_dir(config_path.parent_path().string());
  return ok;
}

// ---------------------------------------------------------------------------
// Configuration rules --validate-config-only can reach without a model or a
// stream (Refs #526). Each case is the minimal valid config with exactly one
// value broken, and asserts the message that names the rule, so a failure
// says which rule stopped firing.
// ---------------------------------------------------------------------------
struct ConfigParts {
  std::string sections;      // extra top-level sections, e.g. "input:\n  latency_ms: -1\n"
  std::string output_extra;  // extra keys under output:, e.g. "  save_every: -1\n"
  std::string insight_extra; // extra keys under output.insight:, e.g. "    video_port_base: 0\n"
  std::string model_path = "models/model.tar.gz";
  std::string host = "127.0.0.1";
};

struct RejectedConfig {
  const char* name;
  ConfigParts parts;
  const char* message;
};

std::string config_body(const ConfigParts& parts) {
  return "model:\n  path: '" + parts.model_path + "'\n" +
         std::string("streams:\n  - rtsp://127.0.0.1:8554/src1\n") + parts.sections + "output:\n" +
         parts.output_extra + "  insight:\n    host: '" + parts.host + "'\n" + parts.insight_extra;
}

bool test_configuration_rules_are_enforced(const std::string& binary) {
  const std::vector<RejectedConfig> cases = {
      {"model-path-empty", {"", "", "", ""}, "model.path must be set"},
      {"insight-host-empty",
       {"", "", "", "models/model.tar.gz", ""},
       "output.insight.host must be set"},
      {"codec-unsupported",
       {"input:\n  codec: vp9\n", "", ""},
       "input.codec must be h264/avc or h265/hevc"},
      {"latency-negative", {"input:\n  latency_ms: -1\n", "", ""}, "input.latency_ms must be >= 0"},
      {"max-width-zero", {"input:\n  max_width: 0\n", "", ""}, "input.max_width must be > 0"},
      {"max-height-zero", {"input:\n  max_height: 0\n", "", ""}, "input.max_height must be > 0"},
      {"frames-negative", {"inference:\n  frames: -1\n", "", ""}, "inference.frames must be >= 0"},
      {"min-score-above",
       {"inference:\n  min_score: 1.5\n", "", ""},
       "inference.min_score must be between 0 and 1"},
      {"nms-below",
       {"inference:\n  nms_iou: -0.5\n", "", ""},
       "inference.nms_iou must be between 0 and 1"},
      {"max-poses-zero",
       {"inference:\n  max_poses: 0\n", "", ""},
       "inference.max_poses must be > 0"},
      {"keypoint-visibility-above",
       {"", "  min_keypoint_visibility: 2\n", ""},
       "output.min_keypoint_visibility must be between 0 and 1"},
      {"warmup-negative",
       {"runtime:\n  warmup_frames: -1\n", "", ""},
       "runtime.warmup_frames must be >= 0"},
      {"video-port-base-zero",
       {"", "", "    video_port_base: 0\n"},
       "output.insight.video_port_base must be > 0"},
      {"metadata-port-base-zero",
       {"", "", "    metadata_port_base: 0\n"},
       "output.insight.metadata_port_base must be > 0"},
      {"save-every-negative", {"", "  save_every: -1\n", ""}, "output.save_every must be >= 0"},
  };

  bool ok = true;
  for (const RejectedConfig& c : cases) {
    const fs::path config_path = write_scratch_config("multi-stream-pose-estimator", std::string("rule_") + c.name, config_body(c.parts));
    const auto result =
        spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
    ok &= expect_true(result.exit_code != 0, std::string(c.name) + " is rejected") &&
          expect_contains(result.stderr_text, c.message, std::string(c.name) + " names its rule");
    remove_dir(config_path.parent_path().string());
  }

  // The control: the same minimal config with nothing broken validates, so the
  // rejections above are about the broken value and not about the baseline.
  const fs::path config_path = write_scratch_config("multi-stream-pose-estimator", "rule_baseline", config_body(ConfigParts{}));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(result.exit_code == 0, "minimal config validates") &&
        expect_contains(result.stdout_text, "Config validated", "validated line is printed");
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
  ok &= test_validate_config_only_accepts_four_streams(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_validate_config_only_rejects_invalid_inflight_limit(binary);
  ok &= test_configuration_rules_are_enforced(binary);
  return ok ? 0 : 1;
}
