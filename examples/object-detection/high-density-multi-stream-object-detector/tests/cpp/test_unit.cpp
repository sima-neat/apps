#include "support/testing/test_process.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr const char* kModelPath = "assets/models/yolo26n-det-bf16-mla_tess-b1.tar.gz";

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
  const std::string temp_dir =
      create_test_scratch_dir("high-density-multi-stream-object-detector", test_name);
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

bool test_validate_config_only_accepts_twenty_four_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_accepts_twenty_four_streams", valid_config(24, 1));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "twenty-four-stream config validates") &&
      expect_contains(result.stdout_text, "streams=24",
                      "validate output reports default stream count") &&
      expect_contains(result.stdout_text, "workers=1", "validate output reports worker count") &&
      expect_contains(result.stdout_text, "queue_depth=4",
                      "validate output reports the proven detector queue default") &&
      expect_contains(result.stdout_text, "internal_queue_depth=1",
                      "validate output reports the bounded global stage queue default") &&
      expect_contains(result.stdout_text, "inference_async=true",
                      "validate output reports nonblocking MLA inference") &&
      expect_contains(result.stdout_text, "max_inflight_per_stream=4",
                      "validate output reports the proven public per-stream credit default") &&
      expect_contains(result.stdout_text, "fan_in_policy=latest",
                      "validate output reports the portable default fan-in policy") &&
      expect_contains(result.stdout_text,
                      (config_path.parent_path() / kModelPath).lexically_normal().string(),
                      "relative model path resolves from the config directory") &&
      expect_contains(result.stdout_text,
                      (config_path.parent_path() / "coco_label.txt").lexically_normal().string(),
                      "relative labels path resolves from the config directory");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_named_profiles(const std::string& binary) {
  struct ProfileExpectation {
    const char* filename;
    int streams;
    int fps;
    int queue_depth;
    int internal_queue_depth;
    int max_inflight_per_stream;
    const char* fan_in_policy;
  };

  constexpr std::array<ProfileExpectation, 3> profiles{{
      {"config.yaml", 16, 25, 16, 1, 1, "latest"},
      {"config-24x720p20fps.yaml", 24, 20, 4, 1, 4, "every_frame"},
      {"config-48x720p10fps.yaml", 48, 10, 1, 2, 1, "latest"},
  }};
  const fs::path common_dir =
      "examples/object-detection/high-density-multi-stream-object-detector/src/common";

  bool ok = true;
  for (const auto& profile : profiles) {
    const fs::path config_path = common_dir / profile.filename;
    const auto result =
        spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
    const std::string label = std::string("named profile ") + profile.filename;
    ok &= expect_true(result.exit_code == 0, label + " validates in the C++ application");
    ok &= expect_contains(result.stdout_text, "streams=" + std::to_string(profile.streams),
                          label + " reports its stream count");
    ok &= expect_contains(result.stdout_text, "input=1280x720@" + std::to_string(profile.fps),
                          label + " reports its input FPS");
    ok &= expect_contains(result.stdout_text, "queue_depth=" + std::to_string(profile.queue_depth),
                          label + " reports its terminal queue depth");
    ok &= expect_contains(result.stdout_text,
                          "internal_queue_depth=" + std::to_string(profile.internal_queue_depth),
                          label + " reports its internal queue depth");
    ok &= expect_contains(result.stdout_text,
                          "max_inflight_per_stream=" +
                              std::to_string(profile.max_inflight_per_stream),
                          label + " reports its per-stream credit");
    ok &= expect_contains(result.stdout_text, std::string("fan_in_policy=") + profile.fan_in_policy,
                          label + " reports its fan-in policy");
    ok &= expect_contains(result.stdout_text,
                          "insight_visible_streams=" + std::to_string(profile.streams),
                          label + " publishes every configured stream");
  }
  return ok;
}

bool test_validate_config_only_accepts_insight_visible_limit(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_accepts_insight_visible_limit",
                   model_header() + "streams:\n" + stream_entries(24) +
                       "input:\n"
                       "  tcp: true\n"
                       "  latency_ms: 100\n"
                       "  skip_rtsp_probe: true\n"
                       "  width: 1280\n"
                       "  height: 720\n"
                       "  fps: 20\n"
                       "inference:\n"
                       "  workers: 1\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n"
                       "    max_visible_streams: 16\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "twenty-four-stream Insight16 config validates") &&
      expect_contains(result.stdout_text, "streams=24",
                      "Insight16 validate output reports stream count") &&
      expect_contains(result.stdout_text, "insight_visible_streams=16",
                      "Insight16 validate output reports visible stream limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_overlapping_insight_ports(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_overlapping_insight_ports",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "inference:\n"
                       "  workers: 1\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n"
                       "    video_port_base: 9000\n"
                       "    metadata_port_base: 9002\n"
                       "    max_visible_streams: 4\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "overlapping Insight port ranges are rejected") &&
      expect_contains(result.stderr_text, "port ranges overlap",
                      "overlap error identifies Insight port ranges");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_forty_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_accepts_forty_streams", valid_config(40, 1));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "forty-stream config validates") &&
                  expect_contains(result.stdout_text, "streams=40",
                                  "validate output reports forty-stream count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_removed_output_paths(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_removed_output_paths",
                   model_header() + "streams:\n" + stream_entries(1) +
                       "inference:\n"
                       "  workers: 1\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n"
                       "  hidden_streams:\n"
                       "    video_sink: dummy\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "removed hidden-stream path is rejected") &&
                  expect_contains(result.stderr_text, "output.hidden_streams was removed",
                                  "removed hidden-stream error explains the replacement");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_insight_visible_limit_above_stream_count(
    const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_insight_visible_limit_above_stream_count",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "input:\n"
                       "  tcp: true\n"
                       "inference:\n"
                       "  workers: 1\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n"
                       "    max_visible_streams: 16\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "oversized Insight visible limit is rejected") &&
      expect_contains(result.stderr_text, "cannot exceed stream count",
                      "oversized Insight visible limit error mentions stream count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_yolov8_decode_type(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_yolov8_decode_type",
                                            valid_config(4, 1, "yolov8"));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "yolov8 decode type validates");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_input_caps(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_input_caps",
                                            model_header() + "streams:\n" + stream_entries(4) +
                                                "input:\n"
                                                "  tcp: true\n"
                                                "  latency_ms: 100\n"
                                                "  decoder_buffers: 8\n"
                                                "  width: 1280\n"
                                                "  height: 720\n"
                                                "  fps: 25\n"
                                                "inference:\n"
                                                "  workers: 1\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "explicit input caps validate");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_accepts_decoder_tuning(const std::string& binary) {
  const fs::path config_path = write_config("test_validate_config_only_accepts_decoder_tuning",
                                            model_header() + "streams:\n" + stream_entries(2) +
                                                "input:\n"
                                                "  tcp: true\n"
                                                "  latency_ms: 100\n"
                                                "  decoder_buffers: 7\n"
                                                "  decoder_input_buffers: 2\n"
                                                "  decoder_tuning: throughput-low-latency\n"
                                                "  skip_rtsp_probe: true\n"
                                                "  width: 3840\n"
                                                "  height: 2160\n"
                                                "  fps: 30\n"
                                                "inference:\n"
                                                "  workers: 1\n"
                                                "output:\n"
                                                "  insight:\n"
                                                "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 0, "decoder tuning config validates") &&
                  expect_contains(result.stdout_text, "decoder_admission=core",
                                  "validate output reports framework-owned decoder admission");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_too_many_streams", valid_config(81, 1));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "eighty-one-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to 80 streams",
                                  "too-many-stream error mentions eighty-stream limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_invalid_worker_count(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_worker_count", valid_config(4, 2));

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "non-shared worker count is rejected") &&
                  expect_contains(result.stderr_text, "set inference.workers to 1",
                                  "invalid-worker error explains the shared-worker requirement");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_checks_fan_in_policy(const std::string& binary) {
  const fs::path every_frame_path =
      write_config("test_validate_config_only_accepts_every_frame_fan_in",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "inference:\n"
                       "  workers: 1\n"
                       "  fan_in_policy: every_frame\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");
  const auto every_frame = spawn_and_wait(
      binary, {"--config", every_frame_path.string(), "--validate-config-only"}, 20000);
  bool ok = expect_true(every_frame.exit_code == 0, "every-frame fan-in policy validates") &&
            expect_contains(every_frame.stdout_text, "fan_in_policy=every_frame",
                            "validate output reports every-frame fan-in");
  remove_dir(every_frame_path.parent_path().string());

  const fs::path invalid_path = write_config("test_validate_config_only_rejects_invalid_fan_in",
                                             model_header() + "streams:\n" + stream_entries(4) +
                                                 "inference:\n"
                                                 "  workers: 1\n"
                                                 "  fan_in_policy: lossy_magic\n"
                                                 "output:\n"
                                                 "  insight:\n"
                                                 "    host: 127.0.0.1\n");
  const auto invalid =
      spawn_and_wait(binary, {"--config", invalid_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(invalid.exit_code == 1, "invalid fan-in policy is rejected") &&
        expect_contains(invalid.stderr_text, "fan_in_policy must be one of: latest, every_frame",
                        "invalid fan-in error lists the supported portable policies");
  remove_dir(invalid_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_checks_internal_queue_depth(const std::string& binary) {
  const fs::path disabled_path =
      write_config("test_validate_config_only_accepts_disabled_internal_queue",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "input:\n"
                       "  tcp: true\n"
                       "inference:\n"
                       "  workers: 1\n"
                       "  internal_queue_depth: 0\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");
  const auto disabled =
      spawn_and_wait(binary, {"--config", disabled_path.string(), "--validate-config-only"}, 20000);
  bool ok = expect_true(disabled.exit_code == 0, "internal stage queues can be disabled") &&
            expect_contains(disabled.stdout_text, "internal_queue_depth=0",
                            "validate output reports disabled internal queues");
  remove_dir(disabled_path.parent_path().string());

  const fs::path oversized_path =
      write_config("test_validate_config_only_rejects_oversized_internal_queue",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "input:\n"
                       "  tcp: true\n"
                       "inference:\n"
                       "  workers: 1\n"
                       "  internal_queue_depth: 33\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");
  const auto oversized = spawn_and_wait(
      binary, {"--config", oversized_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(oversized.exit_code == 1, "oversized internal stage queue is rejected") &&
        expect_contains(oversized.stderr_text, "internal_queue_depth must be <= 32",
                        "internal queue validation identifies the bad setting");
  remove_dir(oversized_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_checks_max_inflight_per_stream(const std::string& binary) {
  const fs::path tuned_path = write_config("test_validate_config_only_accepts_tuned_max_inflight",
                                           model_header() + "streams:\n" + stream_entries(4) +
                                               "inference:\n"
                                               "  workers: 1\n"
                                               "  max_inflight_per_stream: 4\n"
                                               "output:\n"
                                               "  insight:\n"
                                               "    host: 127.0.0.1\n");
  const auto tuned =
      spawn_and_wait(binary, {"--config", tuned_path.string(), "--validate-config-only"}, 20000);
  bool ok = expect_true(tuned.exit_code == 0, "public per-stream credit can be tuned") &&
            expect_contains(tuned.stdout_text, "max_inflight_per_stream=4",
                            "validate output reports tuned per-stream credit");
  remove_dir(tuned_path.parent_path().string());

  const fs::path invalid_path =
      write_config("test_validate_config_only_rejects_invalid_max_inflight",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "inference:\n"
                       "  workers: 1\n"
                       "  max_inflight_per_stream: 0\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");
  const auto invalid =
      spawn_and_wait(binary, {"--config", invalid_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(invalid.exit_code == 1, "zero per-stream credit is rejected") &&
        expect_contains(invalid.stderr_text, "max_inflight_per_stream must be > 0",
                        "per-stream credit validation identifies the bad setting");
  remove_dir(invalid_path.parent_path().string());
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
                                                "  workers: 1\n"
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

bool test_validate_config_only_rejects_invalid_decoder_tuning(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_invalid_decoder_tuning",
                   model_header() + "streams:\n" + stream_entries(2) +
                       "input:\n"
                       "  tcp: true\n"
                       "  decoder_tuning: magic\n"
                       "inference:\n"
                       "  workers: 1\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "invalid decoder tuning config is rejected") &&
                  expect_contains(result.stderr_text, "input.decoder_tuning",
                                  "invalid decoder tuning error mentions setting");
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
  ok &= test_validate_config_only_accepts_twenty_four_streams(binary);
  ok &= test_validate_config_only_accepts_named_profiles(binary);
  ok &= test_validate_config_only_accepts_insight_visible_limit(binary);
  ok &= test_validate_config_only_rejects_overlapping_insight_ports(binary);
  ok &= test_validate_config_only_accepts_forty_streams(binary);
  ok &= test_validate_config_only_rejects_removed_output_paths(binary);
  ok &= test_validate_config_only_accepts_yolov8_decode_type(binary);
  ok &= test_validate_config_only_accepts_input_caps(binary);
  ok &= test_validate_config_only_accepts_decoder_tuning(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_insight_visible_limit_above_stream_count(binary);
  ok &= test_validate_config_only_rejects_invalid_worker_count(binary);
  ok &= test_validate_config_only_checks_fan_in_policy(binary);
  ok &= test_validate_config_only_checks_internal_queue_depth(binary);
  ok &= test_validate_config_only_checks_max_inflight_per_stream(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_validate_config_only_rejects_fps_scheduler_knob(binary);
  ok &= test_validate_config_only_rejects_invalid_decoder_tuning(binary);
  return ok ? 0 : 1;
}
