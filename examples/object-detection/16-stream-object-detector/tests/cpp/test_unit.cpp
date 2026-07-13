#include "support/testing/test_process.h"
#include "examples/object-detection/16-stream-object-detector/src/cpp/encoded_delivery_queue.h"

#include <array>
#include <chrono>
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
      {"config-16x720p25.yaml", 16, 25, 16, 1, 1, "latest"},
      {"config-24x720p20.yaml", 24, 20, 4, 1, 4, "every_frame"},
      {"config-48x720p10.yaml", 48, 10, 1, 2, 1, "latest"},
  }};
  const fs::path common_dir = "examples/object-detection/16-stream-object-detector/src/common";

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

struct TestEncodedFrame {
  int id = 0;
};

bool test_delayed_video_metadata_matching() {
  using Queue = app16::EncodedDeliveryQueue<TestEncodedFrame>;
  Queue queue(/*max_frames=*/4, /*max_metadata=*/4);
  const auto now = Queue::Clock::now();

  bool ok = true;
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{7}, now + std::chrono::milliseconds(10)),
                    "encoded AU is accepted below the queue limit");
  TestEncodedFrame frame;
  ok &= expect_true(!queue.pop_due_frame(now, frame), "encoded AU stays queued until its deadline");
  ok &=
      expect_true(queue.pop_due_frame(now + std::chrono::milliseconds(10), frame) && frame.id == 7,
                  "encoded AU is released at its synchronization deadline");

  ok &= expect_true(!queue.enqueue_metadata(100, "metadata-100"),
                    "first metadata object is queued without overflow");
  ok &= expect_true(!queue.enqueue_metadata(200, "metadata-200"),
                    "second metadata object is queued without overflow");
  ok &= expect_true(queue.take_metadata_through(99).empty(),
                    "video never receives metadata from a future PTS");
  ok &= expect_true(queue.take_metadata_through(150) == "metadata-100",
                    "video receives the newest metadata not newer than its PTS");

  ok &= expect_true(!queue.enqueue_metadata(250, "metadata-250"), "later metadata remains ordered");
  ok &= expect_true(queue.take_metadata_through(300) == "metadata-250",
                    "eligible stale metadata is coalesced to the newest result");

  ok &= expect_true(!queue.enqueue_metadata(400, "metadata-400-first") &&
                        !queue.enqueue_metadata(350, "metadata-350") &&
                        !queue.enqueue_metadata(400, "metadata-400-second"),
                    "out-of-order and equal-PTS metadata is accepted");
  ok &= expect_true(queue.take_metadata_through(399) == "metadata-350",
                    "out-of-order insertion remains sorted by PTS");
  ok &= expect_true(queue.take_metadata_through(400) == "metadata-400-second",
                    "equal-PTS coalescing deterministically keeps the newest arrival");

  ok &= expect_true(queue.pending_count() == 0, "matched video and metadata leave no backlog");
  return ok;
}

bool test_pts_epoch_matching_with_interleaved_reconnect() {
  using Queue = app16::EncodedDeliveryQueue<TestEncodedFrame>;
  constexpr std::int64_t kSecond = 1000000000LL;
  Queue queue(/*max_frames=*/16, /*max_metadata=*/16,
              /*max_frame_bytes=*/1024, /*pts_reset_threshold_ns=*/kSecond);
  const auto due = Queue::Clock::now();
  bool ok = true;

  // The encoded branch observes a reconnect first while old-epoch AUs are
  // still held by the synchronization delay. Inference then finishes one old
  // result before it observes the new epoch. This is the real asynchronous
  // ordering that a raw numeric-PTS sort used to mix up.
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{1}, due, 1, 10 * kSecond),
                    "old epoch frame 1 is queued");
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{2}, due, 1, 11 * kSecond),
                    "old epoch frame 2 is queued");
  ok &=
      expect_true(!queue.enqueue_metadata(10 * kSecond, "old-10"), "old epoch metadata is queued");
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{3}, due, 1, 0),
                    "PTS reset begins a new video epoch");
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{4}, due, 1, kSecond / 10),
                    "new epoch video continues");
  ok &= expect_true(!queue.enqueue_metadata(11 * kSecond, "old-11"),
                    "delayed old inference remains in the old epoch");
  ok &= expect_true(!queue.enqueue_metadata(0, "new-0"),
                    "metadata PTS reset begins the matching new epoch");

  TestEncodedFrame frame;
  std::uint64_t epoch = 99;
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 1 && epoch == 0,
                    "first old video frame retains epoch zero");
  ok &= expect_true(queue.take_metadata_through(epoch, 10 * kSecond) == "old-10",
                    "old video cannot consume new-epoch metadata");
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 2 && epoch == 0,
                    "second old video frame retains epoch zero");
  ok &= expect_true(queue.take_metadata_through(epoch, 11 * kSecond) == "old-11",
                    "late old detection still matches its old video epoch");
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 3 && epoch == 1,
                    "reset video frame is tagged with epoch one");
  ok &= expect_true(queue.take_metadata_through(epoch, 0) == "new-0",
                    "reset video receives only reset-epoch metadata");
  ok &= expect_true(queue.frame_epoch_resets() == 1 && queue.metadata_epoch_resets() == 1,
                    "both PTS discontinuities are counted");

  // If inference starts only after the reconnect, the exact queued AU is the
  // authority that aligns its first result with the already-advanced epoch.
  Queue late_inference(/*max_frames=*/4, /*max_metadata=*/4,
                       /*max_frame_bytes=*/1024, /*pts_reset_threshold_ns=*/kSecond);
  ok &= expect_true(late_inference.enqueue_frame(TestEncodedFrame{5}, due, 1, 5 * kSecond),
                    "pre-reconnect frame is queued for late-inference fixture");
  ok &= expect_true(late_inference.pop_due_frame(due, frame, &epoch),
                    "pre-reconnect frame leaves before inference begins");
  ok &= expect_true(late_inference.enqueue_frame(TestEncodedFrame{6}, due, 1, 0),
                    "late-inference fixture observes video epoch reset");
  ok &= expect_true(!late_inference.enqueue_metadata(0, "first-new-result"),
                    "first inference result uses exact queued AU to align epoch");
  ok &= expect_true(late_inference.pop_due_frame(due, frame, &epoch) && epoch == 1,
                    "late-inference reset frame is epoch one");
  ok &= expect_true(late_inference.take_metadata_through(epoch, 0) == "first-new-result",
                    "late inference result is not stranded in epoch zero");

  Queue stale_old(/*max_frames=*/4, /*max_metadata=*/4,
                  /*max_frame_bytes=*/1024, /*pts_reset_threshold_ns=*/kSecond);
  ok &= expect_true(stale_old.enqueue_frame(TestEncodedFrame{7}, due, 1, 10 * kSecond),
                    "stale-old fixture queues its final old video frame");
  ok &= expect_true(!stale_old.enqueue_metadata(11 * kSecond, "unmatched-old-future"),
                    "old future metadata can remain unmatched at reconnect");
  ok &= expect_true(stale_old.enqueue_frame(TestEncodedFrame{8}, due, 1, 0),
                    "stale-old fixture starts the new epoch");
  ok &= expect_true(!stale_old.enqueue_metadata(0, "fresh-new"),
                    "stale-old fixture queues fresh metadata");
  ok &= expect_true(stale_old.pop_due_frame(due, frame, &epoch) && epoch == 0,
                    "final old frame leaves before reset frame");
  ok &= expect_true(stale_old.take_metadata_through(epoch, 10 * kSecond).empty(),
                    "future old metadata is not sent early");
  ok &= expect_true(stale_old.pop_due_frame(due, frame, &epoch) && epoch == 1,
                    "stale-old fixture reaches reset video");
  std::size_t stale_count = 0;
  ok &= expect_true(stale_old.take_metadata_through(epoch, 0, &stale_count) == "fresh-new" &&
                        stale_count == 1,
                    "new epoch drops unmatched old metadata and reports the miss");
  return ok;
}

bool test_pts_epoch_out_of_order_completion_does_not_poison_new_epoch() {
  using Queue = app16::EncodedDeliveryQueue<TestEncodedFrame>;
  using Status = Queue::MetadataEnqueueStatus;
  constexpr std::int64_t kSecond = 1000000000LL;
  Queue queue(/*max_frames=*/16, /*max_metadata=*/16,
              /*max_frame_bytes=*/1024, /*pts_reset_threshold_ns=*/kSecond);
  const auto due = Queue::Clock::now();
  bool ok = true;

  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{10}, due, 1, 10 * kSecond) &&
                        queue.enqueue_frame(TestEncodedFrame{11}, due, 1, 11 * kSecond) &&
                        queue.enqueue_frame(TestEncodedFrame{20}, due, 1, 0) &&
                        queue.enqueue_frame(TestEncodedFrame{21}, due, 1, kSecond / 10),
                    "poisoning fixture retains old and new epoch AU identities");

  // With multiple model credits, the first new-epoch result may complete
  // before an older in-flight result. Metadata completion order must never be
  // used to invent another epoch.
  ok &= expect_true(queue.enqueue_metadata_status(0, "new-0") == Status::Accepted,
                    "new epoch metadata can complete first");
  ok &= expect_true(queue.enqueue_metadata_status(11 * kSecond, "late-old-11") == Status::Accepted,
                    "late old metadata resolves from encoded AU history");
  ok &= expect_true(queue.metadata_epoch() == 1 && queue.metadata_epoch_resets() == 1,
                    "late old completion cannot advance or rewind metadata epoch high-water");

  TestEncodedFrame frame;
  std::uint64_t epoch = 99;
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 10 && epoch == 0,
                    "poisoning fixture sends first old frame");
  ok &= expect_true(queue.take_metadata_through(epoch, 10 * kSecond).empty(),
                    "future late-old result is not sent early");
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 11 && epoch == 0,
                    "poisoning fixture sends matching old frame");
  ok &= expect_true(queue.take_metadata_through(epoch, 11 * kSecond) == "late-old-11",
                    "late old result stays with old epoch");
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 20 && epoch == 1,
                    "poisoning fixture sends first new frame");
  ok &= expect_true(queue.take_metadata_through(epoch, 0) == "new-0",
                    "new result stays with new epoch");

  // The exact AU can leave the delay queue before its inference completes.
  // Retained CPU-only PTS history still resolves the result to epoch one.
  ok &= expect_true(queue.pop_due_frame(due, frame, &epoch) && frame.id == 21 && epoch == 1,
                    "matching new AU can be sent before metadata completes");
  ok &=
      expect_true(queue.enqueue_metadata_status(kSecond / 10, "new-100ms-late") == Status::Accepted,
                  "metadata arriving after exact AU pop resolves from retained history");
  ok &= expect_true(queue.metadata_epoch() == 1 && queue.metadata_epoch_unresolved() == 0,
                    "new-after-late-old completion does not create phantom epoch two");
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{22}, due, 1, 2 * kSecond / 10) &&
                        queue.pop_due_frame(due, frame, &epoch),
                    "next new-epoch video is available for delayed result");
  ok &= expect_true(queue.take_metadata_through(epoch, 2 * kSecond / 10) == "new-100ms-late",
                    "late result is released only on later video in the same epoch");

  Queue bounded_history(/*max_frames=*/4, /*max_metadata=*/4,
                        /*max_frame_bytes=*/1024, /*pts_reset_threshold_ns=*/kSecond,
                        /*max_frame_history=*/2);
  ok &= expect_true(bounded_history.enqueue_frame(TestEncodedFrame{30}, due, 1, 10 * kSecond) &&
                        bounded_history.enqueue_frame(TestEncodedFrame{31}, due, 1, 0) &&
                        bounded_history.enqueue_frame(TestEncodedFrame{32}, due, 1, kSecond / 10),
                    "bounded history fixture advances to epoch one");
  ok &= expect_true(bounded_history.enqueue_metadata_status(kSecond / 10, "new-current") ==
                        Status::Accepted,
                    "bounded history establishes epoch one from an exact AU");
  ok &= expect_true(bounded_history.enqueue_metadata_status(10 * kSecond, "too-late-old") ==
                        Status::UnresolvedEpoch,
                    "expired old AU identity is rejected instead of guessed");
  ok &= expect_true(bounded_history.metadata_epoch() == 1 &&
                        bounded_history.metadata_epoch_unresolved() == 1,
                    "unresolved old result cannot poison current metadata epoch");

  Queue duplicate_pts(/*max_frames=*/4, /*max_metadata=*/4,
                      /*max_frame_bytes=*/1024, /*pts_reset_threshold_ns=*/kSecond);
  ok &= expect_true(duplicate_pts.enqueue_frame(TestEncodedFrame{40}, due, 1, 0) &&
                        duplicate_pts.enqueue_frame(TestEncodedFrame{41}, due, 1, 10 * kSecond) &&
                        duplicate_pts.enqueue_frame(TestEncodedFrame{42}, due, 1, 0),
                    "duplicate PTS fixture retains the same PTS from two epochs");
  ok &= expect_true(duplicate_pts.enqueue_metadata_status(0, "ambiguous-zero") ==
                        Status::UnresolvedEpoch,
                    "same retained PTS in two epochs is rejected instead of guessed");
  ok &= expect_true(duplicate_pts.metadata_epoch() == 0 &&
                        duplicate_pts.metadata_epoch_unresolved() == 1,
                    "ambiguous duplicate PTS cannot establish or poison metadata epoch");
  return ok;
}

bool test_encoded_delivery_queue_overflow_contract() {
  using Queue = app16::EncodedDeliveryQueue<TestEncodedFrame>;
  Queue queue(/*max_frames=*/2, /*max_metadata=*/2);
  const auto due = Queue::Clock::now();

  bool ok = true;
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{1}, due),
                    "first encoded AU fits bounded queue");
  ok &= expect_true(queue.enqueue_frame(TestEncodedFrame{2}, due),
                    "second encoded AU fits bounded queue");
  ok &= expect_true(!queue.enqueue_frame(TestEncodedFrame{3}, due),
                    "full encoded queue rejects the new AU instead of dropping an old AU");
  ok &=
      expect_true(queue.frame_count() == 2, "encoded overflow preserves the existing AU sequence");
  ok &= expect_true(queue.frame_count_high_water() == 2,
                    "encoded queue reports its admitted frame high-water mark");

  TestEncodedFrame frame;
  ok &= expect_true(queue.peek_due_frame(due, frame) && frame.id == 1,
                    "nonblocking sender can inspect the first due AU");
  ok &= expect_true(queue.frame_count() == 2,
                    "a backpressured sender leaves the inspected AU queued");
  ok &= expect_true(queue.peek_due_frame(due, frame) && frame.id == 1,
                    "retries preserve the same per-channel AU order");
  ok &= expect_true(queue.pop_due_frame(due, frame) && frame.id == 1,
                    "encoded overflow preserves the first queued AU");
  ok &= expect_true(queue.pop_due_frame(due, frame) && frame.id == 2,
                    "encoded overflow preserves the second queued AU");

  ok &= expect_true(!queue.enqueue_metadata(10, "metadata-10"),
                    "first metadata object fits bounded queue");
  ok &= expect_true(!queue.enqueue_metadata(20, "metadata-20"),
                    "second metadata object fits bounded queue");
  ok &= expect_true(queue.enqueue_metadata(30, "metadata-30"),
                    "metadata overflow explicitly reports dropping the oldest object");
  ok &= expect_true(queue.take_metadata_through(100) == "metadata-30",
                    "metadata overflow retains the newest synchronized result");
  return ok;
}

bool test_encoded_delivery_queue_live_capacity() {
  using Queue = app16::EncodedDeliveryQueue<TestEncodedFrame>;
  bool ok = true;
  ok &= expect_true(app16::encoded_delivery_frame_capacity(10, 400) == 68,
                    "10 fps queue includes four delayed and 64 catch-up AUs");
  ok &= expect_true(app16::encoded_delivery_frame_capacity(20, 400) == 72,
                    "20 fps queue includes eight delayed and 64 catch-up AUs");
  ok &= expect_true(app16::encoded_delivery_frame_capacity(60, 400) == 144,
                    "high-rate queue reserves two source-seconds beyond the delay");
  ok &= expect_true(app16::encoded_delivery_frame_capacity(0, 400) == 0,
                    "invalid source rate cannot create an encoded queue");

  Queue catch_up_queue(app16::encoded_delivery_frame_capacity(10, 400),
                       /*max_metadata=*/8, /*max_frame_bytes=*/1024);
  const auto now = Queue::Clock::now();
  bool accepted_catch_up_burst = true;
  for (int id = 0; id < 15; ++id) {
    accepted_catch_up_burst &=
        catch_up_queue.enqueue_frame(TestEncodedFrame{id}, now + std::chrono::milliseconds(50), 1);
  }
  ok &= expect_true(accepted_catch_up_burst,
                    "observed 15-AU RTSP catch-up burst fits the live queue reserve");
  ok &= expect_true(catch_up_queue.frame_count() == 15,
                    "catch-up burst retains every H.264 AU in order");
  ok &= expect_true(catch_up_queue.oldest_frame_due_in_ms(now) == 50,
                    "catch-up reserve admits frames even while none is due yet");

  Queue byte_queue(/*max_frames=*/8, /*max_metadata=*/2, /*max_frame_bytes=*/5);
  ok &= expect_true(byte_queue.enqueue_frame(TestEncodedFrame{1}, now, 3),
                    "encoded byte-limited queue accepts a fitting AU");
  ok &= expect_true(!byte_queue.enqueue_frame(TestEncodedFrame{2}, now, 3),
                    "encoded byte limit rejects an AU without discarding prior data");
  ok &= expect_true(byte_queue.frame_bytes() == 3 && byte_queue.frame_bytes_high_water() == 3,
                    "byte-limit rejection preserves byte accounting and high-water state");
  TestEncodedFrame reclaimed;
  ok &= expect_true(byte_queue.pop_due_frame(now, reclaimed) && reclaimed.id == 1,
                    "popping an AU reclaims its encoded byte budget");
  ok &= expect_true(byte_queue.enqueue_frame(TestEncodedFrame{2}, now, 3),
                    "reclaimed encoded byte budget is reusable");
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
  ok &= test_delayed_video_metadata_matching();
  ok &= test_pts_epoch_matching_with_interleaved_reconnect();
  ok &= test_pts_epoch_out_of_order_completion_does_not_poison_new_epoch();
  ok &= test_encoded_delivery_queue_overflow_contract();
  ok &= test_encoded_delivery_queue_live_capacity();
  return ok ? 0 : 1;
}
