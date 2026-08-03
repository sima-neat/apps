#include "../../src/cpp/detection_egress.h"
#include "../../src/cpp/detection_watchdog.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr const char* kModelPath = "models/yolo26n-det-int8-b1.tar.gz";

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

struct TestBox {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

std::string legacy_metadata_json(const std::vector<TestBox>& boxes,
                                 const std::vector<std::string>& labels, int frame_w, int frame_h,
                                 const high_density::detection_egress::FrameMetadata& frame) {
  nlohmann::json data;
  data["objects"] = nlohmann::json::array();
  int object_index = 1;
  for (const auto& box : boxes) {
    const int x1 = std::max(0, static_cast<int>(box.x1));
    const int y1 = std::max(0, static_cast<int>(box.y1));
    int width = std::max(0, static_cast<int>(box.x2 - box.x1));
    int height = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x1 + width > frame_w)
      width = frame_w - x1;
    if (y1 + height > frame_h)
      height = frame_h - y1;
    const std::string label = box.class_id >= 0 && box.class_id < static_cast<int>(labels.size())
                                  ? labels[static_cast<std::size_t>(box.class_id)]
                                  : "unknown";
    data["objects"].push_back({
        {"id", "obj_" + std::to_string(object_index++)},
        {"label", label},
        {"confidence", box.score},
        {"bbox",
         {static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(std::max(0, width)),
          static_cast<float>(std::max(0, height))}},
    });
  }

  nlohmann::json payload;
  payload["type"] = "object-detection";
  payload["data"] = nlohmann::json::parse(data.dump());
  payload["timestamp"] = frame.pts_ns >= 0 ? frame.pts_ns / 1'000'000 : -1;
  payload["frame_id"] = frame.frame_id >= 0 ? std::to_string(frame.frame_id) : "";
  payload["stream_id"] = std::string(frame.stream_id);
  payload["stream_index"] = frame.stream_index;
  payload["pts_ns"] = frame.pts_ns;
  payload["dts_ns"] = frame.dts_ns;
  payload["duration_ns"] = frame.duration_ns;
  payload["input_seq"] = frame.input_seq;
  payload["orig_input_seq"] = frame.orig_input_seq;
  if (frame.rtp_timestamp.has_value())
    payload["rtp_timestamp"] = *frame.rtp_timestamp;
  return payload.dump();
}

bool test_metadata_fast_path_preserves_insight_payload() {
  const std::vector<TestBox> boxes{
      {10.9f, -2.0f, 35.9f, 20.8f, 0.75f, 0},
      {90.0f, 80.0f, 120.0f, 130.0f, 0.5f, 1},
      {105.0f, 10.0f, 115.0f, 18.0f, 0.25f, 99},
  };
  const std::vector<std::string> labels{"person", "bicycle\n\"quoted\""};
  high_density::detection_egress::FrameMetadata frame;
  frame.stream_index = 7;
  frame.stream_id = "stream7";
  frame.frame_id = 42;
  frame.pts_ns = 1'234'567'890;
  frame.dts_ns = 1'200'000'000;
  frame.duration_ns = 50'000'000;
  frame.input_seq = 88;
  frame.orig_input_seq = 77;
  frame.rtp_timestamp = 111'111;

  const std::string expected = legacy_metadata_json(boxes, labels, 100, 100, frame);
  const std::string actual =
      high_density::detection_egress::serialize(boxes, labels, 100, 100, frame);
  bool ok = expect_true(actual == expected,
                        "single-pass metadata matches the legacy Insight JSON byte-for-byte");

  const auto parsed = nlohmann::json::parse(actual);
  ok &= expect_true(parsed.at("data").at("objects").size() == boxes.size(),
                    "metadata keeps every decoded box");
  ok &= expect_true(parsed.at("data").at("objects").at(0).at("id") == "obj_1" &&
                        parsed.at("data").at("objects").at(1).at("label") == labels.at(1),
                    "metadata preserves object IDs and escaped labels");
  ok &= expect_true(parsed.at("data").at("objects").at(1).at("bbox") ==
                        nlohmann::json({90.0f, 80.0f, 10.0f, 20.0f}),
                    "metadata preserves legacy frame-edge clamping");
  ok &= expect_true(parsed.at("data").at("objects").at(2).at("label") == "unknown",
                    "metadata preserves unknown-class labels");
  ok &= expect_true(parsed.at("rtp_timestamp") == *frame.rtp_timestamp,
                    "metadata preserves the source-aligned RTP timestamp");

  frame.frame_id = -1;
  frame.pts_ns = -1;
  frame.rtp_timestamp.reset();
  const std::string no_pts =
      high_density::detection_egress::serialize(boxes, labels, 100, 100, frame);
  ok &= expect_true(no_pts == legacy_metadata_json(boxes, labels, 100, 100, frame),
                    "single-pass metadata preserves the no-PTS envelope");
  ok &= expect_true(!nlohmann::json::parse(no_pts).contains("rtp_timestamp"),
                    "metadata omits RTP timestamp when source PTS is absent");
  return ok;
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
      expect_contains(result.stdout_text, "max_inflight_total=8",
                      "validate output reports the proven total credit default") &&
      expect_contains(result.stdout_text, "max_missed_detection_rounds=2",
                      "validate output reports the work-based starvation budget") &&
      expect_contains(result.stdout_text, "detection_progress_budget=56",
                      "validate output reports the computed shape-dependent budget") &&
      expect_contains(result.stdout_text, "no_detection_timeout_ms=30000",
                      "validate output reports the global no-progress timeout") &&
      expect_contains(result.stdout_text,
                      (config_path.parent_path() / kModelPath).lexically_normal().string(),
                      "relative model path resolves from the config directory");
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
    int max_inflight_total;
    int detection_progress_budget;
  };

  constexpr std::array<ProfileExpectation, 3> profiles{{
      {"config.yaml", 16, 25, 16, 1, 1, 8, 40},
      {"config-24x720p20fps.yaml", 24, 20, 4, 2, 4, 24, 72},
      {"config-48x720p10fps.yaml", 48, 10, 1, 2, 1, 8, 104},
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
    ok &= expect_contains(result.stdout_text,
                          "max_inflight_total=" + std::to_string(profile.max_inflight_total),
                          label + " reports its total credit");
    ok &= expect_contains(result.stdout_text,
                          "detection_progress_budget=" +
                              std::to_string(profile.detection_progress_budget),
                          label + " reports its computed detection progress budget");
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
                                                "  codec: hevc\n"
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

bool test_validate_config_only_checks_max_inflight_limits(const std::string& binary) {
  const fs::path tuned_path = write_config("test_validate_config_only_accepts_tuned_max_inflight",
                                           model_header() + "streams:\n" + stream_entries(4) +
                                               "inference:\n"
                                               "  workers: 1\n"
                                               "  max_inflight_per_stream: 4\n"
                                               "  max_inflight_total: 12\n"
                                               "output:\n"
                                               "  insight:\n"
                                               "    host: 127.0.0.1\n");
  const auto tuned =
      spawn_and_wait(binary, {"--config", tuned_path.string(), "--validate-config-only"}, 20000);
  bool ok = expect_true(tuned.exit_code == 0, "public per-stream credit can be tuned") &&
            expect_contains(tuned.stdout_text, "max_inflight_per_stream=4",
                            "validate output reports tuned per-stream credit") &&
            expect_contains(tuned.stdout_text, "max_inflight_total=12",
                            "validate output reports tuned total credit");
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

  const fs::path invalid_total_path =
      write_config("test_validate_config_only_rejects_invalid_max_inflight_total",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "inference:\n"
                       "  workers: 1\n"
                       "  max_inflight_total: 0\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");
  const auto invalid_total = spawn_and_wait(
      binary, {"--config", invalid_total_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(invalid_total.exit_code == 1, "zero total credit is rejected") &&
        expect_contains(invalid_total.stderr_text, "max_inflight_total must be > 0",
                        "total credit validation identifies the bad setting");
  remove_dir(invalid_total_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_checks_liveness_limits(const std::string& binary) {
  const std::string base = model_header() + "streams:\n" + stream_entries(4) +
                           "inference:\n"
                           "  workers: 1\n"
                           "  max_inflight_total: 12\n";
  const fs::path tuned_path = write_config("test_validate_config_only_accepts_tuned_liveness",
                                           base + "runtime:\n"
                                                  "  no_detection_timeout_ms: 45000\n"
                                                  "  max_missed_detection_rounds: 3\n"
                                                  "output:\n"
                                                  "  insight:\n"
                                                  "    host: 127.0.0.1\n");
  const auto tuned =
      spawn_and_wait(binary, {"--config", tuned_path.string(), "--validate-config-only"}, 20000);
  bool ok = expect_true(tuned.exit_code == 0, "detector liveness limits can be tuned") &&
            expect_contains(tuned.stdout_text, "no_detection_timeout_ms=45000",
                            "validate output reports the tuned global timeout") &&
            expect_contains(tuned.stdout_text, "max_missed_detection_rounds=3",
                            "validate output reports the tuned missed-round count") &&
            expect_contains(tuned.stdout_text, "detection_progress_budget=24",
                            "validate output reports rounds times streams plus in-flight work");
  remove_dir(tuned_path.parent_path().string());

  const fs::path invalid_timeout_path =
      write_config("test_validate_config_only_rejects_invalid_no_detection_timeout",
                   base + "runtime:\n"
                          "  no_detection_timeout_ms: 0\n"
                          "output:\n"
                          "  insight:\n"
                          "    host: 127.0.0.1\n");
  const auto invalid_timeout = spawn_and_wait(
      binary, {"--config", invalid_timeout_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(invalid_timeout.exit_code == 1, "zero global timeout is rejected") &&
        expect_contains(invalid_timeout.stderr_text, "no_detection_timeout_ms must be > 0",
                        "global timeout validation identifies the bad setting");
  remove_dir(invalid_timeout_path.parent_path().string());

  const fs::path invalid_rounds_path =
      write_config("test_validate_config_only_rejects_invalid_missed_detection_rounds",
                   base + "runtime:\n"
                          "  max_missed_detection_rounds: 0\n"
                          "output:\n"
                          "  insight:\n"
                          "    host: 127.0.0.1\n");
  const auto invalid_rounds = spawn_and_wait(
      binary, {"--config", invalid_rounds_path.string(), "--validate-config-only"}, 20000);
  ok &= expect_true(invalid_rounds.exit_code == 1, "zero missed-detection rounds is rejected") &&
        expect_contains(invalid_rounds.stderr_text, "max_missed_detection_rounds must be > 0",
                        "missed-round validation identifies the bad setting");
  remove_dir(invalid_rounds_path.parent_path().string());
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

bool test_validate_config_only_rejects_legacy_fan_in_policy(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_legacy_fan_in_policy",
                   model_header() + "streams:\n" + stream_entries(4) +
                       "inference:\n"
                       "  workers: 1\n"
                       "  fan_in_policy: realtime-latest-by-stream\n"
                       "output:\n"
                       "  insight:\n"
                       "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "legacy fan-in policy is rejected") &&
                  expect_contains(result.stderr_text, "inference.fan_in_policy was removed",
                                  "fan-in error identifies the removed setting") &&
                  expect_contains(result.stderr_text, "ordinary connect()/build()",
                                  "fan-in error explains automatic lowering");
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

bool test_detection_watchdog_tracks_completed_work() {
  using Watchdog = high_density::DetectionWatchdog;
  using FailureKind = high_density::DetectionFailureKind;
  const auto start = Watchdog::TimePoint{};
  Watchdog watchdog(/*stream_count=*/3, /*priming_observations=*/2,
                    std::chrono::milliseconds(100), /*no_progress_timeout=*/
                    std::chrono::milliseconds(500), /*max_missed_completions=*/4, start);

  bool ok = true;
  watchdog.observe(0, start + std::chrono::milliseconds(5));
  watchdog.observe(0, start + std::chrono::milliseconds(6));
  watchdog.observe(2, start + std::chrono::milliseconds(10));
  ok &= expect_true(!watchdog.check(start + std::chrono::milliseconds(99)),
                    "startup watchdog waits for the configured all-stream deadline");
  const auto startup_failure = watchdog.check(start + std::chrono::milliseconds(100));
  ok &= expect_true(startup_failure.kind == FailureKind::Startup &&
                        startup_failure.streams == std::vector<std::size_t>({1, 2}),
                    "startup watchdog reports streams that have not completed priming twice");

  watchdog.observe(1, start + std::chrono::milliseconds(101));
  watchdog.observe(1, start + std::chrono::milliseconds(102));
  watchdog.observe(0, start + std::chrono::milliseconds(105));
  watchdog.observe(2, start + std::chrono::milliseconds(105));
  ok &= expect_true(watchdog.startup_complete(),
                    "watchdog enters running mode only after every stream has two completions");
  ok &= expect_true(!watchdog.check(start + std::chrono::milliseconds(149)),
                    "wall-clock delay alone does not expire an individual stream");

  watchdog.observe(0, start + std::chrono::milliseconds(150));
  watchdog.observe(2, start + std::chrono::milliseconds(151));
  watchdog.observe(0, start + std::chrono::milliseconds(152));
  watchdog.observe(2, start + std::chrono::milliseconds(153));
  ok &= expect_true(!watchdog.check(start + std::chrono::milliseconds(153)),
                    "exactly the configured missed-completion budget is allowed");
  watchdog.observe(0, start + std::chrono::milliseconds(154));
  watchdog.observe(1, start + std::chrono::milliseconds(155));
  const auto starvation = watchdog.check(start + std::chrono::milliseconds(155));
  ok &= expect_true(starvation.kind == FailureKind::StreamStarvation &&
                        starvation.streams == std::vector<std::size_t>{1},
                    "a later completion cannot clear an already-exceeded progress budget");

  const auto global_stall = watchdog.check(start + std::chrono::milliseconds(655));
  ok &= expect_true(global_stall.kind == FailureKind::GlobalStall && global_stall.streams.empty(),
                    "a separate wall-clock guard detects total detector stagnation");

  Watchdog startup_stall_watchdog(
      /*stream_count=*/3, /*priming_observations=*/2, std::chrono::milliseconds(1000),
      /*no_progress_timeout=*/std::chrono::milliseconds(50),
      /*max_missed_completions=*/4, start);
  startup_stall_watchdog.observe(0, start + std::chrono::milliseconds(5));
  ok &= expect_true(!startup_stall_watchdog.check(start + std::chrono::milliseconds(54)),
                    "startup allows less than the global no-progress timeout");
  const auto startup_stall = startup_stall_watchdog.check(start + std::chrono::milliseconds(55));
  ok &= expect_true(startup_stall.kind == FailureKind::GlobalStall && startup_stall.streams.empty(),
                    "global detector stagnation is enforced while streams are still priming");

  Watchdog recovered_stall_watchdog(
      /*stream_count=*/1, /*priming_observations=*/1, std::chrono::milliseconds(1000),
      /*no_progress_timeout=*/std::chrono::milliseconds(50),
      /*max_missed_completions=*/4, start);
  recovered_stall_watchdog.observe(0, start + std::chrono::milliseconds(50));
  const auto recovered_stall =
      recovered_stall_watchdog.check(start + std::chrono::milliseconds(50));
  ok &= expect_true(recovered_stall.kind == FailureKind::GlobalStall &&
                        recovered_stall.streams.empty(),
                    "late detector progress cannot clear an already-expired global stall");

  constexpr std::size_t stream_count = 48;
  constexpr std::uint64_t progress_budget = 2 * stream_count + 8;
  Watchdog dense_watchdog(stream_count, /*priming_observations=*/2, std::chrono::seconds(60),
                          std::chrono::seconds(30), progress_budget, start);
  for (std::size_t index = 0; index < stream_count; ++index) {
    const auto offset = std::chrono::milliseconds(static_cast<int>(index * 10));
    dense_watchdog.observe(index, start + offset);
    dense_watchdog.observe(index, start + offset + std::chrono::milliseconds(1));
  }
  ok &= expect_true(dense_watchdog.startup_complete(),
                    "staggered 48-stream startup reaches a common progress baseline");
  for (std::uint64_t completion = 0; completion < 57; ++completion) {
    const std::size_t stream_index = 1 + static_cast<std::size_t>(completion % 47);
    dense_watchdog.observe(stream_index, start + std::chrono::seconds(9));
  }
  ok &= expect_true(!dense_watchdog.check(start + std::chrono::seconds(9)),
                    "the observed 48-stream completion gap does not false-timeout");
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--detection-egress-only") {
    return test_metadata_fast_path_preserves_insight_payload() ? 0 : 1;
  }
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_metadata_fast_path_preserves_insight_payload();
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
  ok &= test_validate_config_only_checks_internal_queue_depth(binary);
  ok &= test_validate_config_only_checks_max_inflight_limits(binary);
  ok &= test_validate_config_only_checks_liveness_limits(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_validate_config_only_rejects_fps_scheduler_knob(binary);
  ok &= test_validate_config_only_rejects_legacy_fan_in_policy(binary);
  ok &= test_validate_config_only_rejects_invalid_decoder_tuning(binary);
  ok &= test_detection_watchdog_tracks_completed_work();
  return ok ? 0 : 1;
}
