#include "support/object_detection/detection_egress.h"
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"
#include "received_metadata_tracker.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <utility>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace fs = std::filesystem;

using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_scratch_config;

namespace {

constexpr const char* kModelPath = "models/yolo26n/w8a8/yolo26n_w8a8_mpk.tar.gz";

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
                                 const sima_examples::detection_egress::FrameMetadata& frame) {
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
  sima_examples::detection_egress::FrameMetadata frame;
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
      sima_examples::detection_egress::serialize(boxes, labels, 100, 100, frame);
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
      sima_examples::detection_egress::serialize(boxes, labels, 100, 100, frame);
  ok &= expect_true(no_pts == legacy_metadata_json(boxes, labels, 100, 100, frame),
                    "single-pass metadata preserves the no-PTS envelope");
  ok &= expect_true(!nlohmann::json::parse(no_pts).contains("rtp_timestamp"),
                    "metadata omits RTP timestamp when source PTS is absent");
  return ok;
}

// Every key the application reads, with test values, followed by `extra`.
std::string config_body(const std::string& extra = "", const std::string& skip_key = "") {
  const std::vector<std::pair<std::string, std::string>> lines{
      {"", "model:"},
      {"model.path", "  path: models/yolo26n_w8a8_mpk.tar.gz"},
      {"model.labels", "  labels: coco_label.txt"},
      {"", "input:"},
      {"input.source",
       "  source:\n    - rtsp://127.0.0.1:8554/src1\n    - rtsp://127.0.0.1:8554/src2"},
      {"input.codec", "  codec: h264"},
      {"input.width", "  width: 1280"},
      {"input.height", "  height: 720"},
      {"input.fps", "  fps: 30"},
      {"", "  rtsp:"},
      {"input.rtsp.tcp", "    tcp: true"},
      {"input.rtsp.latency_ms", "    latency_ms: 100"},
      {"", "inference:"},
      {"inference.min_score", "  min_score: 0.3"},
      {"inference.nms_iou", "  nms_iou: 0.6"},
      {"inference.max_detections", "  max_detections: 50"},
      {"inference.max_inflight_per_stream", "  max_inflight_per_stream: 1"},
      {"inference.max_inflight_total", "  max_inflight_total: 8"},
      {"", "output:"},
      {"output.video_enabled", "  video_enabled: true"},
      {"", "  insight:"},
      {"output.insight.host", "    host: 127.0.0.1"},
      {"output.insight.video_port_base", "    video_port_base: 9000"},
      {"output.insight.metadata_port_base", "    metadata_port_base: 9100"},
      {"", "app:"},
      {"app.decoder_buffers", "  decoder_buffers: 8"},
      {"app.queue_depth", "  queue_depth: 16"},
      {"app.internal_queue_depth", "  internal_queue_depth: 1"},
  };
  std::string body;
  for (const auto& [key, line] : lines) {
    if (key.empty() || key != skip_key) {
      body += line + "\n";
    }
  }
  return body + extra;
}

sima_examples::testing::ProcessResult validate(const std::string& binary, const std::string& name,
                                               const std::string& body) {
  const fs::path config_path =
      write_scratch_config("high-density-multi-stream-object-detector", name, body);
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  return result;
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_unknown_argument_fails(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--frames", "10"}, 20000);
  return expect_true(result.exit_code == 1, "an unknown argument exits with code 1") &&
         expect_contains(result.stderr_text, "unknown argument: --frames",
                         "the error names the unknown argument");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_shipped_profiles_validate(const std::string& binary) {
  const fs::path common_dir =
      "examples/object-detection/high-density-multi-stream-object-detector/src/common";
  const std::vector<std::pair<std::string, int>> profiles{
      {"config.yaml", 20}, {"config-30x720p20fps.yaml", 30}, {"config-60x720p10fps.yaml", 60}};
  bool ok = true;
  for (const auto& [filename, streams] : profiles) {
    const auto result = spawn_and_wait(
        binary, {"--config", (common_dir / filename).string(), "--validate-config-only"}, 20000);
    ok &= expect_true(result.exit_code == 0, filename + " validates") &&
          expect_contains(result.stdout_text, "streams=" + std::to_string(streams),
                          filename + " reads every stream");
  }
  return ok;
}

bool test_config_reads_every_key(const std::string& binary) {
  const auto result = validate(binary, "config_reads_every_key", config_body());
  return expect_true(result.exit_code == 0, "a config with every key validates") &&
         expect_contains(result.stdout_text, "streams=2", "both sources are read");
}

bool test_unknown_key_is_rejected(const std::string& binary) {
  const auto result =
      validate(binary, "unknown_key_is_rejected", config_body("streams:\n  - rtsp://old\n"));
  return expect_true(result.exit_code == 1, "a config with an unknown key is rejected") &&
         expect_contains(result.stderr_text, "unknown keys: streams",
                         "the error names the unknown key");
}

bool test_missing_key_is_rejected(const std::string& binary) {
  const auto result =
      validate(binary, "missing_key_is_rejected", config_body("", "app.queue_depth"));
  return expect_true(result.exit_code == 1, "a config without app.queue_depth is rejected") &&
         expect_contains(result.stderr_text, "is missing app.queue_depth",
                         "the error names the missing key");
}

bool test_codec_must_be_h264_or_h265(const std::string& binary) {
  std::string body = config_body();
  body.replace(body.find("codec: h264"), std::string("codec: h264").size(), "codec: mjpeg");
  const auto result = validate(binary, "codec_must_be_h264_or_h265", body);
  return expect_true(result.exit_code == 1, "an unsupported codec is rejected") &&
         expect_contains(result.stderr_text, "input.codec must be h264 or h265",
                         "the error names the supported codecs");
}

bool test_received_metadata_tracker() {
  using Tracker = high_density::testing::ReceivedMetadataTracker;
  const auto start = Tracker::TimePoint{};
  Tracker tracker(2, 2, 3, std::chrono::milliseconds(100), std::chrono::milliseconds(50), start);
  bool ok = true;
  ok &= expect_true(!tracker.observe(0, "0", true, start + std::chrono::milliseconds(5)) &&
                        !tracker.observe(0, "0", true, start + std::chrono::milliseconds(6)),
                    "receiver tracker ignores duplicate frame identities");
  tracker.observe(0, "1", false, start + std::chrono::milliseconds(10));
  tracker.observe(1, "0", true, start + std::chrono::milliseconds(15));
  tracker.observe(1, "1", false, start + std::chrono::milliseconds(20));
  ok &= expect_true(tracker.warmup_complete() && !tracker.measurement_started() &&
                        tracker.total_measured() == 0,
                    "measurement waits for an explicit post-warmup boundary");
  tracker.start_measurement(start + std::chrono::milliseconds(20));
  tracker.observe(0, "2", false, start + std::chrono::milliseconds(30));
  tracker.observe(1, "2", false, start + std::chrono::milliseconds(40));
  ok &= expect_true(tracker.observe(0, "3", false, start + std::chrono::milliseconds(50)) &&
                        tracker.total_measured() == 3 &&
                        tracker.measured_frames() == std::vector<std::uint64_t>({2, 1}) &&
                        std::abs(tracker.elapsed_seconds() - 0.03) < 1e-9,
                    "receiver tracker measures exactly the common post-warmup interval");
  ok &= expect_true(std::all_of(tracker.useful_detection().begin(),
                                tracker.useful_detection().end(), [](bool value) { return value; }),
                    "receiver tracker records useful detections for every stream");

  Tracker deadlines(2, 1, 10, std::chrono::milliseconds(100), std::chrono::milliseconds(50), start);
  deadlines.observe(0, "0", false, start + std::chrono::milliseconds(25));
  ok &= expect_true(deadlines.stalled_streams(start + std::chrono::milliseconds(100)) ==
                        std::vector<std::size_t>{1},
                    "initial deadline identifies the stream with no receiver progress");
  deadlines.observe(1, "0", false, start + std::chrono::milliseconds(100));
  deadlines.start_measurement(start + std::chrono::milliseconds(100));
  deadlines.observe(0, "1", false, start + std::chrono::milliseconds(125));
  ok &= expect_true(deadlines.stalled_streams(start + std::chrono::milliseconds(150)) ==
                        std::vector<std::size_t>{1},
                    "ongoing deadline identifies a stalled measured stream");
  return ok;
}

bool test_metadata_listener_validates_queued_messages() {
  const int probe = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (probe < 0)
    return expect_true(false, "metadata listener test creates a port probe");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(probe, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) {
    ::close(probe);
    return expect_true(false, "metadata listener test binds a port probe");
  }
  socklen_t address_size = sizeof(address);
  if (::getsockname(probe, reinterpret_cast<sockaddr*>(&address), &address_size) < 0) {
    ::close(probe);
    return expect_true(false, "metadata listener test reads the selected port");
  }
  const int port = ntohs(address.sin_port);
  ::close(probe);

  sima_examples::testing::MetadataJsonListenerOptions options;
  options.base_port = port;
  options.timeout_ms = 20;
  sima_examples::testing::MetadataJsonListener listener(options);
  if (!listener.ok())
    return expect_true(false, listener.error());

  const int sender = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (sender < 0)
    return expect_true(false, "metadata listener test creates a sender");
  const std::string first =
      R"({"type":"object-detection","timestamp":1,"frame_id":"warmup-1","data":{"objects":[]}})";
  const std::string second =
      R"({"type":"object-detection","timestamp":2,"frame_id":"warmup-2","data":{"objects":[]}})";
  const std::string malformed = "{not json";
  const auto send = [&](const std::string& payload) {
    return ::sendto(sender, payload.data(), payload.size(), 0,
                    reinterpret_cast<const sockaddr*>(&address), sizeof(address));
  };
  bool ok = expect_true(send(first) == static_cast<ssize_t>(first.size()) &&
                            send(second) == static_cast<ssize_t>(second.size()) &&
                            send(malformed) == static_cast<ssize_t>(malformed.size()),
                        "metadata listener test queues warmup messages");
  ::close(sender);
  ok &= expect_true(listener.wait_for_messages().success,
                    "metadata listener consumes the first warmup message");
  const auto drained = listener.drain_pending();
  ok &= expect_true(drained.messages.size() == 1 &&
                        drained.error.find("json parse failed") != std::string::npos,
                    "metadata listener validates queued warmup messages");
  const auto after_discard = listener.wait_for_messages();
  ok &= expect_true(!after_discard.success && after_discard.timed_out,
                    "drained warmup metadata is not observed later");
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  if (!test_received_metadata_tracker())
    return 1;
  if (!test_metadata_listener_validates_queued_messages())
    return 1;

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
  ok &= test_unknown_argument_fails(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_shipped_profiles_validate(binary);
  ok &= test_config_reads_every_key(binary);
  ok &= test_unknown_key_is_rejected(binary);
  ok &= test_missing_key_is_rejected(binary);
  ok &= test_codec_must_be_h264_or_h265(binary);
  return ok ? 0 : 1;
}
