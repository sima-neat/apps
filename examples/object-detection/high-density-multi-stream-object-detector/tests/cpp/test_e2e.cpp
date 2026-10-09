#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"
#include "received_metadata_tracker.h"

#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace sima_examples::testing;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
constexpr const char* kExample = "high-density-multi-stream-object-detector";
// Clean exit after SIGINT must take no longer than this, as in the Python E2E.
constexpr long long kSigintExitLimitMs = 5000;

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::pair<int, json> validate_metadata_message(const MetadataJsonMessage& message, int base_port) {
  const auto payload = json::parse(message.payload);
  const int index = message.port - base_port;
  require(index >= 0 && index < 16, "metadata arrived on an unexpected port");
  require(payload.at("type") == "object-detection" && payload.at("data").at("objects").is_array(),
          "invalid object-detection metadata schema");
  require(payload.at("stream_index") == index &&
              payload.at("stream_id") == "stream" + std::to_string(index),
          "wrong metadata channel");
  require(!payload.at("frame_id").get<std::string>().empty() &&
              payload.at("pts_ns").get<int64_t>() >= 0 && payload.contains("rtp_timestamp"),
          "missing frame identity");
  return {index, payload};
}

std::string select_source(const std::vector<std::string>& urls, const std::string& codec) {
  std::set<std::string> seen;
  for (const auto& url : urls) {
    if (!seen.insert(url).second)
      continue;
    cv::VideoCapture cap(url, cv::CAP_ANY,
                         {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 20000,
                          cv::CAP_PROP_READ_TIMEOUT_MSEC, 20000});
    if (!cap.isOpened()) {
      std::cerr << "Cannot open source: " << url << "\n";
      continue;
    }
    const double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps = cap.get(cv::CAP_PROP_FPS);
    cap.release();
    if (width == 1280 && height == 720 &&
        (std::abs(fps - 30) < 0.001 || std::abs(fps - 30000.0 / 1001) < 0.001)) {
      std::cout << "source codec=" << codec << " width=1280 height=720 fps=" << fps
                << " unique_publishers=1\n";
      return url;
    }
    std::cerr << "Unsuitable source: width=" << width << " height=" << height
              << " fps=" << fps << "\n";
  }
  throw std::runtime_error("No 720p30 " + codec + " source");
}

void run_case(const std::string& binary, const std::string& model, const std::string& codec,
              const std::vector<std::string>& urls) {
  const auto url = select_source(urls, codec);
  const auto output = create_test_output_dir(kExample, "metadata_throughput_" + codec);
  require(!output.empty(), "cannot create output directory");
  const auto config = fs::path(output).parent_path() / "config.yaml";
  const int port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int video = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  write_e2e_config(
      kExample, config,
      {{"model.path", model},
       {"model.labels",
        (example_common_config_path(kExample).parent_path() / "coco_label.txt").string()},
       {"input.codec", codec},
       {"input.width", "1280"},
       {"input.height", "720"},
       {"input.fps", "0"},
       {"output.video_enabled", "true"},
       {"output.insight.host", "127.0.0.1"},
       {"output.insight.max_visible_streams", "16"},
       {"output.insight.video_port_base", std::to_string(video)},
       {"output.insight.metadata_port_base", std::to_string(port)}},
      {{"streams", std::vector<std::string>(16, url)}});
  MetadataJsonListenerOptions options;
  options.base_port = port;
  options.num_ports = 16;
  options.timeout_ms = 200;
  MetadataJsonListener listener(options);
  require(listener.ok(), listener.error());
  using Tracker = high_density::testing::ReceivedMetadataTracker;
  const auto start = Tracker::Clock::now();
  Tracker tracker(16, 100, 5000,
                  std::chrono::milliseconds(env_int_or_default(
                      "SIMANEAT_APPS_HIGH_DENSITY_INITIAL_PROGRESS_TIMEOUT_MS", 90000)),
                  std::chrono::milliseconds(env_int_or_default(
                      "SIMANEAT_APPS_HIGH_DENSITY_STREAM_PROGRESS_TIMEOUT_MS", 30000)),
                  start);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  // One receiver round: true once the frame target was reached or the receiver failed.
  std::string receiver_failure;
  const auto receiver_done = [&]() -> bool {
    try {
      const auto metadata = listener.wait_for_messages();
      require(metadata.error.empty() || metadata.timed_out,
              "metadata listener failed: " + metadata.error);
      for (const auto& message : metadata.messages) {
        const auto [index, payload] = validate_metadata_message(message, port);
        tracker.observe(static_cast<std::size_t>(index),
                        payload.at("frame_id").get<std::string>(), message.object_count > 0,
                        Tracker::Clock::now());
        if (tracker.warmup_complete() && !tracker.measurement_started()) {
          // Timing includes the flush, so concurrent arrivals can only lower reported FPS.
          tracker.start_measurement(Tracker::Clock::now());
          const auto drained = listener.drain_pending();
          require(drained.error.empty(),
                  "metadata listener failed during warmup drain: " + drained.error);
          for (const auto& queued_message : drained.messages) {
            (void)validate_metadata_message(queued_message, port);
          }
          break;
        }
        if (tracker.complete()) {
          break;
        }
      }

      const auto stalled = tracker.stalled_streams(Tracker::Clock::now());
      if (!stalled.empty()) {
        std::ostringstream detail;
        detail << (tracker.measurement_started() ? "ongoing" : "initial")
               << " metadata progress timeout; missing streams=";
        for (std::size_t i = 0; i < stalled.size(); ++i) {
          if (i != 0)
            detail << ',';
          detail << stalled[i];
        }
        detail << " warmup=" << json(tracker.warmup_frames()).dump()
               << " measured=" << json(tracker.measured_frames()).dump();
        receiver_failure = detail.str();
        return true;
      }
    } catch (const std::exception& error) {
      receiver_failure = error.what();
      return true;
    }
    return tracker.complete();
  };

  // spawn_until() sleeps 100 ms between calls, and each round reads one datagram, so
  // a single round per call would starve all but the first of the 16 metadata
  // sockets. Keep receiving for a slice instead; returning between slices still lets
  // spawn_until() notice an early exit or the overall timeout.
  // spawn_until() sends SIGINT as soon as this returns true; remember when, so the
  // shutdown can be held to this example's limit rather than the harness's longer grace.
  constexpr auto kReceiveSlice = std::chrono::seconds(1);
  std::optional<std::chrono::steady_clock::time_point> stop_requested_at;
  const auto ready = [&] {
    const auto slice_end = std::chrono::steady_clock::now() + kReceiveSlice;
    bool done = false;
    while (!done && std::chrono::steady_clock::now() < slice_end) {
      done = receiver_done();
    }
    if (done) {
      stop_requested_at = std::chrono::steady_clock::now();
    }
    return done;
  };

  const auto result = spawn_until(binary, {"--config", config.string()}, ready, timeout_ms);
  const std::string shutdown_problem = exit_problem(result);
  require(shutdown_problem.empty(), "application did not shut down cleanly: " +
                                        shutdown_problem + "\n" + result.stdout_text +
                                        result.stderr_text);
  if (result.stopped_by_harness && stop_requested_at) {
    const auto shutdown_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now() - *stop_requested_at)
                                 .count();
    require(shutdown_ms <= kSigintExitLimitMs,
            "application took " + std::to_string(shutdown_ms) + " ms to exit after SIGINT; " +
                "the limit is " + std::to_string(kSigintExitLimitMs) + " ms\n" +
                result.stdout_text + result.stderr_text);
  }
  if (receiver_failure.empty() && !result.stopped_by_harness) {
    receiver_failure = "application exited before the receiver reached its frame target";
  }
  require(receiver_failure.empty(), receiver_failure + "\n" + result.stdout_text +
                                        result.stderr_text);
  require(tracker.complete(), "receiver did not reach 5000 unique measured metadata frames");
  require(std::all_of(tracker.useful_detection().begin(), tracker.useful_detection().end(),
                      [](bool useful) { return useful; }),
          "missing useful detections on one or more streams");

  const double elapsed = tracker.elapsed_seconds();
  const double fps = static_cast<double>(tracker.total_measured()) / elapsed;
  const json summary = {{"frames", tracker.total_measured()},
                        {"elapsed_s", elapsed},
                        {"aggregate_fps", fps},
                        {"per_stream_frames", tracker.measured_frames()}};
  std::cout << codec << ": " << summary.dump() << "\n";
  require(tracker.total_measured() == 5000, "expected exactly 5000 measured frames");
  require(elapsed > 0, "invalid measurement interval");
  require(fps > 450, "metadata throughput must exceed 450 FPS");
  for (const auto count : tracker.measured_frames())
    require(static_cast<double>(count) / elapsed > 28.125,
            "per-stream throughput must exceed 28.125 FPS");
  remove_dir(output);
}
} // namespace

int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  const char* raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const auto model = configured_model_path(kExample, raw ? raw : "models");
  if (model.empty() || !fs::exists(model))
    return skip_or_fail("configured high-density model is unavailable");
  int rc = 0;
  int cases_run = 0;
  for (const std::string codec : {"h264", "h265"}) {
    const auto urls = codec == "h264" ? rtsp_h264_urls_from_env() : rtsp_h265_urls_from_env();
    if (urls.empty()) {
      if (skip_or_fail("no RTSP " + codec + " URLs configured") != kSkipCode)
        rc = 1;
      continue;
    }
    ++cases_run;
    try {
      run_case(argv[1], model, codec, urls);
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] " << codec << ": " << error.what() << "\n";
      rc = 1;
    }
  }
  return cases_run == 0 && rc == 0 ? kSkipCode : rc;
}
