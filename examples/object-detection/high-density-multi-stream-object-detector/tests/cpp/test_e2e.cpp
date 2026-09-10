#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace sima_examples::testing;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
constexpr const char* kExample = "high-density-multi-stream-object-detector";

void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::string select_source(const std::vector<std::string>& urls, const std::string& codec) {
  std::set<std::string> seen;
  for (const auto& url : urls) {
    if (!seen.insert(url).second)
      continue;
    const auto probe = spawn_and_wait("/usr/bin/env",
                                      {"ffprobe", "-v", "error", "-rtsp_transport", "tcp",
                                       "-select_streams", "v:0", "-show_entries",
                                       "stream=codec_name,width,height,avg_frame_rate,has_b_frames",
                                       "-of", "json", url},
                                      20000);
    if (probe.exit_code != 0) {
      std::cerr << probe.stderr_text;
      continue;
    }
    const auto streams = json::parse(probe.stdout_text).value("streams", json::array());
    if (streams.empty())
      continue;
    const auto& stream = streams.front();
    const std::string rate = stream.value("avg_frame_rate", "0/1");
    std::istringstream input(rate);
    double numerator = 0, denominator = 0;
    char separator = 0;
    input >> numerator >> separator >> denominator;
    const double fps = denominator > 0 ? numerator / denominator : 0;
    if (stream.value("codec_name", "") == (codec == "h264" ? "h264" : "hevc") &&
        stream.value("width", 0) == 1280 && stream.value("height", 0) == 720 &&
        stream.value("has_b_frames", -1) == 0 &&
        (std::abs(fps - 30) < 0.001 || std::abs(fps - 30000.0 / 1001) < 0.001)) {
      std::cout << "source codec=" << codec << " width=1280 height=720 fps=" << fps
                << " unique_publishers=1\n";
      return url;
    }
    std::cerr << "Unsuitable source: " << stream.dump() << "\n";
  }
  throw std::runtime_error("No 720p30 " + codec + " source without B-frames");
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
       {"input.fps", "30"},
       {"runtime.warmup_frames", "100"},
       {"runtime.profile", "false"},
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
  auto process = std::async(std::launch::async, [&] {
    return spawn_and_wait("/usr/bin/env",
                          {"HIGH_DENSITY_DETECTOR_MEASURE_FRAMES=5000",
                           "HIGH_DENSITY_DETECTOR_FRAMES_PER_STREAM=0", binary,
                           "--config", config.string()},
                          env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000));
  });
  std::vector<MetadataJsonMessage> messages;
  auto drain_until = std::chrono::steady_clock::time_point::max();
  while (true) {
    const auto metadata = listener.wait_for_messages();
    messages.insert(messages.end(), metadata.messages.begin(), metadata.messages.end());
    if (process.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      const auto now = std::chrono::steady_clock::now();
      if (metadata.messages.empty() || now >= drain_until)
        break;
      drain_until = std::min(drain_until, now + std::chrono::seconds(1));
    }
  }
  const auto result = process.get();
  require(result.exit_code == 0, result.stdout_text + result.stderr_text);
  std::vector<std::set<std::string>> received(16);
  std::set<int> detected_streams;
  for (const auto& message : messages) {
    const auto payload = json::parse(message.payload);
    const int index = message.port - port;
    require(payload.at("stream_index") == index &&
                payload.at("stream_id") == "stream" + std::to_string(index),
            "wrong metadata channel");
    require(!payload.at("frame_id").get<std::string>().empty() &&
                payload.at("pts_ns").get<int64_t>() >= 0 && payload.contains("rtp_timestamp"),
            "missing frame identity");
    received.at(index).insert(payload.at("frame_id").get<std::string>());
    if (message.object_count > 0)
      detected_streams.insert(index);
  }
  require(detected_streams.size() == 16, "missing detections on one or more streams");
  json summary;
  int summaries = 0;
  std::istringstream lines(result.stdout_text);
  for (std::string line; std::getline(lines, line);) {
    if (line.rfind("[measurement] ", 0) == 0) {
      summary = json::parse(line.substr(14));
      ++summaries;
    }
  }
  require(summaries == 1, "missing or repeated measurement summary");
  std::cout << codec << ": " << summary.dump() << "\n";
  require(summary.at("frames") == 5000, "expected 5000 measured frames");
  const auto counts = summary.at("per_stream_frames").get<std::vector<int>>();
  require(counts.size() == 16, "expected 16 measured streams");
  int total = 0;
  for (const int count : counts) {
    total += count;
  }
  require(total == 5000, "per-stream counts disagree with total");
  const double elapsed = summary.at("elapsed_s");
  const double fps = summary.at("aggregate_fps");
  require(elapsed > 0 && std::abs(fps - 5000.0 / elapsed) < 0.001, "invalid measurement rate");
  require(fps > 450, "metadata throughput must exceed 450 FPS");
  for (const int count : counts)
    require(count / elapsed > 450.0 / 16, "per-stream throughput must exceed 28.125 FPS");
  std::vector<std::size_t> received_counts;
  for (const auto& frames : received)
    received_counts.push_back(frames.size());
  require(json(received_counts) == summary.at("per_stream_total_sent"),
          "received unique metadata counts disagree with successful sends");
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
  for (const std::string codec : {"h264", "h265"}) {
    try {
      run_case(argv[1], model, codec,
               codec == "h264" ? rtsp_h264_urls_from_env() : rtsp_h265_urls_from_env());
    } catch (const std::exception& error) {
      std::cerr << "[FAIL] " << codec << ": " << error.what() << "\n";
      rc = 1;
    }
  }
  return rc;
}
