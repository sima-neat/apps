#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>

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

int qualification_int(const char* suffix, int fallback, int minimum, int maximum) {
  const std::string name = std::string("SIMANEAT_APPS_TEST_HD_") + suffix;
  const char* raw = env_or_null(name.c_str());
  if (!raw) return fallback;
  std::size_t used = 0;
  const int value = std::stoi(raw, &used);
  require(used == std::string(raw).size() && value >= minimum && value <= maximum,
          name + " must be in [" + std::to_string(minimum) + ", " +
              std::to_string(maximum) + "], got " + raw);
  return value;
}

std::string select_source(const std::vector<std::string>& urls, const std::string& codec,
                          int expected_fps) {
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
        (std::abs(fps - expected_fps) < 0.001 ||
         (expected_fps == 30 && std::abs(fps - 30000.0 / 1001) < 0.001))) {
      std::cout << "source codec=" << codec << " width=1280 height=720 fps=" << fps
                << " unique_publishers=1\n";
      return url;
    }
    std::cerr << "Unsuitable source: width=" << width << " height=" << height
              << " fps=" << fps << "\n";
  }
  throw std::runtime_error("No 720p" + std::to_string(expected_fps) + " " + codec + " source");
}

void run_case(const std::string& binary, const std::string& model, const std::string& codec,
              const std::vector<std::string>& urls) {
  const int streams = qualification_int("STREAMS", 16, 1, 80);
  const int source_fps = qualification_int("SOURCE_FPS", 30, 1, 120);
  const int frames = qualification_int("MEASURE_FRAMES", 5000, streams, 100000000);
  const char* profile_env = env_or_null("SIMANEAT_APPS_TEST_HD_PROFILE");
  const std::string profile = profile_env ? profile_env : "config.yaml";
  require(profile == "config.yaml" || profile == "config-24x720p20fps.yaml" ||
              profile == "config-48x720p10fps.yaml", "HD_PROFILE must name a shipped density profile");
  auto overrides = sima_examples::ScalarConfig::load(
      example_common_config_path(kExample).parent_path() / profile).scalars();
  for (const auto& setting : {std::make_pair("DECODER_BUFFERS", "input.decoder_buffers"),
                              std::make_pair("INPUT_BUFFERS", "input.decoder_input_buffers")}) {
    if (env_or_null((std::string("SIMANEAT_APPS_TEST_HD_") + setting.first).c_str()))
      overrides[setting.second] = std::to_string(qualification_int(setting.first, 1, 1,
          std::string(setting.first) == "DECODER_BUFFERS" ? 64 : 2147483647));
  }
  const auto url = select_source(urls, codec, source_fps);
  const auto output = create_test_output_dir(kExample, "metadata_throughput_" + codec);
  require(!output.empty(), "cannot create output directory");
  const auto config = fs::path(output).parent_path() / "config.yaml";
  const int port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int video = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const ConfigScalars harness_overrides =
      {{"model.path", model},
       {"model.labels",
        (example_common_config_path(kExample).parent_path() / "coco_label.txt").string()},
       {"input.codec", codec},
       {"input.width", "1280"},
       {"input.height", "720"},
       {"runtime.warmup_frames", "100"},
       {"runtime.profile", "false"},
       {"output.video_enabled", "true"},
       {"output.insight.host", "127.0.0.1"},
       {"output.insight.max_visible_streams", std::to_string(streams)},
       {"output.insight.video_port_base", std::to_string(video)},
       {"output.insight.metadata_port_base", std::to_string(port)}};
  for (const auto& item : harness_overrides) overrides[item.first] = item.second;
  write_e2e_config(kExample, config, overrides,
                   {{"streams", std::vector<std::string>(streams, url)}});
  MetadataJsonListenerOptions options;
  options.base_port = port;
  options.num_ports = streams;
  options.timeout_ms = 200;
  MetadataJsonListener listener(options);
  require(listener.ok(), listener.error());
  auto process = std::async(std::launch::async, [&] {
    return spawn_and_wait("/usr/bin/env",
                          {"HIGH_DENSITY_DETECTOR_MEASURE_FRAMES=" + std::to_string(frames),
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
  std::vector<std::set<std::string>> received(streams);
  std::set<int> detected_streams;
  std::vector<int64_t> last_pts(streams, -1);
  for (const auto& message : messages) {
    const auto payload = json::parse(message.payload);
    const int index = message.port - port;
    require(payload.at("stream_index") == index &&
                payload.at("stream_id") == "stream" + std::to_string(index),
            "wrong metadata channel");
    require(!payload.at("frame_id").get<std::string>().empty() &&
                payload.at("pts_ns").get<int64_t>() >= 0 && payload.contains("rtp_timestamp"),
            "missing frame identity");
    require(received.at(index).insert(payload.at("frame_id").get<std::string>()).second,
            "duplicate frame identity");
    const auto pts = payload.at("pts_ns").get<int64_t>();
    require(pts > last_pts.at(index), "non-increasing per-stream timestamp");
    last_pts.at(index) = pts;
    if (message.object_count > 0)
      detected_streams.insert(index);
  }
  require(detected_streams.size() == static_cast<std::size_t>(streams), "missing detections on one or more streams");
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
  require(summary.at("frames") == frames, "wrong measured frame count");
  const auto counts = summary.at("per_stream_frames").get<std::vector<int>>();
  require(counts.size() == static_cast<std::size_t>(streams), "wrong measured stream count");
  int total = 0;
  for (const int count : counts) {
    total += count;
  }
  require(total == frames, "per-stream counts disagree with total");
  const double elapsed = summary.at("elapsed_s");
  const double fps = summary.at("aggregate_fps");
  require(elapsed > 0 && std::abs(fps - static_cast<double>(frames) / elapsed) < 0.001, "invalid measurement rate");
  const double minimum_fps = source_fps == 30 ? 28.125 : 0.95 * source_fps;
  require(fps > streams * minimum_fps, "aggregate metadata rate below target");
  for (const int count : counts)
    require(count / elapsed > minimum_fps, "per-stream metadata rate below target");
  require(summary.at("per_stream_send_failures") == json(std::vector<int>(streams, 0)),
          "metadata send failures");
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
