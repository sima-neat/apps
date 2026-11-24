// End-to-end tests: RTSP streams reach Insight as video and tracking metadata, each on its own
// channel. tests/python/test_e2e.py runs the same checks in Python.
#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/source_cases.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "yolo26-tiny-drone-tracker";
constexpr int kFrames = 150;
// Above the shipped 0.05, so the tracks show that the example applies inference.min_score.
constexpr double kMinScore = 0.1;
// Each stream's results per second. Not verified on a DevKit yet.
constexpr double kMinStreamResultsPerS = 20.0;

using Clock = std::chrono::steady_clock;

// RTP payload types: H.264 96, H.265 98.
int payload_type(const std::string& codec) {
  return codec == "h264" ? 96 : 98;
}

std::string regex_escape(const std::string& text) {
  static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
  return std::regex_replace(text, special, R"(\$&)");
}

// Receives one stream's video and keeps each packet's RTP header until `stop` is set. A copy of
// single-stream's helper: #823 moves it into support/cpp/sima_examples/testing/rtp_video.
class VideoDrain {
public:
  VideoDrain(int port, const std::atomic<bool>& stop) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    timeval timeout{0, 200000};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ok_ = fd_ >= 0 && ::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (ok_) {
      thread_ = std::thread([this, &stop] {
        std::vector<unsigned char> packet(65536);
        for (;;) {
          const auto size = ::recv(fd_, packet.data(), packet.size(), 0);
          if (size >= 12) {
            std::array<unsigned char, 12> header{};
            std::copy_n(packet.begin(), header.size(), header.begin());
            headers_.push_back(header);
          } else if (stop) {
            return;
          }
        }
      });
    }
  }
  ~VideoDrain() {
    if (thread_.joinable()) {
      thread_.join();
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }
  bool ok() const {
    return ok_;
  }
  // Call after `stop` is set.
  const std::vector<std::array<unsigned char, 12>>& headers() {
    if (thread_.joinable()) {
      thread_.join();
    }
    return headers_;
  }

private:
  int fd_ = -1;
  bool ok_ = false;
  std::vector<std::array<unsigned char, 12>> headers_;
  std::thread thread_;
};

// Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a video
// frame's RTP timestamp.
bool pairs(std::int64_t timestamp_ms, const std::set<std::uint32_t>& rtp_timestamps) {
  const auto ticks = static_cast<std::uint32_t>(timestamp_ms * 90);
  return std::any_of(rtp_timestamps.begin(), rtp_timestamps.end(), [&](std::uint32_t rtp) {
    return std::llabs(static_cast<long long>(ticks) - static_cast<long long>(rtp)) <= 90;
  });
}

int run_case(const std::string& binary, const std::string& model_path,
             const MultiStreamSourceCase& source_case, int streams, const std::string& label) {
  const std::string output_dir = create_test_output_dir(
      kExampleName, "test_streams_reach_their_insight_channels_as_video_and_tracks_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path config_dir = fs::path(output_dir).parent_path();
  const fs::path config_path = config_dir / "config.yaml";
  const std::string relative_model =
      fs::relative(fs::absolute(model_path), fs::absolute(config_dir)).string();
  // The app runs from the Apps root, so only a path joined to the config file finds the model.
  if (fs::exists(relative_model)) {
    std::cerr << "[FAIL] " << label << " the working directory also finds " << relative_model
              << "\n";
    return 1;
  }
  const std::vector<std::string> urls(source_case.urls.begin(), source_case.urls.begin() + streams);
  write_e2e_config(kExampleName, config_path,
                   {// Relative to the generated config, so the test runs the app's path join.
                    {"model.path", relative_model},
                    {"input.codec", source_case.codec},
                    {"inference.frames", std::to_string(kFrames)},
                    {"inference.min_score", std::to_string(kMinScore)},
                    // This test is the Insight receiver, so the example sends to loopback.
                    {"output.insight.host", "127.0.0.1"},
                    {"output.insight.video_port_base", std::to_string(video_port)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port)}},
                   {{"input.source", urls}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = "127.0.0.1";
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = streams;
  metadata_options.metadata_type = "tracking";
  metadata_options.data_array_key = "tracks";
  MetadataJsonListener listener(metadata_options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] " << label << " cannot bind the metadata ports: " << listener.error()
              << "\n";
    return 1;
  }
  std::atomic<bool> stop{false};
  std::vector<std::unique_ptr<VideoDrain>> videos;
  for (int index = 0; index < streams; ++index) {
    videos.push_back(std::make_unique<VideoDrain>(video_port + index, stop));
    if (!videos.back()->ok()) {
      std::cerr << "[FAIL] " << label << " cannot bind video port " << video_port + index << "\n";
      stop = true;
      return 1;
    }
  }

  // Read during the run, so no message waits long enough to overflow the socket buffer.
  MetadataJsonListenerResult metadata;
  std::map<int, std::vector<Clock::time_point>> arrivals;
  const auto read_metadata = [&] {
    const std::size_t before = metadata.messages.size();
    // Once a message has arrived, poll_messages returns after each datagram, so read until the
    // sockets are empty.
    listener.poll_messages(metadata, 100);
    while (listener.poll_messages(metadata, 0)) {
    }
    for (std::size_t i = before; i < metadata.messages.size(); ++i) {
      arrivals[metadata.messages[i].port - metadata_port].push_back(Clock::now());
    }
    return false; // The application stops by itself after inference.frames.
  };
  const ProcessResult result =
      spawn_until(binary, {"--config", config_path.string()}, read_metadata, timeout_ms);
  const auto rest = listener.drain_pending();
  metadata.messages.insert(metadata.messages.end(), rest.messages.begin(), rest.messages.end());
  stop = true;

  std::vector<std::string> problems;
  if (const std::string problem = exit_problem(result); !problem.empty()) {
    problems.push_back(problem + "\nstdout:\n" + result.stdout_text + "\nstderr:\n" +
                       result.stderr_text);
  }
  for (const std::string& error : {metadata.error, rest.error}) {
    if (!error.empty()) {
      problems.push_back("metadata listener: " + error);
    }
  }
  for (int index = 0; index < streams; ++index) {
    const std::string stream = "stream " + std::to_string(index);
    const std::regex start_line("(^|\n)\\[" + stream +
                                "\\] rtsp=" + regex_escape(urls[static_cast<std::size_t>(index)]) +
                                " stream=\\d+x\\d+@\\d+ insight=127\\.0\\.0\\.1 video=" +
                                std::to_string(video_port + index) +
                                " metadata=" + std::to_string(metadata_port + index) + "\n");
    if (!std::regex_search(result.stdout_text, start_line)) {
      problems.push_back(stream + " did not print its source and ports");
    }
    std::smatch summary;
    const std::regex summary_line("(^|\n)\\[" + stream + "\\] processed=" +
                                  std::to_string(kFrames) + " tracks_started=(\\d+)\n");
    int tracks_started = 0;
    if (std::regex_search(result.stdout_text, summary, summary_line)) {
      tracks_started = std::stoi(summary[2]);
    } else {
      problems.push_back(stream + " did not report processed=" + std::to_string(kFrames) +
                         " tracks_started=<count>");
    }

    std::set<int> payload_types;
    std::set<std::uint32_t> rtp_timestamps;
    for (const auto& header : videos[static_cast<std::size_t>(index)]->headers()) {
      payload_types.insert(header[1] & 0x7F);
      rtp_timestamps.insert(static_cast<std::uint32_t>(header[4]) << 24 |
                            static_cast<std::uint32_t>(header[5]) << 16 |
                            static_cast<std::uint32_t>(header[6]) << 8 | header[7]);
    }
    if (payload_types != std::set<int>{payload_type(source_case.codec)}) {
      problems.push_back(stream + " video did not arrive with RTP payload type " +
                         std::to_string(payload_type(source_case.codec)));
    }

    int messages = 0;
    int tracks = 0;
    long long previous_frame_id = -1;
    std::vector<std::int64_t> unpaired;
    for (const auto& message : metadata.messages) {
      if (message.port != metadata_port + index) {
        continue;
      }
      ++messages;
      const long long frame_id = std::stoll(message.frame_id);
      if (frame_id <= previous_frame_id) {
        problems.push_back(stream + ": frame IDs out of order: " + message.frame_id + " after " +
                           std::to_string(previous_frame_id));
      }
      previous_frame_id = frame_id;
      // These checks cover the tracks the run sent. A stream without drones sends none, and the
      // count printed below shows how many there were.
      for (const auto& track : nlohmann::json::parse(message.payload).at("data").at("tracks")) {
        const double confidence = track.at("confidence").get<double>();
        const int id = std::stoi(track.at("id").get<std::string>());
        if (track.at("label") != "drone" || confidence < kMinScore || confidence > 1.0 || id < 1 ||
            id > tracks_started) {
          problems.push_back(stream + " frame " + message.frame_id + ": " + track.dump());
        }
        ++tracks;
      }
      // Insight draws each message's tracks on the video frame its timestamp names.
      if (!pairs(message.timestamp_ms, rtp_timestamps)) {
        unpaired.push_back(message.timestamp_ms);
      }
    }
    // UDP on loopback may drop a message under load, so allow a few to go missing.
    if (messages < kFrames * 95 / 100) {
      problems.push_back(stream + ": " + std::to_string(messages) + " of " +
                         std::to_string(kFrames) + " messages arrived");
    }
    if (!unpaired.empty()) {
      problems.push_back(stream + ": " + std::to_string(unpaired.size()) +
                         " timestamps with no video frame, first " +
                         std::to_string(unpaired.front()));
    }
    const auto& times = arrivals[index];
    const std::size_t first = times.size() / 10;
    if (times.size() - first <= 10) {
      problems.push_back(stream + ": " + std::to_string(times.size() - first) +
                         " messages arrived during the run");
      continue;
    }
    const double seconds = std::chrono::duration<double>(times.back() - times[first]).count();
    const double results_per_s = static_cast<double>(times.size() - first - 1) / seconds;
    std::cout << label << " " << stream << ": " << messages << " messages, " << results_per_s
              << " results/s, " << tracks << " tracks in " << tracks_started << " track IDs\n";
    if (results_per_s < kMinStreamResultsPerS) {
      problems.push_back(stream + ": " + std::to_string(results_per_s) + " results/s");
    }
  }
  remove_dir(output_dir);
  for (const auto& problem : problems) {
    std::cerr << "[FAIL] " << label << " " << problem << "\n";
  }
  if (problems.empty()) {
    std::cout << "[OK] " << label << " reached its Insight channels as video and tracks\n";
  }
  return problems.empty() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("configured tracker model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }
  const MultiStreamSourceCase h264{"h264", rtsp_h264_urls_from_env()};
  const MultiStreamSourceCase h265{"h265", rtsp_h265_urls_from_env()};
  // With one source, Core names the samples after the decoder instead of the stream.
  const std::vector<std::tuple<MultiStreamSourceCase, int, std::string>> cases = {
      {h264, 2, "h264"}, {h265, 2, "h265"}, {h264, 1, "h264-one-stream"}};
  int cases_run = 0;
  for (const auto& [source_case, streams, label] : cases) {
    if (source_case.urls.size() < static_cast<std::size_t>(streams)) {
      record_unavailable_case("SIMANEAT_TEST_RTSP_" +
                                  std::string(source_case.codec == "h264" ? "H264" : "H265") +
                                  "_URLS needs " + std::to_string(streams) + " URLs for " + label,
                              "not enough " + source_case.codec + " URLs for " + label, rc);
      continue;
    }
    ++cases_run;
    if (run_case(binary, model_paths.front(), source_case, streams, label) != 0) {
      rc = 1;
    }
  }
  if (cases_run == 0) {
    return rc != 0 ? rc : skip_or_fail("no RTSP URLs configured for " + std::string(kExampleName));
  }
  return rc;
}
