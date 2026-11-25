// End-to-end tests: four RTSP streams, each with its own model, reach Insight as video and as that
// model's metadata, each on its own channel. tests/python/test_e2e.py runs the same checks in
// Python.
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
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-multi-model";
constexpr int kFrames = 150;
// Above the shipped 0.30, so the results show that the example applies inference.min_score.
constexpr double kMinScore = 0.35;
// Each stream's results per second, as a share of the frame rate the app printed for it. A DevKit
// sent 19.9 to 20.0 results/s per stream on the CI streams (e83d42f). That log does not show their
// rate, so the summary line below prints it.
constexpr double kMinShareOfStreamRate = 0.95;

using Clock = std::chrono::steady_clock;

// The shipped config's streams, in order: the package test-scope.yaml downloads for each, the
// Insight metadata type its model sends, and the key of the entries in that type's data.
struct StreamCase {
  std::string file;
  std::string metadata_type;
  std::string key;
};
const std::vector<StreamCase> kStreams = {
    {"yolo11s_w8a8_mpk.tar.gz", "object-detection", "objects"},
    {"yolo26s_seg_w8a16_mpk.tar.gz", "segmentation", "segments"},
    {"yolo26m_pose_w8a8_mpk.tar.gz", "pose-estimation", "poses"},
    {"yolo26m_w8a8_mpk.tar.gz", "object-detection", "objects"},
};

// RTP payload types: H.264 96, H.265 98.
int payload_type(const std::string& codec) {
  return codec == "h264" ? 96 : 98;
}

std::string regex_escape(const std::string& text) {
  static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
  return std::regex_replace(text, special, R"(\$&)");
}

std::set<std::string> coco_labels() {
  std::ifstream file("examples/multi-model/multi-stream-multi-model/src/common/coco_label.txt");
  std::set<std::string> labels;
  for (std::string line; std::getline(file, line);) {
    if (!line.empty()) {
      labels.insert(line);
    }
  }
  return labels;
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

// Why `entry` is not a result the stream's model can send, or "" when it is.
std::string entry_problem(const std::string& metadata_type, const nlohmann::json& entry,
                          const std::set<std::string>& labels) {
  const double confidence = entry.at("confidence").get<double>();
  if (confidence < kMinScore || confidence > 1.0) {
    return "confidence below inference.min_score";
  }
  const std::string label = entry.at("label").get<std::string>();
  if (metadata_type == "pose-estimation") {
    const auto& keypoints = entry.at("keypoints");
    if (label != "person" || keypoints.size() != 17U || keypoints[0].at("name") != "nose") {
      return "not a person with the 17 COCO keypoints";
    }
    return "";
  }
  if (labels.count(label) == 0) {
    return "label not in coco_label.txt";
  }
  if (metadata_type == "segmentation" && entry.at("mask").size() < 3U) {
    return "mask outline with fewer than three points";
  }
  return "";
}

int run_case(const std::string& binary, const std::string& models_dir,
             const MultiStreamSourceCase& source_case) {
  const std::string label = source_case.codec;
  const std::string output_dir = create_test_output_dir(
      kExampleName,
      "test_each_stream_reaches_its_insight_channel_as_video_and_its_models_results_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path config_dir = fs::path(output_dir).parent_path();
  const fs::path config_path = config_dir / "config.yaml";
  const int streams = static_cast<int>(kStreams.size());
  std::vector<std::string> model_paths;
  std::vector<std::string> sources;
  // The shipped entries, each with its path relative to the generated config, so the test runs
  // the app's path join. The labels path starts at the working directory, as shipped.
  const auto shipped = example_common_config(kExampleName).mapping_list("model");
  std::vector<ConfigScalars> models;
  for (int index = 0; index < streams; ++index) {
    const fs::path model = fs::path(models_dir) / kStreams[static_cast<std::size_t>(index)].file;
    model_paths.push_back(fs::relative(fs::absolute(model), fs::absolute(config_dir)).string());
    // The app runs in this test's working directory, so only a path joined to the config file
    // finds a model.
    if (fs::exists(model_paths.back())) {
      std::cerr << "[FAIL] " << label << " the working directory also finds "
                << model_paths.back() << "\n";
      return 1;
    }
    sources.push_back(source_case.urls[static_cast<std::size_t>(index) % source_case.urls.size()]);
    const auto& entry = shipped.at(static_cast<std::size_t>(index));
    models.push_back({{"id", entry.string("id")}, {"path", model_paths.back()}});
    if (const auto labels = entry.string_value("labels")) {
      // The app runs in this test's working directory, so the shipped labels path must resolve
      // from there.
      if (!fs::is_regular_file(*labels)) {
        std::cerr << "[FAIL] " << label << " the working directory does not find " << *labels
                  << "\n";
        return 1;
      }
      models.back()["labels"] = *labels;
    }
  }
  write_e2e_config(kExampleName, config_path,
                   {{"input.codec", source_case.codec},
                    {"inference.frames", std::to_string(kFrames)},
                    {"inference.min_score", std::to_string(kMinScore)},
                    // This test is the Insight receiver, so the example sends to loopback.
                    {"output.insight.host", "127.0.0.1"},
                    {"output.insight.video_port_base", std::to_string(video_port)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port)}},
                   {{"input.source", sources}}, {{"model", models}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = "127.0.0.1";
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = streams;
  for (const auto& stream : kStreams) {
    metadata_options.contracts.push_back({stream.metadata_type, stream.key, 0});
  }
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

  // Read during the run and at exit, so no message waits long enough to overflow the socket
  // buffer.
  MetadataJsonListenerResult metadata;
  std::map<int, std::vector<Clock::time_point>> arrivals;
  const auto read_metadata = [&] {
    const std::size_t before = metadata.messages.size();
    // Wait up to 100 ms for a datagram, then read every one already queued: MetadataSender sends
    // a message above 1200 bytes, such as a frame's poses, as several chunks.
    listener.poll_messages(metadata, 100);
    listener.read_ready(metadata);
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

  const auto labels = coco_labels();
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
  bool stream_0_saw_person = false;
  for (int index = 0; index < streams; ++index) {
    const auto& stream_case = kStreams[static_cast<std::size_t>(index)];
    const std::string stream = "stream " + std::to_string(index);
    const std::regex start_line("(^|\n)\\[" + stream + "\\] [a-z0-9-]+ model=" +
                                regex_escape(model_paths[static_cast<std::size_t>(index)]) +
                                " rtsp=" + regex_escape(sources[static_cast<std::size_t>(index)]) +
                                " stream=\\d+x\\d+@(\\d+) insight=127\\.0\\.0\\.1 video=" +
                                std::to_string(video_port + index) +
                                " metadata=" + std::to_string(metadata_port + index) + "\n");
    std::smatch start_match;
    int stream_fps = 0;
    if (std::regex_search(result.stdout_text, start_match, start_line)) {
      stream_fps = std::stoi(start_match[2].str());
    } else {
      problems.push_back(stream + " did not print its model, source, and ports");
    }
    const std::regex summary_line("(^|\n)\\[" + stream + "\\] processed=" +
                                  std::to_string(kFrames) + " left_out=0\n");
    if (!std::regex_search(result.stdout_text, summary_line)) {
      problems.push_back(stream + " did not report processed=" + std::to_string(kFrames) +
                         " left_out=0");
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
    int entries = 0;
    long long previous_frame_id = -1;
    std::vector<std::int64_t> unpaired;
    for (const auto& message : metadata.messages) {
      if (message.port != metadata_port + index) {
        continue;
      }
      ++messages;
      if (message.metadata_type != stream_case.metadata_type) {
        problems.push_back(stream + " sent " + message.metadata_type + " metadata");
      }
      if (message.frame_id.empty()) {
        problems.push_back(stream + ": a message without a frame ID");
        continue;
      }
      const long long frame_id = std::stoll(message.frame_id);
      if (frame_id <= previous_frame_id) {
        problems.push_back(stream + ": frame IDs out of order: " + message.frame_id + " after " +
                           std::to_string(previous_frame_id));
      }
      previous_frame_id = frame_id;
      // A named payload: a range-for over parse(...).at(...) would keep only the reference at()
      // returns, into a parse result destroyed before the loop.
      const nlohmann::json payload = nlohmann::json::parse(message.payload);
      for (const auto& entry : payload.at("data").at(stream_case.key)) {
        if (const std::string problem = entry_problem(stream_case.metadata_type, entry, labels);
            !problem.empty()) {
          problems.push_back(stream + " frame " + message.frame_id + ": " + problem);
        }
        stream_0_saw_person |= index == 0 && entry.at("label") == "person";
        ++entries;
      }
      // Insight draws each message's entries on the video frame its timestamp names.
      if (!pairs(message.timestamp_ms, rtp_timestamps)) {
        unpaired.push_back(message.timestamp_ms);
      }
    }
    // UDP on loopback may drop a message under load, so allow a few to go missing.
    if (messages < kFrames * 95 / 100) {
      problems.push_back(stream + ": " + std::to_string(messages) + " of " +
                         std::to_string(kFrames) + " messages arrived");
    }
    // The detection and segmentation models find objects in every test stream. The pose model
    // needs a person in view: when stream 0 found a person on the same URL, the pose stream must
    // send poses. The pose counts seen on a DevKit are not verified yet.
    if (stream_case.metadata_type != "pose-estimation" && entries == 0) {
      problems.push_back(stream + " sent no " + stream_case.key);
    }
    if (stream_case.metadata_type == "pose-estimation" && entries == 0 && stream_0_saw_person &&
        sources[static_cast<std::size_t>(index)] == sources.front()) {
      problems.push_back(stream + " sent no poses while stream 0 saw a person");
    }
    // A frame the video link dropped leaves its message unpaired, so allow 5%, as for the count.
    if (static_cast<int>(unpaired.size()) > messages * 5 / 100) {
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
              << " results/s of " << stream_fps << " fps, " << entries << " " << stream_case.key
              << "\n";
    if (results_per_s < kMinShareOfStreamRate * stream_fps) {
      problems.push_back(stream + ": " + std::to_string(results_per_s) + " results/s of a " +
                         std::to_string(stream_fps) + " fps stream");
    }
  }
  remove_dir(output_dir);
  for (const auto& problem : problems) {
    std::cerr << "[FAIL] " << label << " " << problem << "\n";
  }
  if (problems.empty()) {
    std::cout << "[OK] " << label << " reached its Insight channels as video and results\n";
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
  for (const auto& stream : kStreams) {
    if (!fs::is_regular_file(fs::path(models_dir) / stream.file)) {
      return skip_or_fail("model not found under SIMANEAT_APPS_TEST_MODELS_DIR: " + stream.file);
    }
  }
  int rc = 0;
  int cases_run = 0;
  for (const MultiStreamSourceCase& source_case :
       {MultiStreamSourceCase{"h264", rtsp_h264_urls_from_env()},
        MultiStreamSourceCase{"h265", rtsp_h265_urls_from_env()}}) {
    if (source_case.urls.empty()) {
      const std::string env = source_case.codec == "h264" ? "H264" : "H265";
      record_unavailable_case("SIMANEAT_TEST_RTSP_" + env + "_URLS needs 1 URL",
                              "no " + source_case.codec + " URLs", rc);
      continue;
    }
    ++cases_run;
    if (run_case(binary, models_dir, source_case) != 0) {
      rc = 1;
    }
  }
  if (cases_run == 0) {
    return rc != 0 ? rc : skip_or_fail("no RTSP URLs configured for " + std::string(kExampleName));
  }
  return rc;
}
