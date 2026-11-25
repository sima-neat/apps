// End-to-end tests: RTSP streams reach Insight as video and oriented-box metadata, each on its
// own channel. tests/python/test_e2e.py runs the same checks in Python.
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

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
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

constexpr const char* kExampleName = "multi-stream-obb-detector";
constexpr int kFrames = 150;
// Above the shipped 0.30, so the boxes, when the streams show any, show that the example applies
// inference.min_score.
constexpr double kMinScore = 0.4;
// The rate check runs on the suite's model only. Larger variants run slower.
constexpr const char* kDefaultModel = "yolo26n_obb_w8a8_mpk.tar.gz";
// Each stream's results per second, about 80% of Vulcan CI run 45418480248 on 5baed52 (Modalix
// devkit-1, firmware 3.0.1_master_B2798, Core develop/530580ae4d76, 25 fps sources). With one or
// two streams, each stream gave 24.6 to 25.2, the source's rate.
constexpr double kMinStreamFps = 20.0;

using Clock = std::chrono::steady_clock;

// RTP payload types: H.264 96, H.265 98, and RTP/JPEG 26.
int payload_type(const std::string& codec) {
  return codec == "h264" ? 96 : codec == "h265" ? 98 : 26;
}

// Registry package files start with their model ID, such as yolov8n_obb_w8a8_mpk.tar.gz.
std::string model_id(const std::string& model_path) {
  const std::string name = fs::path(model_path).filename().string();
  for (const char* family : {"yolov8", "yolo11"}) {
    if (name.rfind(family, 0) == 0) {
      return family;
    }
  }
  return "yolo26";
}

std::string regex_escape(const std::string& text) {
  static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
  return std::regex_replace(text, special, R"(\$&)");
}

std::set<std::string> read_labels() {
  std::set<std::string> labels;
  std::ifstream input(example_common_config_path(kExampleName).parent_path() / "dota_label.txt");
  for (std::string line; std::getline(input, line);) {
    if (!line.empty()) {
      labels.insert(line);
    }
  }
  return labels;
}

// Receives one stream's video and keeps each packet's RTP payload type until `stop` is set.
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
          if (size >= 2) {
            payload_types_.insert(packet[1] & 0x7F);
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
  const std::set<int>& payload_types() {
    if (thread_.joinable()) {
      thread_.join();
    }
    return payload_types_;
  }

private:
  int fd_ = -1;
  bool ok_ = false;
  std::set<int> payload_types_;
  std::thread thread_;
};

std::string object_problem(const nlohmann::json& obj, const std::set<std::string>& labels) {
  if (obj.size() != 5 || !obj.contains("id") || !obj.contains("label") ||
      !obj.contains("confidence") || !obj.contains("bbox") || !obj.contains("oriented_bbox")) {
    return "unexpected keys";
  }
  if (labels.count(obj["label"].get<std::string>()) == 0) {
    return "label is not a DOTA class";
  }
  const double confidence = obj["confidence"].get<double>();
  if (confidence < kMinScore || confidence > 1.0) {
    return "confidence outside [min_score, 1]";
  }
  const auto& oriented = obj["oriented_bbox"];
  const auto& bbox = obj["bbox"];
  if (oriented.size() != 5 || bbox.size() != 4 || oriented[2].get<double>() <= 0.0 ||
      oriented[3].get<double>() <= 0.0 || !std::isfinite(oriented[4].get<double>())) {
    return "degenerate oriented box";
  }
  // Each value is rounded to 0.1 px, so the centers can differ by up to 0.1 px.
  if (std::abs(bbox[0].get<double>() + bbox[2].get<double>() / 2 - oriented[0].get<double>()) >
          0.15 ||
      std::abs(bbox[1].get<double>() + bbox[3].get<double>() / 2 - oriented[1].get<double>()) >
          0.15) {
    return "bbox is not centered on the oriented box";
  }
  return "";
}

int run_case(const std::string& binary, const std::string& model_path,
             const MultiStreamSourceCase& source_case, int streams, const std::string& label) {
  const std::string output_dir =
      create_test_output_dir(kExampleName, "test_streams_reach_their_insight_channels_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path config_dir = fs::path(output_dir).parent_path();
  const fs::path config_path = config_dir / "config.yaml";
  write_e2e_config(
      kExampleName, config_path,
      {// Relative to the generated config, so the test runs the app's path join.
       {"model.id", model_id(model_path)},
       {"model.path", fs::relative(fs::absolute(model_path), fs::absolute(config_dir)).string()},
       {"input.codec", source_case.codec},
       {"inference.frames", std::to_string(kFrames)},
       {"inference.min_score", std::to_string(kMinScore)},
       // This test is the Insight receiver, so the example sends to loopback.
       {"output.insight.host", "127.0.0.1"},
       {"output.insight.video_port_base", std::to_string(video_port)},
       {"output.insight.metadata_port_base", std::to_string(metadata_port)}},
      {{"input.source",
        std::vector<std::string>(source_case.urls.begin(), source_case.urls.begin() + streams)}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = "127.0.0.1";
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = streams;
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
    // Once the first message has arrived, poll_messages returns after each datagram, so read until
    // the sockets are empty.
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
    problems.push_back(problem);
  }
  for (const std::string& error : {metadata.error, rest.error}) {
    if (!error.empty()) {
      problems.push_back("metadata listener: " + error);
    }
  }
  const auto labels = read_labels();
  int boxes = 0;
  for (int index = 0; index < streams; ++index) {
    const std::string stream = "stream " + std::to_string(index);
    const std::regex start_line("\\[" + stream + "\\] rtsp=" +
                                regex_escape(source_case.urls[static_cast<std::size_t>(index)]) +
                                " stream=\\d+x\\d+@\\d+ insight=127\\.0\\.0\\.1 video=" +
                                std::to_string(video_port + index) +
                                " metadata=" + std::to_string(metadata_port + index) + "\n");
    if (!std::regex_search(result.stdout_text, start_line)) {
      problems.push_back(stream + " did not print its source and ports");
    }
    if (result.stdout_text.find("[" + stream + "] processed=" + std::to_string(kFrames)) ==
        std::string::npos) {
      problems.push_back(stream + " did not report processed=" + std::to_string(kFrames));
    }
    if (videos[static_cast<std::size_t>(index)]->payload_types() !=
        std::set<int>{payload_type(source_case.codec)}) {
      problems.push_back(stream + " video did not arrive with RTP payload type " +
                         std::to_string(payload_type(source_case.codec)));
    }
    int messages = 0;
    for (const auto& message : metadata.messages) {
      if (message.port != metadata_port + index) {
        continue;
      }
      ++messages;
      if (message.metadata_type != "object-detection") {
        problems.push_back(stream + " frame " + message.frame_id + ": type " +
                           message.metadata_type);
      }
      const auto payload = nlohmann::json::parse(message.payload);
      for (const auto& obj : payload.at("data").at("objects")) {
        if (const std::string problem = object_problem(obj, labels); !problem.empty()) {
          problems.push_back(stream + " frame " + message.frame_id + ": " + problem + " " +
                             obj.dump());
        }
        ++boxes;
      }
    }
    // UDP on loopback may drop a message under load, so allow a few to go missing.
    if (messages < kFrames * 95 / 100) {
      problems.push_back(stream + ": " + std::to_string(messages) + " of " +
                         std::to_string(kFrames) + " messages arrived");
    }
    const auto& times = arrivals[index];
    const std::size_t first = times.size() / 10;
    if (times.size() - first <= 10) {
      problems.push_back(stream + ": " + std::to_string(times.size() - first) +
                         " messages arrived during the run");
      continue;
    }
    const double seconds = std::chrono::duration<double>(times.back() - times[first]).count();
    const double fps = static_cast<double>(times.size() - first - 1) / seconds;
    std::cout << label << " " << stream << ": " << messages << " messages, " << fps
              << " results/s\n";
    if (fs::path(model_path).filename() == kDefaultModel && fps < kMinStreamFps) {
      problems.push_back(stream + ": " + std::to_string(fps) + " results/s");
    }
  }
  // No box is not a failure: the model finds objects in top-down aerial video, and the CI
  // streams show streets. Still, the C++ cases in CI run 45418480248 checked 1265 boxes on
  // h264, 680 on h264-one-stream, 810 on h265, and 242 on mjpeg. The unit tests check the
  // object values.
  std::cout << label << ": " << boxes << " oriented boxes checked\n";
  remove_dir(output_dir);
  for (const auto& problem : problems) {
    std::cerr << "[FAIL] " << label << " " << problem << "\n";
  }
  if (!problems.empty()) {
    std::cerr << "[FAIL] " << label << " application exit code " << result.exit_code
              << "\nstdout:\n"
              << result.stdout_text << "\nstderr:\n"
              << result.stderr_text << "\n";
  } else {
    std::cout << "[OK] " << label << " reached its Insight channels\n";
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
    return skip_or_fail("configured detector model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::vector<MultiStreamSourceCase> source_cases = {
      {"h264", rtsp_h264_urls_from_env()},
      {"h265", rtsp_h265_urls_from_env()},
      {"mjpeg", rtsp_mjpeg_urls_from_env()},
  };
  const int cases_rc = run_multistream_source_cases(
      "multi-stream OBB detector", source_cases, 2, [&](const MultiStreamSourceCase& source_case) {
        int case_rc = 0;
        for (const std::string& model_path : model_paths) {
          const std::string label =
              model_case_label(source_case.codec, model_path, model_paths.size());
          if (run_case(binary, model_path, source_case, 2, label) != 0) {
            case_rc = 1;
          }
          // With one source, Core names the samples after the decoder instead of the stream.
          if (source_case.codec == "h264" &&
              run_case(binary, model_path, source_case, 1, label + "-one-stream") != 0) {
            case_rc = 1;
          }
        }
        return case_rc;
      });
  return rc != 0 ? rc : cases_rc;
}
