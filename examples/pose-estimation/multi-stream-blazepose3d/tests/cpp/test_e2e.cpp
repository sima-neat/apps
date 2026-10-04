#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-blazepose3d";
constexpr const char* kInsightHost = "127.0.0.1";
constexpr const char* kPoseModel = "blazepose_ghum_heavy_modalix_bf16_mpk.tar.gz";
constexpr const char* kDetectorModel = "yolo26m-det-int8-b1.tar.gz";
constexpr std::size_t kMaxStreams = 4;

// Counts the UDP datagrams that arrive on each Insight video port.
class VideoListener {
public:
  VideoListener(int base_port, int num_ports) : packets_(static_cast<std::size_t>(num_ports), 0) {
    for (int offset = 0; offset < num_ports; ++offset) {
      const int fd = socket(AF_INET, SOCK_DGRAM, 0);
      if (fd < 0) {
        error_ = "failed to create video UDP socket";
        return;
      }
      const timeval timeout{0, 100000};
      setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_port = htons(static_cast<std::uint16_t>(base_port + offset));
      address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      if (bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close(fd);
        error_ = "failed to bind video UDP port " + std::to_string(base_port + offset);
        return;
      }
      sockets_.push_back(fd);
    }
    for (std::size_t index = 0; index < sockets_.size(); ++index) {
      workers_.emplace_back([this, index]() { receive(index); });
    }
  }

  ~VideoListener() {
    stopping_ = true;
    for (std::thread& worker : workers_) {
      worker.join();
    }
    for (const int fd : sockets_) {
      close(fd);
    }
  }

  bool ok() const {
    return error_.empty() && sockets_.size() == packets_.size();
  }
  const std::string& error() const {
    return error_;
  }

  bool received_all_ports() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::all_of(packets_.begin(), packets_.end(), [](int packets) { return packets > 0; });
  }

private:
  void receive(std::size_t index) {
    std::array<std::uint8_t, 65536> packet{};
    while (!stopping_) {
      if (recv(sockets_[index], packet.data(), packet.size(), 0) > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++packets_[index];
      }
    }
  }

  std::vector<int> sockets_;
  std::vector<std::thread> workers_;
  mutable std::mutex mutex_;
  std::vector<int> packets_;
  std::atomic<bool> stopping_{false};
  std::string error_;
};

void write_config(const fs::path& path, const fs::path& detector, const fs::path& pose,
                  const std::string& codec, const std::vector<std::string>& urls,
                  int video_port_base, int metadata_port_base) {
  std::ofstream output(path);
  output << "models:\n  detector_path: " << detector.string() << "\n  pose_path: " << pose.string()
         << "\nstreams:\n";
  for (std::size_t index = 0; index < urls.size(); ++index) {
    output << "  - id: camera" << index << "\n    url: " << urls[index] << "\n    codec: " << codec
           << "\n    insight_channel: " << index << "\n";
  }
  output << "input:\n  tcp: true\n  latency_ms: 100\n"
            "detector:\n  min_score: 0.30\n  nms_iou: 0.60\n"
            "pose:\n  max_people_per_frame: 2\n  roi_scale: 1.65\n  presence_threshold: 0.0\n"
            "runtime:\n  frames: 30\noutput:\n  insight:\n    host: "
         << kInsightHost << "\n    video_port_base: " << video_port_base
         << "\n    metadata_port_base: " << metadata_port_base << "\n";
}

// Every message carries its port's stream id and 33 world keypoints per pose, and
// at least one port publishes a non-empty 2D/3D pair for one frame.
bool validate_metadata(const MetadataJsonListenerResult& result, int metadata_port_base,
                       std::string& error) {
  using FrameKey = std::tuple<int, int64_t, std::string>;
  std::map<FrameKey, std::size_t> pose_frames;
  std::map<FrameKey, std::size_t> world_pose_frames;
  try {
    for (const auto& message : result.messages) {
      const auto data = nlohmann::json::parse(message.payload).at("data");
      if (data.value("stream_id", "") !=
          "camera" + std::to_string(message.port - metadata_port_base)) {
        error = "metadata on port " + std::to_string(message.port) + " did not carry its stream id";
        return false;
      }
      const bool overlay = message.metadata_type == "pose-estimation";
      if (!overlay &&
          (data.value("id", "") != "world-pose" || data.value("renderer", "") != "blazepose-3d")) {
        error = "auxiliary metadata did not use the world-pose BlazePose 3D envelope";
        return false;
      }
      const auto& poses = overlay ? data.at("poses") : data.at("payload").at("poses");
      for (const auto& pose : poses) {
        const char* points = overlay ? "world_keypoints" : "keypoints";
        if (!pose.contains(points) || pose[points].size() != 33) {
          error = "a " + message.metadata_type + " pose did not carry 33 world keypoints";
          return false;
        }
      }
      (overlay ? pose_frames
               : world_pose_frames)[{message.port, message.timestamp_ms, message.frame_id}] =
          poses.size();
    }
  } catch (const std::exception& exception) {
    error = std::string("failed to validate pose metadata: ") + exception.what();
    return false;
  }
  const bool paired = std::any_of(pose_frames.begin(), pose_frames.end(), [&](const auto& entry) {
    const auto world = world_pose_frames.find(entry.first);
    return entry.second > 0 && world != world_pose_frames.end() && world->second == entry.second;
  });
  if (!paired) {
    error = "no stream published a non-empty 2D/3D BlazePose pair for one frame";
  }
  return paired;
}

int run_case(const std::string& binary, const fs::path& detector, const fs::path& pose,
             const std::string& codec, const std::vector<std::string>& urls) {
  const fs::path run_dir = create_test_scratch_dir(kExampleName, "e2e_" + codec);
  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 300000);
  const fs::path config = run_dir / "config.yaml";
  write_config(config, detector, pose, codec, urls, video_port_base, metadata_port_base);

  MetadataJsonListenerOptions listener_options;
  listener_options.host = kInsightHost;
  listener_options.base_port = metadata_port_base;
  const int num_ports = static_cast<int>(urls.size());
  listener_options.num_ports = num_ports;
  listener_options.timeout_ms = 10000;
  // A frame without people publishes nothing, so not every port must publish
  // metadata; every port must still receive video.
  listener_options.contracts = {{"pose-estimation", "poses", 0},
                                {"auxiliary-visualization", "payload.poses", 0}};
  MetadataJsonListener listener(listener_options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener: " << listener.error() << "\n";
    remove_dir(run_dir.string());
    return 1;
  }
  VideoListener video_listener(video_port_base, num_ports);
  if (!video_listener.ok()) {
    std::cerr << "[FAIL] video listener: " << video_listener.error() << "\n";
    remove_dir(run_dir.string());
    return 1;
  }

  const ProcessResult process = spawn_and_wait(binary, {"--config", config.string()}, timeout_ms);
  // The application has exited; drain the buffered metadata until one port holds
  // a non-empty 2D/3D pair, a datagram is invalid, or 10 s pass.
  MetadataJsonListenerResult metadata;
  std::string metadata_error = "timed out waiting for a non-empty 2D/3D metadata pair";
  const auto metadata_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < metadata_deadline && metadata.error.empty()) {
    listener.poll_messages(metadata, 250);
    std::string error;
    if (validate_metadata(metadata, metadata_port_base, error)) {
      metadata_error.clear();
      break;
    }
    metadata_error = error;
  }
  if (!metadata.error.empty()) {
    metadata_error = metadata.error;
  }
  const std::string exit_error = exit_problem(process);
  int result = 0;
  if (!exit_error.empty()) {
    std::cerr << "[FAIL] " << codec << " app " << exit_error << "\nstdout:\n"
              << process.stdout_text << "\nstderr:\n"
              << process.stderr_text << "\n";
    result = 1;
  } else if (!video_listener.received_all_ports()) {
    std::cerr << "[FAIL] " << codec << " did not emit video on every video port\n";
    result = 1;
  } else if (!metadata_error.empty()) {
    std::cerr << "[FAIL] " << codec << " metadata: " << metadata_error << "\n";
    result = 1;
  } else {
    std::cout << "[OK] " << codec << " produced video on " << urls.size()
              << " streams and paired 2D/3D metadata\n";
  }
  remove_dir(run_dir.string());
  return result;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const auto env_path = [](const char* name, const fs::path& fallback) {
    const char* value = env_or_null(name);
    return value != nullptr ? fs::path(value) : fallback;
  };
  const fs::path models_dir = env_path("SIMANEAT_APPS_TEST_MODELS_DIR", "models");
  const fs::path detector =
      env_path("SIMANEAT_APPS_TEST_DETECTOR_MODEL", models_dir / kDetectorModel);
  const fs::path pose = env_path("SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL", models_dir / kPoseModel);
  if (!fs::exists(detector) || !fs::exists(pose)) {
    return skip_or_fail("YOLO26 and BlazePose model packages are required");
  }

  std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"h264", rtsp_h264_urls_from_env()}, {"h265", rtsp_h265_urls_from_env()}};
  int result = 0;
  int cases_run = 0;
  for (auto& [codec, urls] : cases) {
    // The application accepts at most four streams; CI provides five per codec.
    if (urls.size() > kMaxStreams) {
      urls.resize(kMaxStreams);
    }
    if (urls.empty()) {
      continue;
    }
    ++cases_run;
    result |= run_case(argv[1], detector, pose, codec, urls);
  }
  if (cases_run == 0) {
    return skip_or_fail("no multi-stream H.264 or H.265 RTSP URLs configured");
  }
  return result;
}
