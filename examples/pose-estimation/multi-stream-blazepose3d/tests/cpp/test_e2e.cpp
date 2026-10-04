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
#include <mutex>
#include <set>
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

class RtpVideoListener {
public:
  RtpVideoListener(int base_port, int num_ports, std::string codec)
      : codec_(std::move(codec)), packets_(static_cast<std::size_t>(num_ports), 0) {
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

  ~RtpVideoListener() {
    stopping_ = true;
    for (std::thread& worker : workers_) {
      worker.join();
    }
    for (const int fd : sockets_) {
      close(fd);
    }
  }

  bool ok() const { return error_.empty() && sockets_.size() == packets_.size(); }
  const std::string& error() const { return error_; }

  bool received_all_ports() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::all_of(packets_.begin(), packets_.end(), [](int packets) { return packets > 0; });
  }

private:
  bool is_video_rtp(const std::uint8_t* packet, std::size_t size) const {
    if (size < 13 || packet[0] >> 6 != 2) {
      return false;
    }
    std::size_t header_size = 12 + 4 * (packet[0] & 0x0F);
    if (header_size >= size) {
      return false;
    }
    if ((packet[0] & 0x10) != 0) {
      if (header_size + 4 > size) {
        return false;
      }
      const std::size_t extension_words =
          (static_cast<std::size_t>(packet[header_size + 2]) << 8) | packet[header_size + 3];
      header_size += 4 + 4 * extension_words;
      if (header_size >= size) {
        return false;
      }
    }
    if (codec_ == "h264") {
      const std::uint8_t nal_type = packet[header_size] & 0x1F;
      return (packet[header_size] & 0x80) == 0 && nal_type >= 1 && nal_type <= 29;
    }
    if (header_size + 2 > size) {
      return false;
    }
    const std::uint8_t nal_type = (packet[header_size] >> 1) & 0x3F;
    return (packet[header_size] & 0x80) == 0 && nal_type <= 49 &&
           (packet[header_size + 1] & 0x07) != 0;
  }

  void receive(std::size_t index) {
    std::array<std::uint8_t, 65536> packet{};
    while (!stopping_) {
      const ssize_t size = recv(sockets_[index], packet.data(), packet.size(), 0);
      if (size > 0 && is_video_rtp(packet.data(), static_cast<std::size_t>(size))) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++packets_[index];
      }
    }
  }

  std::vector<int> sockets_;
  std::vector<std::thread> workers_;
  std::string codec_;
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
            "detector:\n  min_score: 0.30\n  nms_iou: 0.60\n  max_detections: 100\n"
            "  max_inflight_per_stream: 4\n"
            "pose:\n  max_people_per_frame: 2\n  roi_scale: 1.65\n"
            "  presence_threshold: 0.0\n  job_timeout_ms: 10000\n  max_pending_jobs: 64\n"
            "runtime:\n  frames: 8\noutput:\n  insight:\n    host: "
         << kInsightHost << "\n    video_port_base: " << video_port_base
         << "\n    metadata_port_base: " << metadata_port_base << "\n  video_enabled: true\n";
}

bool validate_metadata(const MetadataJsonListenerResult& result, int metadata_port_base,
                       int num_ports, std::string& error) {
  bool found_pose = false;
  bool found_world_pose = false;
  using FrameKey = std::tuple<int, int64_t, std::string>;
  std::set<FrameKey> pose_frames;
  std::set<FrameKey> world_pose_frames;
  try {
    for (const auto& message : result.messages) {
      const auto parsed = nlohmann::json::parse(message.payload);
      const FrameKey frame{message.port, message.timestamp_ms, message.frame_id};
      const std::string expected_stream_id =
          "camera" + std::to_string(message.port - metadata_port_base);
      if (parsed.at("data").at("stream_id") != expected_stream_id) {
        error = "metadata did not preserve the configured stream identity";
        return false;
      }
      if (message.metadata_type == "pose-estimation") {
        const auto& poses = parsed.at("data").at("poses");
        for (const auto& pose : poses) {
          found_pose = true;
          if (!pose.contains("keypoints") || !pose["keypoints"].is_array() ||
              pose["keypoints"].size() != 33 || !pose.contains("world_keypoints") ||
              !pose["world_keypoints"].is_array() || pose["world_keypoints"].size() != 33 ||
              !pose.contains("presence") || !pose["presence"].is_number() ||
              pose["presence"].get<double>() < 0.0 || pose["presence"].get<double>() > 1.0) {
            error = "a published pose did not contain presence plus 33 image and world keypoints";
            return false;
          }
          for (const auto& point : pose["world_keypoints"]) {
            if (!point.contains("name") || !point["name"].is_string() || !point.contains("x") ||
                !point["x"].is_number() || !point.contains("y") || !point["y"].is_number() ||
                !point.contains("z") || !point["z"].is_number() ||
                !point.contains("confidence") || !point["confidence"].is_number()) {
              error = "a pose-estimation world keypoint lacked name/x/y/z/confidence";
              return false;
            }
          }
        }
        if (!poses.empty()) {
          pose_frames.insert(frame);
        }
        continue;
      }

      const auto& data = parsed.at("data");
      if (data.at("schema_version") != 1 || data.at("id") != "world-pose" ||
          data.at("renderer") != "blazepose-3d") {
        error = "auxiliary metadata did not use the world-pose BlazePose 3D contract";
        return false;
      }
      const auto& poses = data.at("payload").at("poses");
      for (const auto& pose : poses) {
        found_world_pose = true;
        if (!pose.contains("keypoints") || !pose["keypoints"].is_array() ||
            pose["keypoints"].size() != 33 || !pose.contains("presence") ||
            !pose["presence"].is_number() || pose["presence"].get<double>() < 0.0 ||
            pose["presence"].get<double>() > 1.0) {
          error = "a published 3D pose did not contain presence and exactly 33 keypoints";
          return false;
        }
        for (const auto& point : pose["keypoints"]) {
          if (!point.contains("name") || !point["name"].is_string() || !point.contains("x") ||
              !point["x"].is_number() || !point.contains("y") || !point["y"].is_number() ||
              !point.contains("z") || !point["z"].is_number() || !point.contains("confidence") ||
              !point["confidence"].is_number()) {
            error = "a published 3D keypoint did not contain name/x/y/z/confidence";
            return false;
          }
        }
      }
      if (!poses.empty()) {
        world_pose_frames.insert(frame);
      }
    }
  } catch (const std::exception& exception) {
    error = std::string("failed to validate pose metadata: ") + exception.what();
    return false;
  }
  if (!found_pose) {
    error = "no 2D BlazePose result was published";
    return false;
  }
  if (!found_world_pose) {
    error = "no 3D BlazePose result was published";
    return false;
  }
  for (int port = metadata_port_base; port < metadata_port_base + num_ports; ++port) {
    const bool matched =
        std::any_of(pose_frames.begin(), pose_frames.end(), [&](const auto& frame) {
          return std::get<0>(frame) == port && world_pose_frames.count(frame) != 0;
        });
    if (!matched) {
      error = "2D and 3D metadata did not share a frame identity on port " + std::to_string(port);
      return false;
    }
  }
  return true;
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
  listener_options.require_all_ports = true;
  // Every accepted frame publishes a pair, including empty pose arrays, so the
  // listener accepts empty arrays and validate_metadata() requires a non-empty
  // 2D/3D pair with one frame identity on every port.
  listener_options.contracts = {{"pose-estimation", "poses", 0},
                                {"auxiliary-visualization", "payload.poses", 0}};
  MetadataJsonListener listener(listener_options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener: " << listener.error() << "\n";
    remove_dir(run_dir.string());
    return 1;
  }
  RtpVideoListener video_listener(video_port_base, num_ports, codec);
  if (!video_listener.ok()) {
    std::cerr << "[FAIL] video listener: " << video_listener.error() << "\n";
    remove_dir(run_dir.string());
    return 1;
  }

  const ProcessResult process = spawn_and_wait(binary, {"--config", config.string()}, timeout_ms);
  // The application has exited; drain the buffered metadata until every port
  // holds a non-empty correlated pair, a datagram is invalid, or 10 s pass.
  MetadataJsonListenerResult metadata;
  std::string metadata_error = "timed out waiting for paired 2D/3D metadata on every port";
  const auto metadata_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < metadata_deadline && metadata.error.empty()) {
    listener.poll_messages(metadata, 250);
    std::string error;
    if (validate_metadata(metadata, metadata_port_base, num_ports, error)) {
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
    std::cerr << "[FAIL] " << codec << " did not emit valid RTP on every video port\n";
    result = 1;
  } else if (!metadata_error.empty()) {
    std::cerr << "[FAIL] " << codec << " metadata: " << metadata_error << "\n";
    result = 1;
  } else {
    std::cout << "[OK] " << codec << " produced paired 2D/3D metadata on " << urls.size()
              << " streams\n";
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
  const fs::path models_dir = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR") != nullptr
                                  ? fs::path(env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR"))
                                  : fs::path("models");
  const fs::path detector = env_or_null("SIMANEAT_APPS_TEST_DETECTOR_MODEL") != nullptr
                                ? fs::path(env_or_null("SIMANEAT_APPS_TEST_DETECTOR_MODEL"))
                                : models_dir / kDetectorModel;
  const fs::path pose = env_or_null("SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL") != nullptr
                            ? fs::path(env_or_null("SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL"))
                            : models_dir / kPoseModel;
  if (!fs::exists(detector) || !fs::exists(pose)) {
    return skip_or_fail("YOLO26 and BlazePose model packages are required");
  }

  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"h264", rtsp_h264_urls_from_env()}, {"h265", rtsp_h265_urls_from_env()}};
  int result = 0;
  int cases_run = 0;
  for (const auto& [codec, urls] : cases) {
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
