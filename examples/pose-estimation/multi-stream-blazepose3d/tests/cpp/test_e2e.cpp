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
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
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

struct SourceCaps {
  int width = 0;
  int height = 0;
  int fps = 0;
};

SourceCaps probe_source_caps(const std::string& url) {
  const auto probe =
      spawn_and_wait("/usr/bin/env",
                     {"ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-select_streams", "v:0",
                      "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", url},
                     20000);
  if (probe.exit_code != 0) {
    throw std::runtime_error("failed to probe E2E source " + url + ": " + probe.stderr_text);
  }
  const auto streams =
      nlohmann::json::parse(probe.stdout_text).value("streams", nlohmann::json::array());
  if (streams.size() != 1) {
    throw std::runtime_error("E2E source must expose one video stream: " + url);
  }
  const auto& stream = streams.front();
  const std::string rate = stream.value("avg_frame_rate", "0/1");
  std::istringstream input(rate);
  double numerator = 0.0;
  double denominator = 0.0;
  char separator = 0;
  input >> numerator >> separator >> denominator;
  const SourceCaps caps{stream.value("width", 0), stream.value("height", 0),
                        denominator > 0.0 ? static_cast<int>(std::round(numerator / denominator))
                                          : 0};
  if (caps.width <= 0 || caps.height <= 0 || caps.fps <= 0) {
    throw std::runtime_error("failed to resolve E2E source caps: " + url);
  }
  return caps;
}

struct RtpPayload {
  const std::uint8_t* data;
  std::size_t size;
};

std::optional<RtpPayload> rtp_payload(const std::uint8_t* packet, std::size_t size) {
  if (size < 13 || packet[0] >> 6 != 2 || (packet[1] & 0x7F) != 96) {
    return std::nullopt;
  }
  std::size_t header_size = 12 + 4 * (packet[0] & 0x0F);
  if (header_size >= size) {
    return std::nullopt;
  }
  if ((packet[0] & 0x10) != 0) {
    if (header_size + 4 > size) {
      return std::nullopt;
    }
    const std::size_t extension_words =
        (static_cast<std::size_t>(packet[header_size + 2]) << 8) | packet[header_size + 3];
    if (extension_words > (size - header_size - 4) / 4) {
      return std::nullopt;
    }
    header_size += 4 + 4 * extension_words;
    if (header_size >= size) {
      return std::nullopt;
    }
  }
  std::size_t payload_end = size;
  if ((packet[0] & 0x20) != 0) {
    const std::size_t padding = packet[size - 1];
    if (padding == 0 || padding > payload_end - header_size) {
      return std::nullopt;
    }
    payload_end -= padding;
  }
  if (header_size >= payload_end) {
    return std::nullopt;
  }
  return RtpPayload{packet + header_size, payload_end - header_size};
}

bool is_codec_config_rtp(const std::uint8_t* packet, std::size_t size, const std::string& codec) {
  const auto parsed = rtp_payload(packet, size);
  if (!parsed.has_value()) {
    return false;
  }
  const std::uint8_t* payload = parsed->data;
  const std::size_t payload_size = parsed->size;
  if (codec == "h264") {
    const std::uint8_t nal_type = payload[0] & 0x1F;
    if ((payload[0] & 0x80) != 0) {
      return false;
    }
    if (nal_type == 7) {
      return payload_size >= 2;
    }
    if (nal_type == 28) {
      return payload_size >= 3 && (payload[1] & 0xC0) == 0x80 && (payload[1] & 0x20) == 0 &&
             (payload[1] & 0x1F) == 7;
    }
    if (nal_type != 24) {
      return false;
    }
    std::size_t position = 1;
    bool found_sps = false;
    std::uint8_t max_nri = 0;
    while (position < payload_size) {
      if (position + 2 > payload_size) {
        return false;
      }
      const std::size_t nal_size =
          (static_cast<std::size_t>(payload[position]) << 8) | payload[position + 1];
      position += 2;
      if (nal_size == 0 || nal_size > payload_size - position) {
        return false;
      }
      if ((payload[position] & 0x80) != 0) {
        return false;
      }
      max_nri = std::max(max_nri, static_cast<std::uint8_t>(payload[position] & 0x60));
      found_sps = found_sps || (nal_size >= 2 && (payload[position] & 0x9F) == 7);
      position += nal_size;
    }
    return found_sps && (payload[0] & 0x60) == max_nri;
  }
  if (payload_size < 2 || (payload[0] & 0x80) != 0 || (payload[1] & 0x07) == 0) {
    return false;
  }
  const std::uint8_t nal_type = (payload[0] >> 1) & 0x3F;
  if (nal_type == 32) {
    return payload_size >= 3;
  }
  if (nal_type == 49) {
    return payload_size >= 4 && (payload[2] & 0xC0) == 0x80 && (payload[2] & 0x3F) == 32;
  }
  if (nal_type != 48) {
    return false;
  }
  std::size_t position = 2;
  bool found_vps = false;
  while (position < payload_size) {
    if (position + 2 > payload_size) {
      return false;
    }
    const std::size_t nal_size =
        (static_cast<std::size_t>(payload[position]) << 8) | payload[position + 1];
    position += 2;
    if (nal_size < 2 || nal_size > payload_size - position) {
      return false;
    }
    const std::uint8_t* nal = payload + position;
    found_vps = found_vps || (nal_size >= 3 && (nal[0] & 0x80) == 0 &&
                              ((nal[0] >> 1) & 0x3F) == 32 && (nal[1] & 0x07) != 0);
    position += nal_size;
  }
  return found_vps;
}

bool test_rtp_packet_validation() {
  const auto packet = [](std::initializer_list<std::uint8_t> payload) {
    std::vector<std::uint8_t> bytes{0x80, 96, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    bytes.reserve(bytes.size() + payload.size());
    for (const std::uint8_t value : payload) {
      bytes.push_back(value);
    }
    return bytes;
  };
  const auto accepted = [](const std::vector<std::uint8_t>& bytes, const std::string& codec) {
    return is_codec_config_rtp(bytes.data(), bytes.size(), codec);
  };
  const auto h264 = packet({0x67, 1});
  const auto h265 = packet({0x40, 1, 1});
  const auto overlapping_h264 = packet({0x61, 1});
  const auto overlapping_h265 = packet({0x02, 1});
  const auto h264_stap = packet({0x78, 0, 2, 0x67, 1});
  const auto h264_fu = packet({28, 0x87, 1});
  const auto h265_ap = packet({0x60, 1, 0, 3, 0x40, 1, 1});
  const auto h265_fu = packet({0x62, 1, 0xA0, 1});
  auto wrong_payload = h264;
  wrong_payload[1] = 97;
  return accepted(h264, "h264") && !accepted(h264, "h265") && accepted(h265, "h265") &&
         !accepted(h265, "h264") && !accepted(overlapping_h264, "h264") &&
         !accepted(overlapping_h264, "h265") && !accepted(overlapping_h265, "h264") &&
         !accepted(overlapping_h265, "h265") && accepted(h264_stap, "h264") &&
         accepted(h264_fu, "h264") && accepted(h265_ap, "h265") && accepted(h265_fu, "h265") &&
         !accepted(packet({0x67}), "h264") && !accepted(packet({0x40, 1}), "h265") &&
         !accepted(packet({24, 0, 1, 0x67}), "h264") &&
         !accepted(packet({24, 0, 2, 0x67, 1}), "h264") &&
         !accepted(packet({0x60, 1, 0, 2, 0x40, 1}), "h265") && !accepted(packet({28}), "h264") &&
         !accepted(packet({0x62, 1}), "h265") && !accepted(packet({28, 0xC7, 1}), "h264") &&
         !accepted(packet({0x62, 1, 0xE0, 1}), "h265") && !accepted(wrong_payload, "h264");
}

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
      const ssize_t size = recv(sockets_[index], packet.data(), packet.size(), 0);
      if (size > 0 && is_codec_config_rtp(packet.data(), static_cast<std::size_t>(size), codec_)) {
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
    const SourceCaps caps = probe_source_caps(urls[index]);
    output << "  - id: camera" << index << "\n    url: " << urls[index] << "\n    codec: " << codec
           << "\n    insight_channel: " << index << "\n    width: " << caps.width
           << "\n    height: " << caps.height << "\n    fps: " << caps.fps << "\n";
  }
  output << "input:\n  tcp: true\n  latency_ms: 100\n"
            "detector:\n  min_score: 0.30\n  nms_iou: 0.60\n"
            "pose:\n  max_people_per_frame: 2\n  roi_scale: 1.65\n  presence_threshold: 0.0\n"
            "runtime:\n  frames: 30\noutput:\n  insight:\n    host: "
         << kInsightHost << "\n    video_port_base: " << video_port_base
         << "\n    metadata_port_base: " << metadata_port_base << "\n";
}

// True when pose[name] holds 33 points with finite coordinates on every required axis.
bool has_body_points(const nlohmann::json& pose, const char* name, bool require_z) {
  const auto has_finite_coordinate = [](const nlohmann::json& point, const char* axis) {
    return point.contains(axis) && point.at(axis).is_number() &&
           std::isfinite(point.at(axis).get<double>());
  };
  return pose.contains(name) && pose.at(name).size() == 33 &&
         std::all_of(pose.at(name).begin(), pose.at(name).end(), [&](const nlohmann::json& point) {
           return has_finite_coordinate(point, "x") && has_finite_coordinate(point, "y") &&
                  (!require_z || has_finite_coordinate(point, "z"));
         });
}

// Every message carries its port's stream id and 33 valid keypoints per pose (image
// and world for 2D poses), and at least one port publishes a non-empty 2D/3D pair
// for one frame.
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
        if (!has_body_points(pose, "keypoints", !overlay) ||
            (overlay && !has_body_points(pose, "world_keypoints", true))) {
          error = "a " + message.metadata_type + " pose did not carry 33 valid keypoints";
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
  // Every accepted frame publishes a pair, even without people, so every port
  // must receive a correlated 2D/3D pair.
  listener_options.require_all_ports = true;
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
  // The application has exited; drain the buffered metadata until every port holds
  // a 2D/3D pair, one of them non-empty, a datagram is invalid, or 10 s pass.
  MetadataJsonListenerResult metadata;
  std::string metadata_error = "timed out waiting for a non-empty 2D/3D metadata pair";
  const auto metadata_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (std::chrono::steady_clock::now() < metadata_deadline && metadata.error.empty()) {
    listener.poll_messages(metadata, 250);
    std::string error;
    if (validate_metadata(metadata, metadata_port_base, error) && metadata.success) {
      metadata_error.clear();
      break;
    }
    metadata_error = error.empty() ? "not every metadata port received a 2D/3D pair" : error;
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
  if (!test_rtp_packet_validation()) {
    std::cerr << "[FAIL] RTP payload type and codec-header validation\n";
    return 1;
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
