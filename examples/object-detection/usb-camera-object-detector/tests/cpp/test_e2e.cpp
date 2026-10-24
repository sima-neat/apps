// E2E test for usb-camera-object-detector.
// There is no USB camera on the test target, so this drives the same graph from
// the fixed-image NV12 source declared under `testing.e2e` in config.yaml. That
// exercises the NV12 branch, video sender, model, box decode, and metadata send.
// USB capture and Input/JpegParse/SimaDecode require camera validation separately.
#include "support/testing/test_config.h"
#include "support/testing/metadata_json_listener.h"
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <fstream>
#include <thread>
#include <stdexcept>
#include <cerrno>
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr char kExample[] = "usb-camera-object-detector";
constexpr int kFrames = 30;

class VideoReceiver {
public:
  explicit VideoReceiver(const fs::path& path, int port) : path_(path) {
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(port);
    timeval timeout{0, 100000};
    if (socket_ < 0 || ::bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ||
        ::setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout))) {
      if (socket_ >= 0) ::close(socket_);
      throw std::runtime_error("could not bind video receiver");
    }
    worker_ = std::jthread([this](std::stop_token stop) { receive(stop); });
  }
  ~VideoReceiver() { close(); }
  void close() {
    worker_.request_stop();
    if (worker_.joinable()) worker_.join();
    if (socket_ >= 0) { ::close(socket_); socket_ = -1; }
  }
  void verify() {
    close();
    if (!error_.empty()) throw std::runtime_error(error_);
    const auto probe = spawn_and_wait("/usr/bin/env",
        {"ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_read_frames", "-of", "json", path_.string()}, 20000);
    if (probe.exit_code || !probe.stderr_text.empty())
      throw std::runtime_error("received H.264 did not decode: " + probe.stderr_text);
    const auto stream = nlohmann::json::parse(probe.stdout_text).at("streams").at(0);
    if (stream.at("width") != 1920 || stream.at("height") != 1080 ||
        std::stoi(stream.at("nb_read_frames").get<std::string>()) < 3)
      throw std::runtime_error("received video has wrong dimensions or fewer than 3 frames");
  }
private:
  void receive(std::stop_token stop) {
    try {
      std::ofstream out(path_, std::ios::binary);
      if (!out) throw std::runtime_error("cannot write received H.264");
      std::array<unsigned char, 65536> packet{};
      std::vector<unsigned char> fragment;
      int sequence = -1;
      const auto write_nal = [&](const unsigned char* data, std::size_t size) {
        out.write("\0\0\0\1", 4);
        out.write(reinterpret_cast<const char*>(data), size);
      };
      while (!stop.stop_requested()) {
        const auto size = ::recv(socket_, packet.data(), packet.size(), 0);
        if (size < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) continue;
        if (size <= 12 || packet[0] != 0x80 || (packet[1] & 0x7f) != 96)
          throw std::runtime_error("invalid H.264 RTP packet");
        const int current = (packet[2] << 8) | packet[3];
        if (sequence >= 0 && current != ((sequence + 1) & 65535))
          throw std::runtime_error("lost RTP packet");
        sequence = current;
        const auto* payload = packet.data() + 12;
        const int kind = payload[0] & 31;
        if (kind >= 1 && kind <= 23) {
          if (!fragment.empty()) throw std::runtime_error("incomplete RTP fragment");
          write_nal(payload, size - 12);
        } else if (kind == 28 && size > 14) {
          if (payload[1] & 0x80) {
            if (!fragment.empty()) throw std::runtime_error("incomplete RTP fragment");
            fragment.push_back((payload[0] & 0xe0) | (payload[1] & 31));
          }
          if (fragment.empty()) throw std::runtime_error("missing RTP fragment start");
          fragment.insert(fragment.end(), payload + 2, payload + size - 12);
          if (payload[1] & 0x40) {
            write_nal(fragment.data(), fragment.size());
            fragment.clear();
          }
        } else {
          throw std::runtime_error("unexpected H.264 RTP payload");
        }
      }
    } catch (const std::exception& error) { error_ = error.what(); }
  }
  fs::path path_;
  int socket_ = -1;
  std::string error_;
  std::jthread worker_;
};

void verify_metadata(const MetadataJsonListenerResult& result) {
  if (!result.success) throw std::runtime_error(result.error);
  for (const auto& message : result.messages) {
    if (message.timestamp_ms < 0) throw std::runtime_error("negative metadata timestamp");
    const auto objects = nlohmann::json::parse(message.payload).at("data").at("objects");
    if (objects.empty()) throw std::runtime_error("no objects detected in fixture");
    for (const auto& object : objects) {
      const auto box = object.at("bbox").get<std::vector<double>>();
      const auto confidence = object.at("confidence").get<double>();
      if (object.at("label").get<std::string>().empty() || !(confidence >= 0.3 && confidence <= 1) ||
          box.size() != 4 || !(box[0] >= 0 && box[1] >= 0 && box[2] > 0 && box[3] > 0 &&
                              box[0] + box[2] <= 1920 && box[1] + box[3] <= 1080))
        throw std::runtime_error("invalid detection metadata");
    }
  }
}

} // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  const std::string model_path = configured_model_path(kExample, models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured detection model not found under "
                        "SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  std::string labels_file;
  if (const char* labels_env = env_or_null("SIMANEAT_APPS_TEST_LABELS_FILE")) {
    labels_file = labels_env;
  }
  const std::string example_dir = fs::path(binary).parent_path().string();
  for (const auto& candidate :
       {std::string("examples/object-detection/usb-camera-object-detector/src/common/"
                    "coco_label.txt"),
        example_dir + "/src/common/coco_label.txt"}) {
    if (labels_file.empty() && fs::exists(candidate)) {
      labels_file = candidate;
    }
  }
  if (labels_file.empty()) {
    return skip_or_fail("src/common/coco_label.txt not found; set "
                        "SIMANEAT_APPS_TEST_LABELS_FILE");
  }

  const auto out_dir = create_test_output_dir(kExample, "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 19200);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 19300);
  VideoReceiver video(fs::path(out_dir) / "received.h264", video_port);
  MetadataJsonListenerOptions options;
  options.base_port = metadata_port;
  options.num_ports = 1;
  options.min_object_count = 1;
  options.timeout_ms = 5000;
  MetadataJsonListener listener(options);
  if (!listener.ok()) throw std::runtime_error(listener.error());

  // write_e2e_config folds `testing.e2e.*` into the runtime keys, so the
  // fixed-image source override in config.yaml is applied here automatically.
  const fs::path config_path = fs::path(out_dir).parent_path() / "config.yaml";
  write_e2e_config(kExample, config_path,
                   {{"model.path", model_path},
                    {"model.labels", labels_file},
                    {"inference.frames", std::to_string(kFrames)},
                    {"output.insight.host", "127.0.0.1"},
                    {"output.insight.video_port", std::to_string(video_port)},
                    {"output.insight.metadata_port", std::to_string(metadata_port)}});

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else if (result.stdout_text.find("source=override") == std::string::npos) {
    std::cerr << "[FAIL] run did not report the overridden source\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    rc = 1;
  } else if (result.stdout_text.find("processed=" + std::to_string(kFrames)) == std::string::npos) {
    std::cerr << "[FAIL] run did not publish " << kFrames << " frames\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    rc = 1;
  } else {
    video.verify();
    const auto received = listener.wait_for_messages();
    const auto following = listener.wait_for_messages();
    verify_metadata(received);
    verify_metadata(following);
    if (following.messages.back().timestamp_ms <= received.messages.back().timestamp_ms)
      throw std::runtime_error("metadata timestamps did not advance");
    std::cout << "[OK] received decodable H.264 and nonempty valid detection metadata\n";
    std::cout << "[OK] published " << kFrames << " frames from the fixed-image source\n";
  }

  remove_dir(out_dir);
  return rc;
}
 catch (const std::exception& error) {
  std::cerr << "[FAIL] " << error.what() << "\n";
  return 1;
}
