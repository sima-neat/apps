// E2E test for single-stream-instance-segmenter.
// Runs every supported model family over every supported input path and checks the three
// outputs the application advertises: annotated frames, Insight segmentation metadata,
// and Insight video.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <regex>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

const char* kExample = "single-stream-instance-segmenter";

struct FamilyCase {
  const char* family;
  const char* model_file;
};

struct SourceCase {
  const char* type;
  const char* codec;
  const char* environment;
};

// Model families and the packages tests/test-scope.yaml downloads for them.
const std::vector<FamilyCase> kFamilies = {
    {"yolo26", "yolo26m-seg-bf16-b1.tar.gz"},
    {"yolov8", "yolo_v8n_seg_mpk.tar.gz"},
};

// Input paths the application supports, and the environment variable holding each URL.
const std::vector<SourceCase> kSources = {
    {"rtsp", "h264", "SIMANEAT_TEST_RTSP_H264_URL"},
    {"rtsp", "mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL"},
    {"http", "mjpeg", "SIMANEAT_TEST_HTTP_MJPEG_URL"},
};

bool valid_metadata(const MetadataJsonListenerResult& result) {
  for (const auto& message : result.messages) {
    if (message.frame_id.empty() ||
        message.frame_id.find_first_not_of("0123456789") != std::string::npos ||
        message.timestamp_ms < 0) {
      return false;
    }
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& segment : payload.at("data").at("segments")) {
      if (!segment.contains("label") || segment.at("label").get<std::string>().empty() ||
          !segment.contains("confidence") || !segment.at("confidence").is_number() ||
          segment.at("confidence").get<double>() < 0.0 ||
          segment.at("confidence").get<double>() > 1.0 || !segment.contains("bbox") ||
          !segment.at("bbox").is_array() || segment.at("bbox").size() != 4U ||
          !segment.contains("mask_format") || segment.at("mask_format") != "polygon" ||
          !segment.contains("mask") || !segment.at("mask").is_array() ||
          segment.at("mask").size() < 3U) {
        return false;
      }
      for (const auto& value : segment.at("bbox")) {
        if (!value.is_number() || value.get<double>() < 0.0) {
          return false;
        }
      }
      for (const auto& point : segment.at("mask")) {
        if (!point.is_array() || point.size() != 2U || !point[0].is_number() ||
            !point[1].is_number() || point[0].get<double>() < 0.0 ||
            point[1].get<double>() < 0.0) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Every result due a picture is accounted for: written, or reported as having lost the decoded
/// frame it needed. A source faster than the model legitimately produces some of the latter on
/// the host-decoded route, so the run is checked for complete accounting rather than a fixed
/// yield the platform cannot promise.
bool accounted_saves(const std::string& summary, int attempts, int files) {
  std::smatch match;
  const std::regex pattern(R"(saved=(\d+) unpaired=(\d+))");
  if (!std::regex_search(summary, match, pattern)) {
    return false;
  }
  const int saved = std::stoi(match[1].str());
  const int unpaired = std::stoi(match[2].str());
  return saved + unpaired == attempts && saved > 0 && files == saved;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path models_dir = models_dir_raw != nullptr ? models_dir_raw : "models";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const int frames = e2e_int(kExample, "testing.e2e.inference", "frames");
  const int save_every = e2e_int(kExample, "testing.e2e.output", "save_every");
  const int attempts = save_every > 0 ? frames / save_every : 0;

  for (const auto& source : kSources) {
    const char* source_url = env_or_null(source.environment);
    if (source_url == nullptr) {
      return skip_or_fail(std::string(source.environment) + " is required for " + kExample +
                          " e2e");
    }
    for (const auto& model : kFamilies) {
      const fs::path model_path = models_dir / model.model_file;
      if (!fs::exists(model_path)) {
        return skip_or_fail(std::string("missing ") + model.family + " package: " +
                            model_path.string());
      }

      const std::string case_name =
          std::string(model.family) + "_" + source.type + "_" + source.codec;
      const std::string output_dir = create_test_output_dir(kExample, "test_" + case_name);
      if (output_dir.empty()) {
        return 1;
      }
      const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
      write_e2e_config(kExample, config_path,
                       {{"model.family", model.family},
                        {"model.path", model_path.string()},
                        {"source.type", source.type},
                        {"source.codec", source.codec},
                        {"source.url", source_url},
                        // HTTPS MJPEG sources commonly carry a self-signed certificate.
                        {"source.ssl_strict",
                         std::string(source.type) == "http" ? "false" : "true"},
                        {"output.save_dir", output_dir},
                        // This test is the Insight receiver, so it publishes to loopback.
                        {"output.insight.host", "127.0.0.1"},
                        {"output.insight.video_port", std::to_string(video_port)},
                        {"output.insight.metadata_port", std::to_string(metadata_port)}});

      MetadataJsonListenerOptions listener_options;
      listener_options.host = "127.0.0.1";
      listener_options.base_port = metadata_port;
      listener_options.num_ports = 1;
      listener_options.timeout_ms = 5000;
      listener_options.metadata_type = "segmentation";
      listener_options.data_array_key = "segments";
      listener_options.require_all_ports = true;
      listener_options.min_object_count = 1;
      MetadataJsonListener listener(listener_options);
      if (!listener.ok()) {
        std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
        remove_dir(output_dir);
        return 1;
      }

      const int video_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
      sockaddr_in video_address{};
      video_address.sin_family = AF_INET;
      video_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      video_address.sin_port = htons(static_cast<uint16_t>(video_port));
      if (video_socket < 0 || ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address),
                                     sizeof(video_address)) != 0) {
        if (video_socket >= 0) {
          ::close(video_socket);
        }
        std::cerr << "[FAIL] could not bind Insight video port\n";
        remove_dir(output_dir);
        return 1;
      }

      const ProcessResult process =
          spawn_and_wait(argv[1], {"--config", config_path.string()}, timeout_ms);
      std::array<unsigned char, 65536> video_packet{};
      const auto video_bytes =
          ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
      ::close(video_socket);

      int rc = 0;
      if (process.exit_code != 0) {
        std::cerr << "[FAIL] " << case_name << " exited with " << process.exit_code
                  << "\nstdout:\n"
                  << process.stdout_text << "\nstderr:\n"
                  << process.stderr_text << "\n";
        rc = 1;
      } else if (process.stdout_text.find("model=" + std::string(model.family)) ==
                     std::string::npos ||
                 process.stdout_text.find("processed=" + std::to_string(frames)) ==
                     std::string::npos) {
        std::cerr << "[FAIL] " << case_name << " did not run the configured family to completion\n"
                  << process.stdout_text;
        rc = 1;
      } else if (!accounted_saves(process.stdout_text, attempts, count_output_files(output_dir))) {
        std::cerr << "[FAIL] " << case_name << " did not account for every annotated frame\n"
                  << process.stdout_text;
        rc = 1;
      } else if (!all_output_files_nonempty(output_dir)) {
        std::cerr << "[FAIL] " << case_name << " wrote empty annotated frames\n";
        rc = 1;
      } else if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 ||
                 (video_packet[1] & 0x7F) != 96) {
        // VideoSender always re-encodes to RTP H.264 (payload type 96) for Insight.
        std::cerr << "[FAIL] " << case_name << " did not publish RTP H.264 video to Insight\n";
        rc = 1;
      } else {
        const auto metadata = listener.wait_for_messages();
        if (!metadata.success || !valid_metadata(metadata)) {
          std::cerr << "[FAIL] " << case_name << " metadata invalid: " << metadata.error << "\n";
          rc = 1;
        } else {
          std::cout << "[OK] " << case_name << " published " << count_output_files(output_dir)
                    << " annotated frames, Insight video, and segmentation metadata\n";
        }
      }

      remove_dir(output_dir);
      if (rc != 0) {
        return rc;
      }
    }
  }
  return 0;
}
