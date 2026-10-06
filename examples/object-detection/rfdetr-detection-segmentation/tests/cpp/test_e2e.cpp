#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

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

struct Variant {
  std::string name;
  std::string backbone;
  std::string transformer;
};

struct SourceCase {
  std::string codec;
  const char* environment;
};

struct TestCase {
  std::string task;
  Variant model;
  SourceCase source;
};

bool valid_metadata(const MetadataJsonListenerResult& result, bool segmentation) {
  const char* data_key = segmentation ? "segments" : "objects";
  for (const auto& message : result.messages) {
    if (message.frame_id.empty() ||
        message.frame_id.find_first_not_of("0123456789") != std::string::npos ||
        message.timestamp_ms < 0) {
      return false;
    }
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& entry : payload.at("data").at(data_key)) {
      if (!entry.contains("label") || entry.at("label").get<std::string>().empty() ||
          !entry.contains("confidence") || !entry.at("confidence").is_number() ||
          entry.at("confidence").get<double>() < 0.0 ||
          entry.at("confidence").get<double>() > 1.0 ||
          !entry.contains("bbox") || !entry.at("bbox").is_array() ||
          entry.at("bbox").size() != 4U) {
        return false;
      }
      for (const auto& value : entry.at("bbox")) {
        if (!value.is_number() || value.get<double>() < 0.0) {
          return false;
        }
      }
      if (!segmentation) {
        continue;
      }
      if (!entry.contains("mask_format") || entry.at("mask_format") != "polygon" ||
          !entry.contains("mask") || !entry.at("mask").is_array() || entry.at("mask").size() < 3U) {
        return false;
      }
      for (const auto& point : entry.at("mask")) {
        if (!point.is_array() || point.size() != 2U || !point[0].is_number() ||
            !point[1].is_number() || point[0].get<double>() < 0.0 || point[1].get<double>() < 0.0) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const char* models_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const fs::path models_dir = models_raw != nullptr ? models_raw : "models";
  const std::vector<Variant> detection_models = {
      {"small", "rfdetr-small-backbone.tar.gz", "rfdetr-small-transformer.tar.gz"},
      {"medium", "rfdetr-medium-backbone.tar.gz", "rfdetr-medium-transformer.tar.gz"},
  };
  const Variant segmentation_model = {"medium", "rfdetr-seg-medium-backbone.tar.gz",
                                      "rfdetr-seg-medium-transformer.tar.gz"};
  const std::vector<SourceCase> sources = {
      {"h264", "SIMANEAT_TEST_RTSP_H264_URL"},
      {"h265", "SIMANEAT_TEST_RTSP_H265_URL"},
      {"mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL"},
  };
  std::vector<TestCase> cases;
  for (const auto& source : sources) {
    for (const auto& model : detection_models) {
      cases.push_back({"detection", model, source});
    }
    cases.push_back({"segmentation", segmentation_model, source});
  }
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path labels =
      example_common_config_path("rfdetr-detection-segmentation").parent_path() / "coco-labels.txt";

  for (const auto& test : cases) {
    const char* rtsp_url = env_or_null(test.source.environment);
    if (rtsp_url == nullptr) {
      return skip_or_fail(std::string(test.source.environment) + " is required for RF-DETR e2e");
    }
    const fs::path backbone = models_dir / test.model.backbone;
    const fs::path transformer = models_dir / test.model.transformer;
    if (!fs::exists(backbone) || !fs::exists(transformer)) {
      return skip_or_fail("missing RF-DETR " + test.task + " artifacts");
    }
    const std::string case_name = test.task + "_" + test.model.name + "_" + test.source.codec;
    const std::string output_dir =
        create_test_output_dir("rfdetr-detection-segmentation", "test_" + case_name);
    if (output_dir.empty()) {
      return 1;
    }
    const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
    ConfigScalars overrides = {
        {"model.task", test.task},
        {"model.labels", labels.string()},
        {"source.rtsp_url", rtsp_url},
        {"source.codec", test.source.codec},
        {"inference.frames", "20"},
        {"inference." + test.task + ".min_score", "0.2"},
        {"output.insight.host", "127.0.0.1"},
        {"output.insight.video_port", std::to_string(video_port)},
        {"output.insight.metadata_port", std::to_string(metadata_port)},
    };
    if (test.task == "detection") {
      overrides["model.detection.variant"] = test.model.name;
      overrides["model.detection." + test.model.name + ".backbone"] = backbone.string();
      overrides["model.detection." + test.model.name + ".transformer"] = transformer.string();
    } else {
      overrides["model.segmentation.backbone"] = backbone.string();
      overrides["model.segmentation.transformer"] = transformer.string();
    }
    write_e2e_config("rfdetr-detection-segmentation", config_path, overrides);

    const bool segmentation = test.task == "segmentation";
    MetadataJsonListenerOptions options;
    options.host = "127.0.0.1";
    options.base_port = metadata_port;
    options.num_ports = 1;
    options.timeout_ms = 5000;
    options.metadata_type = segmentation ? "segmentation" : "object-detection";
    options.data_array_key = segmentation ? "segments" : "objects";
    options.require_all_ports = true;
    options.min_object_count = 1;
    MetadataJsonListener listener(options);
    if (!listener.ok()) {
      std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
      return 1;
    }

    const int video_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in video_address{};
    video_address.sin_family = AF_INET;
    video_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    video_address.sin_port = htons(video_port);
    if (video_socket < 0 ||
        ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address), sizeof(video_address)) != 0) {
      if (video_socket >= 0) {
        ::close(video_socket);
      }
      std::cerr << "[FAIL] could not bind Insight video port\n";
      return 1;
    }
    const ProcessResult process =
        spawn_and_wait(argv[1], {"--config", config_path.string()}, timeout_ms);
    std::array<unsigned char, 65536> video_packet{};
    const auto video_bytes =
        ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
    ::close(video_socket);
    if (process.exit_code != 0) {
      std::cerr << "[FAIL] RF-DETR " << case_name << " exited with " << process.exit_code
                << "\nstdout:\n"
                << process.stdout_text << "\nstderr:\n"
                << process.stderr_text << "\n";
      return 1;
    }
    const auto metadata = listener.wait_for_messages();
    const auto following = listener.wait_for_messages();
    if (!metadata.success || !following.success || !valid_metadata(metadata, segmentation) ||
        !valid_metadata(following, segmentation)) {
      std::cerr << "[FAIL] invalid RF-DETR " << case_name << " metadata: " << metadata.error
                << "\n";
      return 1;
    }
    const auto& first = metadata.messages.back();
    const auto& last = following.messages.back();
    if (std::stoll(last.frame_id) <= std::stoll(first.frame_id) ||
        last.timestamp_ms <= first.timestamp_ms ||
        process.stdout_text.find("RF-DETR " + test.task + ": completed=20 ") == std::string::npos) {
      std::cerr << "[FAIL] RF-DETR " << case_name
                << " did not complete with progressing frame identities\n";
      return 1;
    }
    // VideoSender uses RTP H.265 (98) or H.264 (96). MJPEG is re-encoded as H.264.
    const int payload_type = test.source.codec == "h265" ? 98 : 96;
    if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 ||
        (video_packet[1] & 0x7f) != payload_type) {
      std::cerr << "[FAIL] RF-DETR " << case_name << " did not publish the expected RTP video\n";
      return 1;
    }
    remove_dir(output_dir);
  }
  std::cout << "[OK] RF-DETR published video and valid detection and segmentation metadata\n";
  return 0;
}
