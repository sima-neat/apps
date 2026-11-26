#include "sima_examples/labels.h"
#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

struct Variant {
  std::string id;
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

// Every result names a label from the label file and passes inference.min_score (0.2), and
// every segment carries a polygon.
bool valid_metadata(const MetadataJsonListenerResult& result, bool segmentation,
                    const std::set<std::string>& labels) {
  const char* data_key = segmentation ? "segments" : "objects";
  for (const auto& message : result.messages) {
    if (message.frame_id.empty() ||
        message.frame_id.find_first_not_of("0123456789") != std::string::npos ||
        message.timestamp_ms < 0) {
      std::cerr << "[FAIL] bad frame identity: " << message.frame_id << "\n";
      return false;
    }
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& entry : payload.at("data").at(data_key)) {
      const double confidence = entry.at("confidence").get<double>();
      if (labels.count(entry.at("label").get<std::string>()) != 1 || confidence < 0.2 ||
          confidence > 1.0 || entry.at("bbox").size() != 4U) {
        std::cerr << "[FAIL] result outside the label file or below min_score: " << entry.dump()
                  << "\n";
        return false;
      }
      if (segmentation && (entry.at("mask_format") != "polygon" || entry.at("mask").size() < 3U)) {
        std::cerr << "[FAIL] segment without a polygon: " << entry.dump() << "\n";
        return false;
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
      {"det-nano", "rf_detr_nano_w8a8_mixed_w16a16_backbone_mpk.tar.gz",
       "rf_detr_nano_w8a8_mixed_w16a16_decoder_mpk.tar.gz"},
      {"det-small", "rf_detr_small_w8a8_mixed_w16a16_backbone_mpk.tar.gz",
       "rf_detr_small_w8a8_mixed_w16a16_decoder_mpk.tar.gz"},
      {"det-medium", "rf_detr_medium_w8a8_mixed_w16a16_backbone_mpk.tar.gz",
       "rf_detr_medium_w8a8_mixed_w16a16_decoder_mpk.tar.gz"},
      {"det-large", "rf_detr_large_w8a8_mixed_w16a16_backbone_mpk.tar.gz",
       "rf_detr_large_w8a8_mixed_w16a16_decoder_mpk.tar.gz"},
  };
  // Each segmentation size uses the registry build profile config.yaml ships with.
  const std::vector<Variant> segmentation_models = {
      {"seg-nano", "rf_detr_seg_nano_w8a16_w8a8_mixed_backbone_mpk.tar.gz",
       "rf_detr_seg_nano_w8a16_w8a8_mixed_decoder_mpk.tar.gz"},
      {"seg-small", "rf_detr_seg_small_w8a16_w8a8_mixed_backbone_mpk.tar.gz",
       "rf_detr_seg_small_w8a16_w8a8_mixed_decoder_mpk.tar.gz"},
      {"seg-medium", "rf_detr_seg_medium_w8a16_w8a8_mixed_backbone_mpk.tar.gz",
       "rf_detr_seg_medium_w8a16_w8a8_mixed_decoder_mpk.tar.gz"},
  };
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
    for (const auto& model : segmentation_models) {
      cases.push_back({"segmentation", model, source});
    }
  }
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const char* apps_root = env_or_null("APPS_ROOT");
  const fs::path labels =
      fs::path(apps_root != nullptr ? apps_root : ".") / "support" / "labels" / "coco91.txt";
  const auto label_list = sima_examples::load_labels(labels);
  const std::set<std::string> label_set(label_list.begin(), label_list.end());

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
    const std::string case_name = test.model.id + "_" + test.source.codec;
    const std::string output_dir =
        create_test_output_dir("rfdetr-detection-segmentation", "test_" + case_name);
    if (output_dir.empty()) {
      return 1;
    }
    const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
    const bool segmentation = test.task == "segmentation";
    const ConfigScalars overrides = {
        {"model.id", test.model.id},
        {"model.path", backbone.string()},
        {"model.labels", labels.string()},
        {"input.source", rtsp_url},
        {"input.codec", test.source.codec},
        {"inference.frames", "20"},
        {"inference.min_score", "0.2"},
        // The config comments recommend 100 results per frame for detection, 24 for
        // segmentation.
        {"inference.max_detections", segmentation ? "24" : "100"},
        {"output.insight.host", "127.0.0.1"},
        {"output.insight.video_port_base", std::to_string(video_port)},
        {"output.insight.metadata_port_base", std::to_string(metadata_port)},
    };
    write_e2e_config("rfdetr-detection-segmentation", config_path, overrides);

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
    if (!metadata.success || !following.success ||
        !valid_metadata(metadata, segmentation, label_set) ||
        !valid_metadata(following, segmentation, label_set)) {
      std::cerr << "[FAIL] invalid RF-DETR " << case_name << " metadata: " << metadata.error
                << "\n";
      return 1;
    }
    const auto& first = metadata.messages.back();
    const auto& last = following.messages.back();
    if (std::stoll(last.frame_id) <= std::stoll(first.frame_id) ||
        last.timestamp_ms <= first.timestamp_ms ||
        process.stdout_text.find("RF-DETR " + test.model.id + ": sent 20 results to Insight") ==
            std::string::npos) {
      std::cerr << "[FAIL] RF-DETR " << case_name
                << " did not complete with progressing frame identities\n";
      return 1;
    }
    // VideoSender forwards the stream as received: RTP H.264 (96), H.265 (98), or JPEG (26).
    const int payload_type = test.source.codec == "h265"    ? 98
                             : test.source.codec == "mjpeg" ? 26
                                                            : 96;
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
