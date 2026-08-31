#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

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

bool valid_objects(const MetadataJsonListenerResult& result) {
  for (const auto& message : result.messages) {
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& object : payload.at("data").at("objects")) {
      if (!object.contains("label") || object.at("label").get<std::string>().empty() ||
          !object.contains("confidence") || !object.at("confidence").is_number() ||
          !object.contains("bbox") || !object.at("bbox").is_array() ||
          object.at("bbox").size() != 4U) {
        return false;
      }
      for (const auto& value : object.at("bbox")) {
        if (!value.is_number() || value.get<double>() < 0.0) {
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
  const std::vector<Variant> variants = {
      {"small", "rfdetr-small-backbone.tar.gz", "rfdetr-small-transformer.tar.gz"},
      {"medium", "rfdetr-medium-backbone.tar.gz", "rfdetr-medium-transformer.tar.gz"},
  };
  const std::vector<SourceCase> sources = {
      {"h264", "SIMANEAT_TEST_RTSP_H264_URL"},
      {"h265", "SIMANEAT_TEST_RTSP_H265_URL"},
      {"mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL"},
  };
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const fs::path labels =
      example_common_config_path("rfdetr-object-detector").parent_path() / "coco-labels.txt";

  for (const auto& source : sources) {
    const char* rtsp_url = env_or_null(source.environment);
    if (rtsp_url == nullptr) {
      return skip_or_fail(std::string(source.environment) + " is required for RF-DETR e2e");
    }
    for (const auto& variant : variants) {
      const fs::path backbone = models_dir / variant.backbone;
      const fs::path transformer = models_dir / variant.transformer;
      if (!fs::exists(backbone) || !fs::exists(transformer)) {
        return skip_or_fail("missing RF-DETR " + variant.name + " artifacts");
      }
      const std::string output_dir = create_test_output_dir(
          "rfdetr-object-detector", "test_" + variant.name + "_" + source.codec);
      if (output_dir.empty()) {
        return 1;
      }
      const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
      write_e2e_config("rfdetr-object-detector", config_path,
                       {{"model.variant", variant.name},
                        {"model.labels", labels.string()},
                        {"model." + variant.name + ".backbone", backbone.string()},
                        {"model." + variant.name + ".transformer", transformer.string()},
                        {"source.rtsp_url", rtsp_url},
                        {"source.codec", source.codec},
                        {"inference.frames", "20"},
                        {"inference.min_score", "0.2"},
                        {"output.insight.host", "127.0.0.1"},
                        {"output.insight.video_port", std::to_string(video_port)},
                        {"output.insight.metadata_port", std::to_string(metadata_port)}});

      MetadataJsonListenerOptions options;
      options.host = "127.0.0.1";
      options.base_port = metadata_port;
      options.num_ports = 1;
      options.timeout_ms = 5000;
      options.metadata_type = "object-detection";
      options.data_array_key = "objects";
      options.require_all_ports = true;
      options.min_object_count = 1;
      MetadataJsonListener listener(options);
      if (!listener.ok()) {
        std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
        return 1;
      }

      const ProcessResult process =
          spawn_and_wait(argv[1], {"--config", config_path.string()}, timeout_ms);
      if (process.exit_code != 0) {
        std::cerr << "[FAIL] RF-DETR " << variant.name << " " << source.codec << " exited with "
                  << process.exit_code << "\nstdout:\n"
                  << process.stdout_text << "\nstderr:\n"
                  << process.stderr_text << "\n";
        return 1;
      }
      const auto metadata = listener.wait_for_messages();
      if (!metadata.success || !valid_objects(metadata)) {
        std::cerr << "[FAIL] invalid RF-DETR " << variant.name << " " << source.codec
                  << " metadata: " << metadata.error << "\n";
        return 1;
      }
      remove_dir(output_dir);
    }
  }
  std::cout << "[OK] RF-DETR Small and Medium published valid H.264, H.265, and MJPEG metadata\n";
  return 0;
}
