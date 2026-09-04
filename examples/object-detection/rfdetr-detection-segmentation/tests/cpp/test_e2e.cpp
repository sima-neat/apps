#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr int kPerformanceFrames = 300;

struct Variant {
  std::string name;
  std::string backbone;
  std::string transformer;
  double minimum_fps;
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
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& entry : payload.at("data").at(data_key)) {
      if (!entry.contains("label") || entry.at("label").get<std::string>().empty() ||
          !entry.contains("confidence") || !entry.at("confidence").is_number() ||
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

double output_fps(const std::string& output) {
  std::smatch match;
  if (!std::regex_search(output, match, std::regex(R"(output_fps=([0-9]+(?:\.[0-9]+)?))"))) {
    return -1.0;
  }
  return std::stod(match[1].str());
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
      {"small", "rfdetr-small-backbone.tar.gz", "rfdetr-small-transformer.tar.gz", 65.0},
      {"medium", "rfdetr-medium-backbone.tar.gz", "rfdetr-medium-transformer.tar.gz", 45.0},
  };
  const Variant segmentation_model = {"medium", "rfdetr-seg-medium-backbone.tar.gz",
                                      "rfdetr-seg-medium-transformer.tar.gz", 35.0};
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
        {"inference.frames", test.source.codec == "h265" || test.source.codec == "mjpeg"
                                 ? std::to_string(kPerformanceFrames)
                                 : "20"},
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

    const ProcessResult process =
        spawn_and_wait(argv[1], {"--config", config_path.string()}, timeout_ms);
    if (process.exit_code != 0) {
      std::cerr << "[FAIL] RF-DETR " << case_name << " exited with " << process.exit_code
                << "\nstdout:\n"
                << process.stdout_text << "\nstderr:\n"
                << process.stderr_text << "\n";
      return 1;
    }
    const auto metadata = listener.wait_for_messages();
    if (!metadata.success || !valid_metadata(metadata, segmentation)) {
      std::cerr << "[FAIL] invalid RF-DETR " << case_name << " metadata: " << metadata.error
                << "\n";
      return 1;
    }
    if (test.source.codec == "h265" || test.source.codec == "mjpeg") {
      const double measured_fps = output_fps(process.stdout_text);
      std::cout << "[perf] task=" << test.task << " model=" << test.model.name
                << " codec=" << test.source.codec << " frames=" << kPerformanceFrames
                << " output_fps=" << measured_fps << " minimum=" << test.model.minimum_fps << "\n";
      if (measured_fps < test.model.minimum_fps) {
        std::cerr << "[FAIL] RF-DETR " << case_name << " output FPS was below "
                  << test.model.minimum_fps << "\nstdout:\n"
                  << process.stdout_text << "\n";
        return 1;
      }
    }
    remove_dir(output_dir);
  }
  std::cout << "[OK] RF-DETR published valid detection and segmentation metadata at the expected "
               "throughput\n";
  return 0;
}
