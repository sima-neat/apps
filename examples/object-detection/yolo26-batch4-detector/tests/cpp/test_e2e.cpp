// E2E test for yolo26-batch4-detector.
// Runs the four-stream batched pipeline and verifies that sampled debug frames
// are written and that per-stream detection metadata reaches Insight.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {
constexpr const char* kExample = "yolo26-batch4-detector";
constexpr const char* kE2eInsightHost = "127.0.0.1";
// The model is compiled for batch 4, so the test needs all four lanes fed.
constexpr std::size_t kRequiredStreams = 4;
} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];

  const std::vector<std::string> rtsp_urls = rtsp_h264_urls_from_env();
  if (rtsp_urls.size() < kRequiredStreams) {
    return skip_or_fail("need four RTSP URLs to fill the batch-4 lanes");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";
  const std::string model_path = configured_model_path(kExample, models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured batch-4 model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir =
      create_test_output_dir(kExample, "test_batch4_insight_and_save_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path labels_file =
      example_common_config_path(kExample).parent_path() / "coco_label.txt";
  if (!fs::exists(labels_file)) {
    return skip_or_fail("src/common/coco_label.txt not found for yolo26-batch4-detector");
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const int save_every = e2e_int(kExample, "testing.e2e.output", "save_every");
  const int total_saved_frames = e2e_int(kExample, "testing.e2e.output", "total_saved_frames");
  write_e2e_config(kExample, config_path,
                   {{"model.path", model_path},
                    {"model.labels", labels_file.string()},
                    {"output.debug_dir", output_dir},
                    {"output.save_every", std::to_string(save_every)},
                    {"output.insight.host", kE2eInsightHost},
                    {"output.insight.video_port_base", std::to_string(video_port_base)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port_base)},
                    {"inference.frames", "140"}},
                   {{"streams", {rtsp_urls[0], rtsp_urls[1], rtsp_urls[2], rtsp_urls[3]}}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port_base;
  metadata_options.num_ports = static_cast<int>(kRequiredStreams);
  metadata_options.timeout_ms = std::min(timeout_ms, 30000);
  metadata_options.require_all_ports = true;
  metadata_options.min_object_count = 1;
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << metadata_listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  // Drain UDP while the application runs. Waiting until process exit can
  // overflow receive buffers when four lanes publish at once.
  auto metadata_future = std::async(
      std::launch::async, [&metadata_listener] { return metadata_listener.wait_for_messages(); });
  const ProcessResult result = spawn_until_output_files(binary, {"--config", config_path.string()},
                                                        output_dir, total_saved_frames, timeout_ms);
  const MetadataJsonListenerResult metadata = metadata_future.get();

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const int files = count_output_files(output_dir);
    if (files < total_saved_frames) {
      std::cerr << "[FAIL] expected at least " << total_saved_frames
                << " sampled output files, got " << files << "\n";
      rc = 1;
    } else if (!all_output_files_nonempty(output_dir)) {
      std::cerr << "[FAIL] some sampled output files are empty\n";
      rc = 1;
    } else {
      for (std::size_t lane = 0; lane < kRequiredStreams; ++lane) {
        const std::string prefix = "stream_" + std::to_string(lane) + "_frame_";
        const bool lane_present =
            std::any_of(fs::directory_iterator(output_dir), fs::directory_iterator{},
                        [&](const fs::directory_entry& entry) {
                          return entry.is_regular_file() && entry.file_size() > 0U &&
                                 entry.path().filename().string().starts_with(prefix);
                        });
        if (!lane_present) {
          std::cerr << "[FAIL] no non-empty debug JPEG for lane " << lane << "\n";
          rc = 1;
        }
      }
      if (rc == 0) {
        std::cout << "[OK] batch-4 detector produced " << files << " sampled output files\n";
      }
    }
  }
  if (rc == 0) {
    if (!metadata.success) {
      std::cerr << "[FAIL] object-detection metadata was not received on all lanes: "
                << metadata.error << "\n";
      rc = 1;
    } else {
      for (const auto& message : metadata.messages) {
        const auto payload = nlohmann::json::parse(message.payload);
        if (!payload.contains("_insight") || !payload["_insight"].contains("rtp_timestamp") ||
            !payload["_insight"]["rtp_timestamp"].is_number_unsigned() ||
            payload["_insight"]["rtp_timestamp"].get<std::uint64_t>() > 0xffffffffULL) {
          std::cerr << "[FAIL] metadata is missing _insight.rtp_timestamp\n";
          rc = 1;
          break;
        }
        for (const auto& object : payload["data"]["objects"]) {
          if (!object.contains("bbox") || !object["bbox"].is_array() ||
              object["bbox"].size() != 4U) {
            std::cerr << "[FAIL] detection metadata has an invalid bbox\n";
            rc = 1;
            break;
          }
          const double x = object["bbox"][0].get<double>();
          const double y = object["bbox"][1].get<double>();
          const double width = object["bbox"][2].get<double>();
          const double height = object["bbox"][3].get<double>();
          const double confidence = object.value("confidence", -1.0);
          if (object.value("label", std::string{}).empty() || !std::isfinite(confidence) ||
              confidence < 0.35 || confidence > 1.0 || !std::isfinite(x) || !std::isfinite(y) ||
              !std::isfinite(width) || !std::isfinite(height) || x < 0.0 || y < 0.0 ||
              width <= 0.0 || height <= 0.0) {
            std::cerr << "[FAIL] detection metadata bbox is not finite and positive\n";
            rc = 1;
            break;
          }
        }
      }
    }
    if (rc == 0) {
      std::cout << "[OK] object-detection metadata received on "
                << metadata.ports_with_valid_json.size() << " lanes\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
