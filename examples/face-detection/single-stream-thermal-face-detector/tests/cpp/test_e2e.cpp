// E2E test for single-stream-thermal-face-detector.
// Runs the single-stream RTSP pipeline and verifies the app publishes valid
// pose-estimation metadata JSON (the 5 facial landmarks) to the Insight metadata
// UDP port. A local UDP listener stands in for Insight, so no running viewer is
// required -- only an RTSP H.264 source (SIMANEAT_TEST_RTSP_H264_URL) and the model.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {
constexpr const char* kE2eInsightHost = "127.0.0.1";
} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* rtsp_url = env_or_null("SIMANEAT_TEST_RTSP_H264_URL");
  if (!rtsp_url) {
    return skip_or_fail(
        "SIMANEAT_TEST_RTSP_H264_URL is required for single-stream-thermal-face-detector e2e");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  const std::string model_path = configured_model_path("single-stream-thermal-face-detector", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured yolov5s-face model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir = create_test_output_dir("single-stream-thermal-face-detector", "test_metadata_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  write_e2e_config("single-stream-thermal-face-detector", config_path,
                   {{"model.path", model_path},
                    {"source.rtsp_url", rtsp_url},
                    {"output.insight.host", kE2eInsightHost},
                    {"output.insight.video_port", std::to_string(video_port)},
                    {"output.insight.metadata_port", std::to_string(metadata_port)},
                    {"inference.frames", "140"}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = 1;
  metadata_options.timeout_ms = 5000;
  metadata_options.metadata_type = "pose-estimation";
  metadata_options.data_array_key = "poses";
  metadata_options.require_all_ports = true;
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << metadata_listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const MetadataJsonListenerResult metadata = metadata_listener.wait_for_messages();
    if (!metadata.success) {
      std::cerr << "[FAIL] pose-estimation metadata was not received: " << metadata.error << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] yolov5s-face published pose-estimation metadata on "
                << metadata.ports_with_valid_json.size() << " port(s)\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
