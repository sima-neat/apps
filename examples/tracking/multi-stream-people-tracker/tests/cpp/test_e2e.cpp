// E2E test for multi-stream-people-tracker.
// Runs the RTSP pipeline and verifies sampled debug frames are written.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];

  const std::vector<std::string> rtsp_urls = rtsp_urls_from_env();
  if (rtsp_urls.size() < 2) {
    return skip_or_fail("need at least two RTSP URLs for multistream e2e");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";
  const std::string model_path = configured_model_path("multi-stream-people-tracker", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured detector model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir = create_test_output_dir(
      "multi-stream-people-tracker", "test_multi_stream_insight_and_save_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const std::string insight_host = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       ? env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       : "127.0.0.1";
  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int total_saved_frames =
      e2e_int("multi-stream-people-tracker", "testing.e2e.output", "total_saved_frames");
  write_e2e_config("multi-stream-people-tracker", config_path,
                   {{"model.path", model_path},
                    {"output.debug_dir", output_dir},
                    {"output.insight.host", insight_host},
                    {"output.insight.video_port_base", std::to_string(video_port_base)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port_base)},
                    {"inference.frames", "140"}},
                   {{"streams", {rtsp_urls[0], rtsp_urls[1]}}});

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = insight_host;
  metadata_options.base_port = metadata_port_base;
  metadata_options.num_ports = 2;
  metadata_options.timeout_ms = 5000;
  metadata_options.require_all_ports = true;
  metadata_options.metadata_type = "tracking";
  metadata_options.data_array_key = "tracks";
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << metadata_listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const ProcessResult result =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);

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
      std::cout << "[OK] multi-camera people tracker produced " << files
                << " sampled output files\n";
    }
  }
  if (rc == 0) {
    const MetadataJsonListenerResult metadata = metadata_listener.wait_for_messages();
    if (!metadata.success) {
      std::cerr << "[FAIL] tracking metadata was not received on all streams: " << metadata.error
                << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] tracking metadata received on " << metadata.ports_with_valid_json.size()
                << " streams\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
