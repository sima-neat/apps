// E2E test for rfdetr-stream-instance-segmenter.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const char* rtsp_url = env_or_null("SIMANEAT_TEST_RTSP_H264_URL");
  if (!rtsp_url) {
    return skip_or_fail(
        "SIMANEAT_TEST_RTSP_H264_URL is required for rfdetr-stream-instance-segmenter e2e");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  const std::string model_path =
      configured_model_path("rfdetr-stream-instance-segmenter", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured RF-DETR-Seg model archive not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir =
      create_test_output_dir("rfdetr-stream-instance-segmenter", "test_full_pipeline_rtsp_h264");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const std::string insight_host = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       ? env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       : "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int total_saved_frames = e2e_int("rfdetr-stream-instance-segmenter", "testing.e2e.output",
                                        "total_saved_frames");
  write_e2e_config("rfdetr-stream-instance-segmenter", config_path,
                   {{"source.url", rtsp_url},
                    {"model.path", model_path},
                    {"output.save_dir", output_dir},
                    {"output.insight.host", insight_host},
                    {"output.insight.video_port", std::to_string(video_port)},
                    {"output.insight.metadata_port", std::to_string(metadata_port)}});

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(argv[1], {"--config", config_path.string()},
                                                        output_dir, total_saved_frames, timeout_ms);

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
      std::cout << "[OK] rfdetr stream instance segmenter produced " << files
                << " sampled output files\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
