// E2E test for single-stream-instance-segmenter.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const char* rtsp_url = env_or_null("SIMANEAT_APPS_TEST_RTSP_URL");
  if (!rtsp_url) {
    return skip_or_fail(
        "SIMANEAT_APPS_TEST_RTSP_URL is required for single-stream-instance-segmenter e2e");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";
  const std::string model_path =
      configured_model_path("single-stream-instance-segmenter", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured instance segmentation model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir =
      create_test_output_dir("single-stream-instance-segmenter", "test_full_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const sima_examples::ScalarConfig& common =
      example_common_config("single-stream-instance-segmenter");
  const std::string insight_host = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       ? env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       : "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int save_every =
      e2e_int("single-stream-instance-segmenter", "testing.e2e.output", "save_every");
  const int total_saved_frames =
      e2e_int("single-stream-instance-segmenter", "testing.e2e.output", "total_saved_frames");

  {
    std::ofstream config_file(config_path);
    config_file
        << "source:\n"
        << "  rtsp_url: " << rtsp_url << "\n"
        << "  latency_ms: " << common.int_or("source.latency_ms", 100) << "\n"
        << "  tcp: " << (common.bool_or("source.tcp", true) ? "true" : "false") << "\n"
        << "model:\n"
        << "  path: " << model_path << "\n"
        << "  labels: "
        << common.string_or(
               "model.labels",
               "examples/segmentation/single-stream-instance-segmenter/src/common/coco_label.txt")
        << "\n"
        << "inference:\n"
        << "  frames: " << common.int_or("inference.frames", 0) << "\n"
        << "  min_score: " << common.double_or("inference.min_score", 0.55) << "\n"
        << "  nms_iou: " << common.double_or("inference.nms_iou", 0.60) << "\n"
        << "  max_detections: " << common.int_or("inference.max_detections", 50) << "\n"
        << "runtime:\n"
        << "  profile: " << (common.bool_or("runtime.profile", false) ? "true" : "false") << "\n"
        << "  profile_interval: " << common.int_or("runtime.profile_interval", 100) << "\n"
        << "output:\n"
        << "  save_dir: " << output_dir << "\n"
        << "  save_every: " << save_every << "\n"
        << "  mask_alpha: " << common.double_or("output.mask_alpha", 0.55) << "\n"
        << "  mask_threshold: " << common.double_or("output.mask_threshold", 0.50) << "\n"
        << "  draw_boxes: " << (common.bool_or("output.draw_boxes", true) ? "true" : "false")
        << "\n"
        << "  insight:\n"
        << "    host: " << insight_host << "\n"
        << "    video_port: " << video_port << "\n"
        << "    metadata_port: " << metadata_port << "\n";
  }

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
      std::cout << "[OK] single-stream instance segmenter produced " << files
                << " sampled output files\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
