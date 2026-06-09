// E2E test for multi-stream-object-detector.
// Runs the RTSP pipeline and verifies sampled debug frames are written.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
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
  const std::string model_path = configured_model_path("multi-stream-object-detector", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured detector model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir = create_test_output_dir(
      "multi-stream-object-detector", "test_multi_stream_insight_and_save_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const sima_examples::ScalarConfig& common = example_common_config("multi-stream-object-detector");
  const std::string insight_host = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       ? env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       : "127.0.0.1";
  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const bool input_tcp = common.bool_or("input.tcp", true);
  const int input_latency_ms = common.int_or("input.latency_ms", 100);
  const int worker_count = common.int_or("runtime.worker_count", 4);
  const int mailbox_depth = common.int_or("runtime.mailbox_depth", 1);
  const bool profile = common.bool_or("runtime.profile", false);
  const int frames = common.int_or("inference.frames", 0);
  const int fps = common.int_or("inference.fps", 0);
  const bool video_enabled = common.bool_or("output.video_enabled", true);
  const std::string video_mode = common.string_or("output.video_mode", "annotated");
  const int save_every =
      e2e_int("multi-stream-object-detector", "testing.e2e.output", "save_every");
  const int total_saved_frames =
      e2e_int("multi-stream-object-detector", "testing.e2e.output", "total_saved_frames");
  const double min_score = e2e_double("multi-stream-object-detector", "inference", "min_score");
  const double nms_iou = e2e_double("multi-stream-object-detector", "inference", "nms_iou");
  const int max_detections = e2e_int("multi-stream-object-detector", "inference", "max_detections");
  {
    std::ofstream out(config_path);
    out << "model:\n"
        << "  path: " << model_path
        << "\n"
           "\n"
           "input:\n"
           "  tcp: "
        << (input_tcp ? "true" : "false") << "\n"
        << "  latency_ms: " << input_latency_ms << "\n"
        << "\n"
           "runtime:\n"
           "  worker_count: "
        << worker_count
        << "\n"
           "  mailbox_depth: "
        << mailbox_depth
        << "\n"
           "  profile: "
        << (profile ? "true" : "false")
        << "\n"
           "\n"
           "inference:\n"
           "  frames: "
        << frames
        << "\n"
           "  fps: "
        << fps << "\n"
        << "  min_score: " << min_score << "\n"
        << "  nms_iou: " << nms_iou << "\n"
        << "  max_detections: " << max_detections << "\n"
        << "\n"
           "output:\n"
           "  insight:\n"
           "    host: "
        << insight_host
        << "\n"
           "    video_port_base: "
        << video_port_base
        << "\n"
           "    metadata_port_base: "
        << metadata_port_base
        << "\n"
           "  video_enabled: "
        << (video_enabled ? "true" : "false")
        << "\n"
           "  video_mode: "
        << video_mode
        << "\n"
           "  debug_dir: "
        << output_dir
        << "\n"
           "  save_every: "
        << save_every
        << "\n"
           "\n"
           "streams:\n";
    for (std::size_t index = 0; index < 2; ++index) {
      out << "  - " << rtsp_urls[index] << "\n";
    }
  }

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(binary, {"--config", config_path.string()},
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
      std::cout << "[OK] multi-camera object detector produced " << files
                << " sampled output files\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
