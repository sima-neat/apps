// E2E test for fastsam-multistream.
// Runs the multi-stream RTSP pipeline and verifies segmentation metadata is streamed.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

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

  const std::vector<std::string> rtsp_urls = rtsp_h264_urls_from_env();
  if (rtsp_urls.size() < 4) {
    return skip_or_fail("need four RTSP URLs for fastsam-multistream e2e");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "assets/models";
  const std::string model_path = configured_model_path("fastsam-multistream", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured FastSAM model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const sima_examples::ScalarConfig& config = example_common_config("fastsam-multistream");
  const fs::path clip_image =
      fs::path(models_dir) / fs::path(config.string_or("clip.image_encoder_path", "")).filename();
  const fs::path clip_text =
      fs::path(models_dir) / fs::path(config.string_or("clip.text_encoder_path", "")).filename();
  const fs::path clip_consts =
      fs::path(models_dir) / fs::path(config.string_or("clip.text_host_consts", "")).filename();
  for (const fs::path& artifact : {clip_image, clip_text, clip_consts}) {
    if (!fs::exists(artifact)) {
      return skip_or_fail("configured CLIP model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
    }
  }

  const std::string output_dir =
      create_test_output_dir("fastsam-multistream", "test_multi_stream_segmentation_metadata");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path text_features =
      example_common_config_path("fastsam-multistream").parent_path() / "text_features.npy";
  if (!fs::exists(text_features)) {
    return skip_or_fail("src/common/text_features.npy not found for fastsam-multistream");
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  write_e2e_config(
      "fastsam-multistream", config_path,
      {{"model.path", model_path},
       {"clip.image_encoder_path", clip_image.string()},
       {"clip.text_encoder_path", clip_text.string()},
       {"clip.text_host_consts", clip_consts.string()},
       {"clip.text_features_path", text_features.string()},
       {"runtime.frames", "90"},
       {"output.insight.host", kE2eInsightHost},
       {"output.insight.video_port_base", std::to_string(video_port_base)},
       {"output.insight.metadata_port_base", std::to_string(metadata_port_base)}},
      {{"source.rtsp_urls", {rtsp_urls[0], rtsp_urls[1], rtsp_urls[2], rtsp_urls[3]}}});

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port_base;
  metadata_options.num_ports = 4;
  metadata_options.timeout_ms = 5000;
  metadata_options.metadata_type = "segmentation";
  metadata_options.data_array_key = "segments";
  metadata_options.require_all_ports = true;
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << metadata_listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const ProcessResult result = spawn_and_wait(binary, {config_path.string()}, timeout_ms);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const MetadataJsonListenerResult metadata = metadata_listener.wait_for_messages();
    if (!metadata.success) {
      std::cerr << "[FAIL] segmentation metadata was not received on all streams: "
                << metadata.error << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] fastsam segmentation metadata received on all 4 streams\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}
