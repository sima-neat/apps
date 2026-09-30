// E2E test for multi-stream-people-tracker.
// Runs the RTSP pipeline and verifies sampled debug frames are written.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-people-tracker";
constexpr const char* kE2eInsightHost = "127.0.0.1";

int run_source_case(const std::string& binary, const std::string& model_path,
                    const MultiStreamSourceCase& source_case) {
  const std::string output_dir = create_test_output_dir(
      kExampleName, "test_multi_stream_" + source_case.codec + "_insight_and_save_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int total_saved_frames = e2e_int(kExampleName, "testing.e2e.output", "total_saved_frames");
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  write_e2e_config(kExampleName, config_path,
                   {{"model.path", model_path},
                    {"input.codec", source_case.codec},
                    {"output.debug_dir", output_dir},
                    {"output.insight.host", kE2eInsightHost},
                    {"output.insight.video_port_base", std::to_string(video_port_base)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port_base)},
                    {"inference.frames", "140"}},
                   {{"streams", {source_case.urls[0], source_case.urls[1]}}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port_base;
  metadata_options.num_ports = 2;
  metadata_options.timeout_ms = 5000;
  metadata_options.require_all_ports = true;
  metadata_options.metadata_type = "tracking";
  metadata_options.data_array_key = "tracks";
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] " << source_case.codec
              << " metadata listener failed: " << metadata_listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const ProcessResult result = spawn_until_output_files(binary, {"--config", config_path.string()},
                                                        output_dir, total_saved_frames, timeout_ms);

  int rc = 0;
  const std::string exit_problem_text = exit_problem(result);
  if (!exit_problem_text.empty()) {
    std::cerr << "[FAIL] " << source_case.codec << " " << exit_problem_text << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const int files = count_output_files(output_dir);
    // Two streams are configured above, so both must advance on their own.
    const std::string problem = streamed_frames_problem(output_dir, total_saved_frames, 2);
    if (!problem.empty()) {
      std::cerr << "[FAIL] " << source_case.codec << " " << problem << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << source_case.codec << " multi-camera people tracker produced " << files
                << " sampled output files\n";
    }
  }
  if (rc == 0) {
    const MetadataJsonListenerResult metadata = metadata_listener.wait_for_messages();
    if (!metadata.success) {
      std::cerr << "[FAIL] " << source_case.codec
                << " tracking metadata was not received on all streams: " << metadata.error << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << source_case.codec << " tracking metadata received on "
                << metadata.ports_with_valid_json.size() << " streams\n";
    }
  }

  remove_dir(output_dir);
  return rc;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  const std::string model_path = configured_model_path(kExampleName, models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail("configured detector model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::vector<MultiStreamSourceCase> source_cases = {
      {"h264", rtsp_h264_urls_from_env()},
      {"h265", rtsp_h265_urls_from_env()},
  };

  return run_multistream_source_cases("multi-stream people tracker", source_cases, 2,
                                      [&](const MultiStreamSourceCase& source_case) {
                                        return run_source_case(binary, model_path, source_case);
                                      });
}
