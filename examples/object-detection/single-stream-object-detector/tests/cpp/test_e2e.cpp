#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "single-stream-object-detector";

struct SourceCase {
  const char* name;
  const char* env_key;
  const char* type;
  const char* codec;
  int fps;
  bool ssl_strict;
};

void record_unavailable_source(const std::string& fail_reason, const std::string& skip_reason,
                               int& rc) {
  if (require_e2e_mode()) {
    std::cerr << "[FAIL] " << fail_reason << "\n";
    rc = 1;
  } else {
    std::cerr << "[SKIP] " << skip_reason << "\n";
  }
}

int run_source_case(const std::string& binary, const std::string& model_path,
                    const SourceCase& source_case, const char* source_url) {
  const std::string output_dir =
      create_test_output_dir(kExampleName, std::string("test_full_pipeline_") + source_case.name);
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const std::string insight_host = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       ? env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       : "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int total_saved_frames = e2e_int(kExampleName, "testing.e2e.output", "total_saved_frames");
  ConfigScalars overrides{{"source.type", source_case.type},
                          {"source.codec", source_case.codec},
                          {"source.url", source_url},
                          {"source.ssl_strict", source_case.ssl_strict ? "true" : "false"},
                          {"model.path", model_path},
                          {"output.save_dir", output_dir},
                          {"output.insight.host", insight_host},
                          {"output.insight.video_port", std::to_string(video_port)},
                          {"output.insight.metadata_port", std::to_string(metadata_port)}};
  if (source_case.fps > 0) {
    overrides["source.fps"] = std::to_string(source_case.fps);
  }
  write_e2e_config(kExampleName, config_path, overrides);

  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const ProcessResult result = spawn_until_output_files(binary, {"--config", config_path.string()},
                                                        output_dir, total_saved_frames, timeout_ms);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] " << source_case.name << " exit code " << result.exit_code << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const int files = count_output_files(output_dir);
    if (files < total_saved_frames) {
      std::cerr << "[FAIL] " << source_case.name << " expected at least " << total_saved_frames
                << " sampled output files, got " << files << "\n";
      rc = 1;
    } else if (!all_output_files_nonempty(output_dir)) {
      std::cerr << "[FAIL] " << source_case.name << " some sampled output files are empty\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << source_case.name << " produced " << files
                << " sampled output files\n";
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
    return skip_or_fail("configured single-stream detector model not found under "
                        "SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::vector<SourceCase> source_cases = {
      {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
      {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
      {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 0, true},
      {"http_mjpeg", "SIMANEAT_TEST_HTTP_MJPEG_URL", "http", "mjpeg", 30, false},
  };

  int cases_run = 0;
  int rc = 0;
  for (const SourceCase& source_case : source_cases) {
    const char* source_url = env_or_null(source_case.env_key);
    if (!source_url) {
      record_unavailable_source(
          std::string(source_case.env_key) + " is required for " + source_case.name + " e2e",
          std::string("set ") + source_case.env_key + " to run " + source_case.name + " e2e", rc);
      continue;
    }
    ++cases_run;
    if (run_source_case(binary, model_path, source_case, source_url) != 0) {
      rc = 1;
    }
  }

  if (cases_run == 0) {
    if (require_e2e_mode()) {
      std::cerr << "[FAIL] no single-stream object detector source e2e URLs configured\n";
      return 1;
    } else {
      std::cerr << "[SKIP] no single-stream object detector source e2e URLs configured\n";
      return kSkipCode;
    }
  }

  return rc;
}
