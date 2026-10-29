// E2E test for adaptive-resolution-object-detector (C++).
// Runs every mode over two RTSP streams and verifies that each stream publishes
// detection metadata and moving annotated frames.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "adaptive-resolution-object-detector";
constexpr const char* kE2eInsightHost = "127.0.0.1";
constexpr int kStreams = 2;

// write_e2e_config writes lists as plain scalars, and streams.sources is a list
// of {id, url} mappings, so the entries are inserted under `streams:` here.
void insert_stream_sources(const fs::path& config_path, const std::vector<std::string>& urls) {
  std::ifstream in(config_path);
  std::stringstream patched;
  std::string line;
  bool inserted = false;
  while (std::getline(in, line)) {
    patched << line << "\n";
    if (!inserted && line == "streams:") {
      patched << "  sources:\n";
      for (std::size_t index = 0; index < urls.size(); ++index) {
        patched << "    - id: cam-" << index + 1 << "\n"
                << "      url: \"" << urls[index] << "\"\n";
      }
      inserted = true;
    }
  }
  in.close();
  if (!inserted) {
    throw std::runtime_error("e2e config has no streams section: " + config_path.string());
  }
  std::ofstream out(config_path);
  out << patched.str();
}

int run_mode_case(const std::string& binary, const std::string& model_path,
                  const fs::path& labels_file, const std::vector<std::string>& urls,
                  const std::string& mode, const std::string& label) {
  const std::string output_dir = create_test_output_dir(
      kExampleName, "test_" + mode + "_" + label + "_publishes_every_stream");
  if (output_dir.empty()) {
    return 1;
  }

  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int total_saved_frames = e2e_int(kExampleName, "testing.e2e.output", "total_saved_frames");
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  // group runs with one stream per group, so the two streams are detected by
  // two independent processes publishing on their own channels.
  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  write_e2e_config(kExampleName, config_path,
                   {{"mode", mode},
                    {"model.path", model_path},
                    {"model.labels", labels_file.string()},
                    {"streams.max_streams", std::to_string(kStreams)},
                    {"group.size", "1"},
                    {"inference.frames", "140"},
                    {"output.save_dir", output_dir},
                    {"output.insight.host", kE2eInsightHost},
                    {"output.insight.video_port", std::to_string(video_port)},
                    {"output.insight.metadata_port", std::to_string(metadata_port)}});
  insert_stream_sources(config_path, {urls.begin(), urls.begin() + kStreams});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = kStreams;
  metadata_options.timeout_ms = 5000;
  metadata_options.require_all_ports = true;
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] " << label << " metadata listener failed: " << metadata_listener.error()
              << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const ProcessResult result = spawn_until_output_files(binary, {"--config", config_path.string()},
                                                        output_dir, total_saved_frames, timeout_ms);

  int rc = 0;
  const std::string exit_problem_text = exit_problem(result);
  if (!exit_problem_text.empty()) {
    std::cerr << "[FAIL] " << label << " " << exit_problem_text << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const std::string problem = streamed_frames_problem(output_dir, total_saved_frames, kStreams);
    if (!problem.empty()) {
      std::cerr << "[FAIL] " << label << " " << problem << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << label << " produced " << count_output_files(output_dir)
                << " sampled output files\n";
    }
  }
  if (rc == 0) {
    const MetadataJsonListenerResult metadata = metadata_listener.wait_for_messages();
    if (!metadata.success) {
      std::cerr << "[FAIL] " << label
                << " object-detection metadata was not received on all streams: " << metadata.error
                << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << label << " object-detection metadata received on "
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
  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("configured detector model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const fs::path labels_file =
      example_common_config_path(kExampleName).parent_path() / "coco_label.txt";
  if (!fs::exists(labels_file)) {
    return skip_or_fail("src/common/coco_label.txt not found for " + std::string(kExampleName));
  }

  const std::vector<MultiStreamSourceCase> source_cases = {{"h264", rtsp_h264_urls_from_env()}};
  const int cases_rc = run_multistream_source_cases(
      "adaptive resolution object detector", source_cases, kStreams,
      [&](const MultiStreamSourceCase& source_case) {
        int case_rc = 0;
        for (const std::string mode : {"live", "scale", "group"}) {
          for (const std::string& model_path : model_paths) {
            const std::string label =
                model_case_label(mode + "_" + source_case.codec, model_path, model_paths.size());
            if (run_mode_case(binary, model_path, labels_file, source_case.urls, mode, label) !=
                0) {
              case_rc = 1;
            }
          }
        }
        return case_rc;
      });
  return rc != 0 ? rc : cases_rc;
}
