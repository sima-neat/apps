// E2E test for multi-stream-tracker.
// Runs the RTSP pipeline and verifies sampled debug frames are written.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include "examples/tracking/multi-stream-tracker/src/cpp/utils/tracker_api.cpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-tracker";
constexpr const char* kE2eInsightHost = "127.0.0.1";

struct SourceCase {
  std::string codec;
  std::vector<std::string> urls;
};

/// The shared e2e writer keeps scalar keys only, so copy the `tracking:` block
/// (the class list) from the example's common config.
void append_tracking_block(const fs::path& config_path) {
  std::ifstream common(example_common_config_path(kExampleName));
  std::ofstream out(config_path, std::ios::app);
  bool in_block = false;
  std::string line;
  while (std::getline(common, line)) {
    const bool top_level = !line.empty() && line[0] != ' ' && line[0] != '#';
    if (top_level) {
      in_block = line.rfind("tracking:", 0) == 0;
    }
    if (in_block) {
      out << line << "\n";
    }
  }
}

/// Collects the `class:` values from the example's common config, so the e2e
/// expectation follows the shipped configuration instead of a hardcoded list.
std::vector<multi_stream_tracker::ClassEntry> configured_class_entries() {
  std::ifstream common(example_common_config_path(kExampleName));
  std::vector<multi_stream_tracker::ClassEntry> entries;
  std::string line;
  bool in_block = false;
  while (std::getline(common, line)) {
    const bool top_level = !line.empty() && line[0] != ' ' && line[0] != '#';
    if (top_level) {
      in_block = line.rfind("tracking:", 0) == 0;
    }
    if (!in_block) {
      continue;
    }
    const auto dash = line.find("- class:");
    if (dash == std::string::npos) {
      continue;
    }
    std::string value = line.substr(dash + 8);
    const auto hash = value.find('#');
    if (hash != std::string::npos) {
      value = value.substr(0, hash);
    }
    const auto first = value.find_first_not_of(" \t\"'");
    const auto last = value.find_last_not_of(" \t\r\n\"'");
    if (first == std::string::npos) {
      continue;
    }
    entries.push_back({{"class", value.substr(first, last - first + 1)}});
  }
  return entries;
}

/// Checks that every published track carries a configured class label, a
/// positive integer id and a box inside the frame. Whether any track is
/// published depends on the stream content, so an empty run is not a failure
/// here; that tracking produces tracks is covered by the unit tests.
bool tracking_metadata_is_valid(const MetadataJsonListenerResult& metadata,
                                const std::set<std::string>& expected_labels) {
  int checked = 0;
  for (const auto& message : metadata.messages) {
    // Insight wraps the array: {"data": {"tracks": [...]}, "frame_id": ..., "type": ...}
    const auto payload = nlohmann::json::parse(message.payload, nullptr, false);
    if (payload.is_discarded() || !payload.contains("data") || !payload["data"].is_object() ||
        !payload["data"].contains("tracks") || !payload["data"]["tracks"].is_array()) {
      std::cerr << "[FAIL] port " << message.port << ": 'data.tracks' must be an array\n";
      return false;
    }
    for (const auto& track : payload["data"]["tracks"]) {
      if (!track.contains("id") || !track.contains("label") || !track.contains("confidence") ||
          !track.contains("bbox") || track.size() != 4) {
        std::cerr << "[FAIL] port " << message.port << " frame " << message.frame_id
                  << ": unexpected tracking metadata keys\n";
        return false;
      }
      const auto label = track["label"].get<std::string>();
      if (expected_labels.count(label) == 0) {
        std::cerr << "[FAIL] port " << message.port << " frame " << message.frame_id << ": label '"
                  << label << "' is not a configured class\n";
        return false;
      }
      const auto id = track["id"].get<std::string>();
      if (id.empty() || id == "0" || id.find_first_not_of("0123456789") != std::string::npos) {
        std::cerr << "[FAIL] port " << message.port << " frame " << message.frame_id
                  << ": invalid track id '" << id << "'\n";
        return false;
      }
      const auto& bbox = track["bbox"];
      if (!bbox.is_array() || bbox.size() != 4 || bbox[0].get<double>() < 0.0 ||
          bbox[1].get<double>() < 0.0 || bbox[2].get<double>() <= 0.0 ||
          bbox[3].get<double>() <= 0.0) {
        std::cerr << "[FAIL] port " << message.port << " frame " << message.frame_id
                  << ": degenerate bbox\n";
        return false;
      }
      ++checked;
    }
  }
  std::cout << "[OK] validated " << checked << " published tracks\n";
  return true;
}

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
                    const SourceCase& source_case) {
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
  append_tracking_block(config_path);

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port_base;
  metadata_options.num_ports = 2;
  metadata_options.timeout_ms = 5000;
  metadata_options.require_all_ports = true;
  metadata_options.metadata_type = "tracking";
  metadata_options.data_array_key = "tracks";
  // min_object_count stays at 0 on purpose. Requiring a track here would make
  // the suite depend on the CI stream carrying a configured class: the H.265
  // fixture does, the H.264 fixture published 140 empty frames on both
  // streams. require_all_ports still fails if the application stops
  // publishing tracking metadata, and every published track is checked.
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
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] " << source_case.codec << " exit code " << result.exit_code << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    const int files = count_output_files(output_dir);
    if (files < total_saved_frames) {
      std::cerr << "[FAIL] " << source_case.codec << " expected at least " << total_saved_frames
                << " sampled output files, got " << files << "\n";
      rc = 1;
    } else if (!all_output_files_nonempty(output_dir)) {
      std::cerr << "[FAIL] " << source_case.codec << " some sampled output files are empty\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << source_case.codec << " multi-stream tracker produced " << files
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
      std::set<std::string> expected_labels;
      for (const auto& config : multi_stream_tracker::parse_class_configs(
               configured_class_entries())) {
        expected_labels.insert(config.label);
      }
      if (!tracking_metadata_is_valid(metadata, expected_labels)) {
        rc = 1;
      } else {
        std::cout << "[OK] " << source_case.codec << " tracking metadata received on "
                  << metadata.ports_with_valid_json.size()
                  << " streams with valid labels and stable ids\n";
      }
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

  const std::vector<SourceCase> source_cases = {
      {"h264", rtsp_h264_urls_from_env()},
      {"h265", rtsp_h265_urls_from_env()},
  };

  int cases_run = 0;
  int rc = 0;
  for (const SourceCase& source_case : source_cases) {
    if (source_case.urls.size() < 2) {
      record_unavailable_source("need at least two RTSP " + source_case.codec +
                                    " URLs for multistream e2e",
                                "set at least two RTSP " + source_case.codec + " URLs to run " +
                                    source_case.codec + " multistream e2e",
                                rc);
      continue;
    }
    ++cases_run;
    if (run_source_case(binary, model_path, source_case) != 0) {
      rc = 1;
    }
  }

  if (cases_run == 0) {
    if (require_e2e_mode()) {
      std::cerr << "[FAIL] no multi-stream tracker RTSP e2e URLs configured\n";
      return 1;
    }
    std::cerr << "[SKIP] no multi-stream tracker RTSP e2e URLs configured\n";
    return kSkipCode;
  }

  return rc;
}
