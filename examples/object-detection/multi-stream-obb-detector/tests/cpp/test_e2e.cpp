// E2E test for multi-stream-obb-detector.
// Runs two RTSP streams through the detector, then checks the saved debug frames and the
// oriented-box metadata published to Insight.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-obb-detector";
constexpr const char* kE2eInsightHost = "127.0.0.1";

std::set<std::string> read_labels(const fs::path& labels_file) {
  std::set<std::string> labels;
  std::ifstream input(labels_file);
  for (std::string line; std::getline(input, line);) {
    if (!line.empty()) {
      labels.insert(line);
    }
  }
  return labels;
}

/// Checks that every published object is a well-formed oriented box. Whether any object is
/// published depends on the stream: the model detects top-down aerial objects, and the CI
/// streams are street scenes. An empty run is therefore not a failure here; the unit test
/// covers the object values.
bool oriented_objects_are_valid(const MetadataJsonListenerResult& metadata,
                                const std::set<std::string>& labels) {
  int checked = 0;
  for (const auto& message : metadata.messages) {
    const std::string where = "port " + std::to_string(message.port) + " frame " + message.frame_id;
    // Insight wraps the array: {"data": {"objects": [...]}, "frame_id": ..., "type": ...}
    const auto payload = nlohmann::json::parse(message.payload, nullptr, false);
    if (payload.is_discarded() || !payload.contains("data") ||
        !payload["data"].contains("objects") || !payload["data"]["objects"].is_array()) {
      std::cerr << "[FAIL] " << where << ": 'data.objects' must be an array\n";
      return false;
    }
    for (const auto& obj : payload["data"]["objects"]) {
      if (obj.size() != 5 || !obj.contains("id") || !obj.contains("label") ||
          !obj.contains("confidence") || !obj.contains("bbox") || !obj.contains("oriented_bbox")) {
        std::cerr << "[FAIL] " << where << ": unexpected object keys " << obj.dump() << "\n";
        return false;
      }
      if (labels.count(obj["label"].get<std::string>()) == 0) {
        std::cerr << "[FAIL] " << where << ": label " << obj["label"] << " is not a DOTA class\n";
        return false;
      }
      const double confidence = obj["confidence"].get<double>();
      const auto& oriented = obj["oriented_bbox"];
      const auto& bbox = obj["bbox"];
      if (confidence < 0.0 || confidence > 1.0 || oriented.size() != 5 || bbox.size() != 4 ||
          oriented[2].get<double>() <= 0.0 || oriented[3].get<double>() <= 0.0 ||
          !std::isfinite(oriented[4].get<double>())) {
        std::cerr << "[FAIL] " << where << ": degenerate object " << obj.dump() << "\n";
        return false;
      }
      const double bbox_cx = bbox[0].get<double>() + bbox[2].get<double>() / 2;
      const double bbox_cy = bbox[1].get<double>() + bbox[3].get<double>() / 2;
      // Each published value is rounded to 0.1 px, so the centers can differ by 0.125 px.
      if (std::abs(bbox_cx - oriented[0].get<double>()) > 0.15 ||
          std::abs(bbox_cy - oriented[1].get<double>()) > 0.15) {
        std::cerr << "[FAIL] " << where << ": bbox is not centered on the oriented box "
                  << obj.dump() << "\n";
        return false;
      }
      ++checked;
    }
  }
  std::cout << "[OK] validated " << checked << " published oriented boxes\n";
  return true;
}

int run_source_case(const std::string& binary, const std::string& model_path,
                    const fs::path& labels_file, const MultiStreamSourceCase& source_case,
                    const std::string& label) {
  const std::string output_dir = create_test_output_dir(
      kExampleName, "test_multi_stream_" + label + "_insight_and_save_pipeline");
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
                    {"model.labels", labels_file.string()},
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
    // Two streams are configured above, so both must advance on their own.
    const std::string problem = streamed_frames_problem(output_dir, total_saved_frames, 2);
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
    } else if (!oriented_objects_are_valid(metadata, read_labels(labels_file))) {
      rc = 1;
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
      example_common_config_path(kExampleName).parent_path() / "dota_label.txt";
  if (!fs::exists(labels_file)) {
    return skip_or_fail("src/common/dota_label.txt not found for multi-stream-obb-detector");
  }

  const std::vector<MultiStreamSourceCase> source_cases = {
      {"h264", rtsp_h264_urls_from_env()},
      {"h265", rtsp_h265_urls_from_env()},
  };

  const int cases_rc = run_multistream_source_cases(
      "multi-stream oriented-box detector", source_cases, 2,
      [&](const MultiStreamSourceCase& source_case) {
        int case_rc = 0;
        for (const std::string& model_path : model_paths) {
          const std::string label =
              model_case_label(source_case.codec, model_path, model_paths.size());
          if (run_source_case(binary, model_path, labels_file, source_case, label) != 0) {
            case_rc = 1;
          }
        }
        return case_rc;
      });
  return rc != 0 ? rc : cases_rc;
}
