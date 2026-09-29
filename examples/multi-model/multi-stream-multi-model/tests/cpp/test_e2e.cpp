// E2E test for multi-stream-multi-model.
// Runs four RTSP streams through four different models and verifies that each stream publishes
// its own Insight contract and writes its own annotated debug frames.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-multi-model";
constexpr const char* kE2eInsightHost = "127.0.0.1";

/// One slot per stream: the task, the head layout of its package, the package file the scope
/// downloads, and the Insight contract that stream must publish on.
struct StreamSlot {
  const char* task;
  const char* decode;
  const char* model_file;
  const char* metadata_type;
  const char* data_array_key;
};

const std::vector<StreamSlot>& stream_slots() {
  static const std::vector<StreamSlot> slots = {
      {"detection", "yolov8", "yolo_11s_mpk.tar.gz", "object-detection", "objects"},
      {"segmentation", "yolov8", "yolo_11s_seg_mpk.tar.gz", "segmentation", "segments"},
      {"pose", "yolo26", "yolo26m-pose-int8-b1.tar.gz", "pose-estimation", "poses"},
      {"detection", "yolo26", "yolo26m-det-int8-b1.tar.gz", "object-detection", "objects"},
  };
  return slots;
}

struct SourceCase {
  std::string codec;
  std::vector<std::string> urls;
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

/// Append the per-stream mapping list to a generated config.
///
/// `write_e2e_config` rebuilds the config from flat scalars, which cannot express a YAML sequence
/// of mappings, so the stream list is written here instead of overridden through the harness.
void append_streams_block(const fs::path& config_path, const std::vector<std::string>& urls,
                          const std::vector<std::string>& model_paths) {
  std::ofstream out(config_path, std::ios::app);
  out << "streams:\n";
  for (std::size_t index = 0; index < stream_slots().size(); ++index) {
    const StreamSlot& slot = stream_slots()[index];
    out << "  - url: " << urls[index % urls.size()] << "\n"
        << "    task: " << slot.task << "\n"
        << "    decode: " << slot.decode << "\n"
        << "    model: " << model_paths[index] << "\n";
  }
}

using MetadataListeners = std::vector<std::unique_ptr<MetadataJsonListener>>;

/// Bind one listener per stream. The four channels carry three different Insight contracts, so a
/// single shared listener could not tell a wrong-typed stream from a silent one.
///
/// The sockets must be bound before the application starts, or its datagrams arrive with nothing
/// listening and are dropped.
MetadataListeners bind_metadata_listeners(const std::string& codec, int metadata_port_base) {
  MetadataListeners listeners;
  for (std::size_t index = 0; index < stream_slots().size(); ++index) {
    MetadataJsonListenerOptions options;
    options.host = kE2eInsightHost;
    options.base_port = metadata_port_base + static_cast<int>(index);
    options.num_ports = 1;
    options.timeout_ms = 5000;
    options.metadata_type = stream_slots()[index].metadata_type;
    options.data_array_key = stream_slots()[index].data_array_key;
    options.require_all_ports = true;
    listeners.push_back(std::make_unique<MetadataJsonListener>(options));
    if (!listeners.back()->ok()) {
      std::cerr << "[FAIL] " << codec << " metadata listener for stream " << index
                << " failed: " << listeners.back()->error() << "\n";
      return {};
    }
  }
  return listeners;
}

bool check_metadata(const std::string& codec, int metadata_port_base,
                    const MetadataListeners& listeners) {
  bool ok = true;
  for (std::size_t index = 0; index < listeners.size(); ++index) {
    const MetadataJsonListenerResult result = listeners[index]->wait_for_messages();
    if (!result.success) {
      std::cerr << "[FAIL] " << codec << " stream " << index << " did not publish "
                << stream_slots()[index].metadata_type << " metadata on port "
                << metadata_port_base + static_cast<int>(index) << ": " << result.error << "\n";
      ok = false;
    } else {
      std::cout << "[OK] " << codec << " stream " << index << " published "
                << stream_slots()[index].metadata_type << " metadata\n";
    }
  }
  return ok;
}

bool every_stream_saved_a_frame(const std::string& output_dir) {
  for (std::size_t index = 0; index < stream_slots().size(); ++index) {
    const std::string prefix = "stream_" + std::to_string(index) + "_frame_";
    bool found = false;
    for (const auto& entry : fs::recursive_directory_iterator(output_dir)) {
      if (entry.is_regular_file() && entry.path().filename().string().rfind(prefix, 0) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "[FAIL] stream " << index << " saved no annotated debug frame\n";
      return false;
    }
  }
  return true;
}

int run_source_case(const std::string& binary, const std::vector<std::string>& model_paths,
                    const fs::path& labels_file, const SourceCase& source_case) {
  const std::string output_dir = create_test_output_dir(
      kExampleName, "test_every_stream_publishes_its_own_model_contract_" + source_case.codec);
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
                   {{"model.labels", labels_file.string()},
                    {"input.codec", source_case.codec},
                    {"output.debug_dir", output_dir},
                    {"output.insight.host", kE2eInsightHost},
                    {"output.insight.video_port_base", std::to_string(video_port_base)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port_base)},
                    {"inference.frames", "140"}});
  append_streams_block(config_path, source_case.urls, model_paths);

  const MetadataListeners listeners =
      bind_metadata_listeners(source_case.codec, metadata_port_base);
  if (listeners.empty()) {
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
    } else if (!every_stream_saved_a_frame(output_dir)) {
      rc = 1;
    } else {
      std::cout << "[OK] " << source_case.codec << " multi-model pipeline produced " << files
                << " sampled output files across " << stream_slots().size() << " streams\n";
    }
  }
  if (rc == 0 && !check_metadata(source_case.codec, metadata_port_base, listeners)) {
    rc = 1;
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

  std::vector<std::string> model_paths;
  for (const StreamSlot& slot : stream_slots()) {
    const fs::path model_path = fs::path(models_dir) / slot.model_file;
    if (!fs::exists(model_path)) {
      return skip_or_fail("configured model not found under SIMANEAT_APPS_TEST_MODELS_DIR: " +
                          model_path.string());
    }
    model_paths.push_back(model_path.string());
  }

  const fs::path labels_file =
      example_common_config_path(kExampleName).parent_path() / "coco_label.txt";
  if (!fs::exists(labels_file)) {
    return skip_or_fail("src/common/coco_label.txt not found for multi-stream-multi-model");
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
    if (run_source_case(binary, model_paths, labels_file, source_case) != 0) {
      rc = 1;
    }
  }

  if (cases_run == 0) {
    if (require_e2e_mode()) {
      std::cerr << "[FAIL] no multi-stream multi-model RTSP e2e URLs configured\n";
      return 1;
    }
    std::cerr << "[SKIP] no multi-stream multi-model RTSP e2e URLs configured\n";
    return kSkipCode;
  }

  return rc;
}
