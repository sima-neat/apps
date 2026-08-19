#include "support/testing/metadata_json_listener.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "multi-stream-blazepose3d";
constexpr const char* kInsightHost = "127.0.0.1";
constexpr const char* kPoseModel = "blazepose_heavy_3d_bf16_nopad_neat_mpk.tar.gz";
constexpr const char* kDetectorModel = "yolo26m-det-int8-b1.tar.gz";

void write_config(const fs::path& path, const fs::path& detector, const fs::path& pose,
                  const std::string& codec, const std::vector<std::string>& urls,
                  int video_port_base, int metadata_port_base) {
  std::ofstream output(path);
  output << "models:\n  detector_path: " << detector.string() << "\n  pose_path: " << pose.string()
         << "\nstreams:\n";
  for (std::size_t index = 0; index < urls.size(); ++index) {
    output << "  - id: camera" << index << "\n    url: " << urls[index] << "\n    codec: " << codec
           << "\n    insight_channel: " << index << "\n";
  }
  output << "input:\n  tcp: true\n  latency_ms: 100\n"
            "detector:\n  min_score: 0.30\n  nms_iou: 0.60\n  max_detections: 100\n"
            "  max_inflight_per_stream: 4\n  max_inflight_total: 16\n"
            "pose:\n  max_people_per_frame: 2\n  roi_scale: 1.65\n"
            "  presence_threshold: 0.50\n  job_timeout_ms: 10000\n  max_pending_jobs: 64\n"
            "runtime:\n  frames: 8\noutput:\n  insight:\n    host: "
         << kInsightHost << "\n    video_port_base: " << video_port_base
         << "\n    metadata_port_base: " << metadata_port_base << "\n  video_enabled: true\n";
}

bool validate_metadata(const MetadataJsonListenerResult& result, std::string& error) {
  bool found_pose = false;
  try {
    for (const auto& message : result.messages) {
      const auto poses = nlohmann::json::parse(message.payload).at("data").at("poses");
      if (!poses.is_array()) {
        error = "pose metadata data.poses is not an array";
        return false;
      }
      for (const auto& pose : poses) {
        found_pose = true;
        if (!pose.contains("keypoints") || !pose["keypoints"].is_array() ||
            pose["keypoints"].size() != 33) {
          error = "a published pose did not contain exactly 33 keypoints";
          return false;
        }
      }
    }
  } catch (const std::exception& exception) {
    error = std::string("failed to validate pose metadata: ") + exception.what();
    return false;
  }
  if (!found_pose) {
    error = "no BlazePose result was published";
  }
  return found_pose;
}

int run_case(const std::string& binary, const fs::path& detector, const fs::path& pose,
             const std::string& codec, const std::vector<std::string>& urls) {
  const fs::path run_dir = create_test_scratch_dir(kExampleName, "e2e_" + codec);
  const int video_port_base = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port_base =
      env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 300000);
  const fs::path config = run_dir / "config.yaml";
  write_config(config, detector, pose, codec, urls, video_port_base, metadata_port_base);

  MetadataJsonListenerOptions listener_options;
  listener_options.host = kInsightHost;
  listener_options.base_port = metadata_port_base;
  listener_options.num_ports = static_cast<int>(urls.size());
  listener_options.timeout_ms = 10000;
  listener_options.require_all_ports = true;
  listener_options.metadata_type = "pose-estimation";
  listener_options.data_array_key = "poses";
  MetadataJsonListener listener(listener_options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener: " << listener.error() << "\n";
    remove_dir(run_dir.string());
    return 1;
  }

  const ProcessResult process = spawn_and_wait(binary, {"--config", config.string()}, timeout_ms);
  const MetadataJsonListenerResult metadata = listener.wait_for_messages();
  int result = 0;
  if (process.exit_code != 0) {
    std::cerr << "[FAIL] " << codec << " app exit code " << process.exit_code << "\nstdout:\n"
              << process.stdout_text << "\nstderr:\n"
              << process.stderr_text << "\n";
    result = 1;
  } else if (!metadata.success) {
    std::cerr << "[FAIL] " << codec << " metadata: " << metadata.error << "\n";
    result = 1;
  } else {
    std::string error;
    if (!validate_metadata(metadata, error)) {
      std::cerr << "[FAIL] " << codec << " metadata: " << error << "\n";
      result = 1;
    } else {
      std::cout << "[OK] " << codec << " produced 33-keypoint metadata on " << urls.size()
                << " streams\n";
    }
  }
  remove_dir(run_dir.string());
  return result;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const fs::path models_dir = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR") != nullptr
                                  ? fs::path(env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR"))
                                  : fs::path("models");
  const fs::path detector = env_or_null("SIMANEAT_APPS_TEST_DETECTOR_MODEL") != nullptr
                                ? fs::path(env_or_null("SIMANEAT_APPS_TEST_DETECTOR_MODEL"))
                                : models_dir / kDetectorModel;
  const fs::path pose = env_or_null("SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL") != nullptr
                            ? fs::path(env_or_null("SIMANEAT_APPS_TEST_BLAZEPOSE_MODEL"))
                            : models_dir / kPoseModel;
  if (!fs::exists(detector) || !fs::exists(pose)) {
    return skip_or_fail("YOLO26 and BlazePose model packages are required");
  }

  const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
      {"h264", rtsp_h264_urls_from_env()}, {"h265", rtsp_h265_urls_from_env()}};
  int result = 0;
  int cases_run = 0;
  for (const auto& [codec, urls] : cases) {
    if (urls.empty()) {
      continue;
    }
    ++cases_run;
    result |= run_case(argv[1], detector, pose, codec, urls);
  }
  if (cases_run == 0) {
    return skip_or_fail("no multi-stream H.264 or H.265 RTSP URLs configured");
  }
  return result;
}
