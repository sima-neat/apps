// End-to-end test: the RTSP stream's faces reach Insight as pose-estimation metadata, and the
// application stops cleanly when interrupted. A local UDP listener stands in for Insight, so only
// an RTSP H.264 source (SIMANEAT_TEST_RTSP_H264_URL) and the model are needed.
#include "sima_examples/labels.h"
#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {
constexpr const char* kE2eInsightHost = "127.0.0.1";
const std::vector<std::string> kLandmarkNames = {"eye_l", "eye_r", "nose_tip", "mouth_l",
                                                 "mouth_r"};
// The shipped config's inference values, which every pose must respect.
constexpr double kMinScore = 0.25;
constexpr std::size_t kMaxDetections = 50;

// Why a pose message breaks the contract, or "" when every pose holds: at most kMaxDetections
// poses, the label from the label file, five named keypoints, each scoring above kMinScore.
std::string poses_problem(const nlohmann::json& poses, const std::string& label) {
  if (poses.size() > kMaxDetections) {
    return "more than " + std::to_string(kMaxDetections) + " poses in one message";
  }
  for (const auto& pose : poses) {
    if (pose.at("label") != label) {
      return "pose label " + pose.at("label").dump() + " is not the label file's " + label;
    }
    std::vector<std::string> names;
    for (const auto& point : pose.at("keypoints")) {
      names.push_back(point.at("name"));
      if (point.at("confidence").get<double>() <= kMinScore) {
        return "keypoint confidence " + point.at("confidence").dump() + " is not above " +
               std::to_string(kMinScore);
      }
    }
    if (names != kLandmarkNames) {
      return "keypoints are not the five named landmarks: " + pose.at("keypoints").dump();
    }
  }
  return "";
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* rtsp_url = env_or_null("SIMANEAT_TEST_RTSP_H264_URL");
  if (!rtsp_url) {
    return skip_or_fail(
        "SIMANEAT_TEST_RTSP_H264_URL is required for single-stream-thermal-face-detector e2e");
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  const std::string model_path =
      configured_model_path("single-stream-thermal-face-detector", models_dir);
  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "configured yolov5s-face model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const std::string output_dir =
      create_test_output_dir("single-stream-thermal-face-detector", "test_metadata_pipeline");
  if (output_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  write_e2e_config("single-stream-thermal-face-detector", config_path,
                   {{"model.path", model_path},
                    {"input.source", rtsp_url},
                    {"output.insight.host", kE2eInsightHost},
                    {"output.insight.video_port_base", std::to_string(video_port)},
                    {"output.insight.metadata_port_base", std::to_string(metadata_port)},
                    {"inference.frames", "0"},
                    {"inference.min_score", std::to_string(kMinScore)},
                    {"inference.max_detections", std::to_string(kMaxDetections)}});

  MetadataJsonListenerOptions metadata_options;
  metadata_options.host = kE2eInsightHost;
  metadata_options.base_port = metadata_port;
  metadata_options.num_ports = 1;
  metadata_options.timeout_ms = 5000;
  metadata_options.metadata_type = "pose-estimation";
  metadata_options.data_array_key = "poses";
  metadata_options.require_all_ports = true;
  MetadataJsonListener metadata_listener(metadata_options);
  if (!metadata_listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << metadata_listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  // The application runs until stopped (frames = 0). Once the listener has a verdict,
  // the metadata it asked for or a datagram that was not valid metadata, the harness
  // interrupts the application and the result carries how it shut down; the shipped
  // config's finite run is the Python suite's.
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  MetadataJsonListenerResult metadata;
  const ProcessResult result = spawn_until(
      binary, {"--config", config_path.string()},
      [&] { return metadata_listener.poll_messages(metadata, 250); }, timeout_ms);

  int rc = 0;
  const std::string exit_problem_text = exit_problem(result);
  if (!exit_problem_text.empty()) {
    std::cerr << "[FAIL] " << exit_problem_text << "\n";
    std::cerr << "stdout:\n" << result.stdout_text << "\n";
    std::cerr << "stderr:\n" << result.stderr_text << "\n";
    rc = 1;
  } else {
    if (!metadata.success) {
      std::cerr << "[FAIL] pose-estimation metadata was not received: " << metadata.error << "\n";
      rc = 1;
    } else {
      const std::string label =
          sima_examples::load_labels(
              "examples/face-detection/single-stream-thermal-face-detector/src/common/"
              "face_label.txt")
              .front();
      std::string problem;
      for (const auto& message : metadata.messages) {
        if (problem.empty()) {
          problem =
              poses_problem(nlohmann::json::parse(message.payload).at("data").at("poses"), label);
        }
      }
      if (!problem.empty()) {
        std::cerr << "[FAIL] " << problem << "\n";
        rc = 1;
      } else {
        std::cout << "[OK] yolov5s-face published valid pose-estimation metadata on "
                  << metadata.ports_with_valid_json.size() << " port(s) and stopped cleanly\n";
      }
    }
  }

  remove_dir(output_dir);
  return rc;
}
