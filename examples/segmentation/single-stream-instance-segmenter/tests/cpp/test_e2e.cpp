// E2E test for single-stream-instance-segmenter.
// Runs every supported model family over every supported input path and checks the three
// outputs the application advertises: annotated frames, Insight segmentation metadata,
// and Insight video.
#include "support/testing/metadata_json_listener.h"
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <nlohmann/json.hpp>

#include <regex>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

const char* kExample = "single-stream-instance-segmenter";

struct ModelCase {
  std::string family;
  std::string model_path;
};

// The YOLOv8 package tests/test-scope.yaml downloads. The YOLO26 packages come from the
// scope's selection, so the nightly run also exercises the YOLO26 variants.
const char* kYoloV8ModelFile = "yolo_v8n_seg_mpk.tar.gz";

bool valid_metadata(const MetadataJsonListenerResult& result) {
  for (const auto& message : result.messages) {
    if (message.frame_id.empty() ||
        message.frame_id.find_first_not_of("0123456789") != std::string::npos ||
        message.timestamp_ms < 0) {
      return false;
    }
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& segment : payload.at("data").at("segments")) {
      if (!segment.contains("label") || segment.at("label").get<std::string>().empty() ||
          !segment.contains("confidence") || !segment.at("confidence").is_number() ||
          segment.at("confidence").get<double>() < 0.0 ||
          segment.at("confidence").get<double>() > 1.0 || !segment.contains("bbox") ||
          !segment.at("bbox").is_array() || segment.at("bbox").size() != 4U ||
          !segment.contains("mask_format") || segment.at("mask_format") != "polygon" ||
          !segment.contains("mask") || !segment.at("mask").is_array() ||
          segment.at("mask").size() < 3U) {
        return false;
      }
      for (const auto& value : segment.at("bbox")) {
        if (!value.is_number() || value.get<double>() < 0.0) {
          return false;
        }
      }
      for (const auto& point : segment.at("mask")) {
        if (!point.is_array() || point.size() != 2U || !point[0].is_number() ||
            !point[1].is_number() || point[0].get<double>() < 0.0 ||
            point[1].get<double>() < 0.0) {
          return false;
        }
      }
    }
  }
  return true;
}

/// Every result due a picture is accounted for: written, or reported as having lost the decoded
/// frame it needed. A source faster than the model legitimately produces some of the latter on
/// the host-decoded route, so the run is checked for complete accounting rather than a fixed
/// yield the platform cannot promise.
bool accounted_saves(const std::string& summary, int attempts, int files) {
  std::smatch match;
  const std::regex pattern(R"(saved=(\d+) unpaired=(\d+))");
  if (!std::regex_search(summary, match, pattern)) {
    return false;
  }
  const int saved = std::stoi(match[1].str());
  const int unpaired = std::stoi(match[2].str());
  return saved + unpaired == attempts && saved > 0 && files == saved;
}

int run_case(const std::string& binary, const ModelCase& model,
             const StreamSourceCase& source_case, const std::string& source_url,
             const std::string& case_name) {
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const int frames = e2e_int(kExample, "testing.e2e.inference", "frames");
  const int save_every = e2e_int(kExample, "testing.e2e.output", "save_every");
  const int attempts = save_every > 0 ? frames / save_every : 0;

  const std::string output_dir = create_test_output_dir(kExample, "test_" + case_name);
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  ConfigScalars overrides{{"model.family", model.family},
                          {"model.path", model.model_path},
                          {"source.type", source_case.type},
                          {"source.codec", source_case.codec},
                          {"source.url", source_url},
                          {"source.ssl_strict", source_case.ssl_strict ? "true" : "false"},
                          {"output.save_dir", output_dir},
                          // This test is the Insight receiver, so it publishes to loopback.
                          {"output.insight.host", "127.0.0.1"},
                          {"output.insight.video_port", std::to_string(video_port)},
                          {"output.insight.metadata_port", std::to_string(metadata_port)}};
  if (source_case.fps > 0) {
    overrides["source.fps"] = std::to_string(source_case.fps);
  }
  write_e2e_config(kExample, config_path, overrides);

  MetadataJsonListenerOptions listener_options;
  listener_options.host = "127.0.0.1";
  listener_options.base_port = metadata_port;
  listener_options.num_ports = 1;
  listener_options.timeout_ms = 5000;
  listener_options.metadata_type = "segmentation";
  listener_options.data_array_key = "segments";
  listener_options.require_all_ports = true;
  listener_options.min_object_count = 1;
  MetadataJsonListener listener(listener_options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
    remove_dir(output_dir);
    return 1;
  }

  const int video_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in video_address{};
  video_address.sin_family = AF_INET;
  video_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  video_address.sin_port = htons(static_cast<uint16_t>(video_port));
  if (video_socket < 0 || ::bind(video_socket, reinterpret_cast<sockaddr*>(&video_address),
                                 sizeof(video_address)) != 0) {
    if (video_socket >= 0) {
      ::close(video_socket);
    }
    std::cerr << "[FAIL] could not bind Insight video port\n";
    remove_dir(output_dir);
    return 1;
  }

  const ProcessResult process =
      spawn_and_wait(binary, {"--config", config_path.string()}, timeout_ms);
  std::array<unsigned char, 65536> video_packet{};
  const auto video_bytes =
      ::recv(video_socket, video_packet.data(), video_packet.size(), MSG_DONTWAIT);
  ::close(video_socket);

  int rc = 0;
  std::string frames_problem;
  if (process.exit_code != 0) {
    std::cerr << "[FAIL] " << case_name << " exited with " << process.exit_code
              << "\nstdout:\n"
              << process.stdout_text << "\nstderr:\n"
              << process.stderr_text << "\n";
    rc = 1;
  } else if (process.stdout_text.find("model=" + model.family) == std::string::npos ||
             process.stdout_text.find("processed=" + std::to_string(frames)) ==
                 std::string::npos) {
    std::cerr << "[FAIL] " << case_name << " did not run the configured family to completion\n"
              << process.stdout_text;
    rc = 1;
  } else if (!accounted_saves(process.stdout_text, attempts, count_output_files(output_dir))) {
    std::cerr << "[FAIL] " << case_name << " did not account for every annotated frame\n"
              << process.stdout_text;
    rc = 1;
  } else if (frames_problem = streamed_frames_problem(output_dir, count_output_files(output_dir));
             !frames_problem.empty()) {
    std::cerr << "[FAIL] " << case_name << " " << frames_problem << "\n";
    rc = 1;
  } else if (video_bytes <= 12 || (video_packet[0] >> 6) != 2 ||
             (video_packet[1] & 0x7F) != 96) {
    // VideoSender always re-encodes to RTP H.264 (payload type 96) for Insight.
    std::cerr << "[FAIL] " << case_name << " did not publish RTP H.264 video to Insight\n";
    rc = 1;
  } else {
    const auto metadata = listener.wait_for_messages();
    if (!metadata.success || !valid_metadata(metadata)) {
      std::cerr << "[FAIL] " << case_name << " metadata invalid: " << metadata.error << "\n";
      rc = 1;
    } else {
      std::cout << "[OK] " << case_name << " published " << count_output_files(output_dir)
                << " annotated frames, Insight video, and segmentation metadata\n";
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
  const std::string models_dir = models_dir_raw != nullptr ? models_dir_raw : "models";

  int rc = 0;
  std::vector<ModelCase> models;
  for (const std::string& path :
       available_model_paths(configured_model_paths(kExample, models_dir), rc)) {
    models.push_back({"yolo26", path});
  }
  const fs::path yolov8_path = fs::path(models_dir) / kYoloV8ModelFile;
  if (fs::exists(yolov8_path)) {
    models.push_back({"yolov8", yolov8_path.string()});
  } else {
    record_unavailable_case("missing yolov8 package: " + yolov8_path.string(),
                            "yolov8 package not found: " + yolov8_path.string(), rc);
  }
  if (models.empty()) {
    return skip_or_fail("no instance segmentation model found under "
                        "SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  // One case per source/codec combination the application accepts. The
  // validator only allows http with mjpeg, so that is the single http case.
  const std::vector<StreamSourceCase> source_cases = {
      {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
      {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
      {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 0, true},
      // HTTPS MJPEG sources commonly carry a self-signed certificate.
      {"http_mjpeg", "SIMANEAT_TEST_HTTP_MJPEG_URL", "http", "mjpeg", 30, false},
  };

  const int cases_rc = run_single_stream_source_cases(
      "single-stream instance segmenter", source_cases,
      [&](const StreamSourceCase& source_case, const std::string& source_url) {
        int case_rc = 0;
        for (const ModelCase& model : models) {
          const std::string case_name =
              model_case_label(model.family + "_" + source_case.name, model.model_path, 2);
          if (run_case(binary, model, source_case, source_url, case_name) != 0) {
            case_rc = 1;
          }
        }
        return case_rc;
      });
  return rc != 0 ? rc : cases_rc;
}
