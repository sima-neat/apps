// End-to-end tests: --calibrate builds the bank, the shipped images get the right verdicts, and
// each RTSP codec reaches Insight as video and defect-region metadata.
// tests/python/test_e2e.py runs the same cases against main.py.

#include "sima_examples/testing/metadata_json_listener.h"
#include "sima_examples/testing/source_cases.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "patchcore-anomaly-detector";
constexpr int kFrames = 60;
// Relative to the Apps root, which the tests run from.
const fs::path kDatasetDir = "assets/datasets/patchcore";

// A config whose bank is calibrated into `output_dir` from the shipped nominal set; `input`
// holds the input section's keys.
fs::path write_config(const std::string& output_dir, const std::string& model_path,
                      const std::string& input, int video_port, int metadata_port) {
  const fs::path config_path = fs::path(output_dir).parent_path() / "config.yaml";
  std::ofstream(config_path) << "model:\n  path: " << model_path << "\ninput:\n"
                             << input << "inference:\n  frames: " << kFrames
                             << "\n  max_detections: 10\noutput:\n  dir: " << output_dir
                             << "/outlined\n  insight:\n    host: 127.0.0.1\n"
                             << "    video_port_base: " << video_port
                             << "\n    metadata_port_base: " << metadata_port
                             << "\napp:\n  memory_bank_dir: " << output_dir << "/bank"
                             << "\n  nominal_images_dir: "
                             << fs::absolute(kDatasetDir / "nominal").string()
                             << "\n  coreset_ratio: 0.0004\n  threshold_percentile: 99.0"
                             << "\n  threshold_margin: 1.2\n";
  return config_path;
}

bool run_ok(const std::string& binary, const std::vector<std::string>& args,
            const std::string& label, ProcessResult& result) {
  const int timeout_ms = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  result = spawn_and_wait(binary, args, timeout_ms);
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] " << label << " exited with " << result.exit_code << "\nstdout:\n"
              << result.stdout_text << "\nstderr:\n"
              << result.stderr_text << "\n";
    return false;
  }
  return true;
}

// Calibrates, scores one folder of shipped images, and checks each verdict and outline.
int run_image_case(const std::string& binary, const std::string& model_path,
                   const std::string& folder, const std::set<std::string>& anomalous) {
  const std::string label = "image_dir " + folder;
  const std::string output_dir = create_test_output_dir(kExampleName, "test_image_" + folder);
  if (output_dir.empty()) {
    return 1;
  }
  const fs::path input_dir = fs::absolute(kDatasetDir / folder);
  const fs::path config_path =
      write_config(output_dir, model_path,
                   "  type: image_dir\n  source: " + input_dir.string() + "\n", 9000, 9100);
  ProcessResult result;
  bool ok = run_ok(binary, {"--config", config_path.string(), "--calibrate"}, label, result) &&
            run_ok(binary, {"--config", config_path.string()}, label, result);
  for (const auto& entry : ok ? fs::directory_iterator(input_dir) : fs::directory_iterator()) {
    const std::string name = entry.path().filename().string();
    const bool expected = anomalous.count(name) > 0;
    const std::regex verdict(name + ": " + (expected ? "ANOMALOUS" : "normal") + " score=");
    const cv::Mat source = cv::imread(entry.path().string());
    const cv::Mat outlined = cv::imread(output_dir + "/outlined/" + name);
    if (!std::regex_search(result.stdout_text, verdict) || outlined.empty() ||
        outlined.size() != source.size()) {
      std::cerr << "[FAIL] " << label << ": " << name << " is not "
                << (expected ? "ANOMALOUS" : "normal") << " with its image written\nstdout:\n"
                << result.stdout_text << "\n";
      ok = false;
      continue;
    }
    // Only an anomalous image is drawn on, and its outline covers a part of it.
    cv::Mat diff;
    cv::absdiff(source, outlined, diff);
    cv::Mat changed_mask;
    cv::transform(diff, changed_mask, cv::Matx13f(1, 1, 1));
    const double changed = cv::countNonZero(changed_mask) / static_cast<double>(source.total());
    if (expected ? !(changed > 0.0 && changed < 0.5) : changed != 0.0) {
      std::cerr << "[FAIL] " << label << ": " << name << " changed " << changed
                << " of its pixels\n";
      ok = false;
    }
  }
  remove_dir(output_dir);
  if (ok) {
    std::cout << "[OK] " << label << " verdicts and outlines\n";
  }
  return ok ? 0 : 1;
}

// Every segment is an anomaly with its box and outline in the frame and a confidence above 0.5;
// returns the segment count, or -1 for an invalid one.
int valid_segments(const std::vector<MetadataJsonMessage>& messages, int width, int height) {
  int count = 0;
  for (const auto& message : messages) {
    // A named value: a range-for over a member of a temporary would outlive the temporary.
    const auto payload = nlohmann::json::parse(message.payload);
    for (const auto& segment : payload.at("data").at("segments")) {
      const auto bbox = segment.at("bbox").get<std::array<int, 4>>();
      bool inside =
          bbox[0] >= 0 && bbox[1] >= 0 && bbox[0] + bbox[2] <= width && bbox[1] + bbox[3] <= height;
      for (const auto& point : segment.at("mask")) {
        const int x = point.at(0).get<int>();
        const int y = point.at(1).get<int>();
        inside &= x >= 0 && x < width && y >= 0 && y < height;
      }
      // A region lies above the patch threshold, so its confidence is above 0.5.
      const double confidence = segment.at("confidence").get<double>();
      if (segment.at("label") != "anomaly" || segment.at("mask_format") != "polygon" || !inside ||
          confidence <= 0.5 || confidence > 1.0) {
        std::cerr << "[FAIL] unexpected segment: " << segment.dump() << "\n";
        return -1;
      }
      ++count;
    }
  }
  return count;
}

// The first video packet's RTP header, or zeros when none arrives within 5 seconds.
std::array<unsigned char, 12> first_video_header(int socket) {
  const timeval timeout{5, 0};
  ::setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  std::array<unsigned char, 65536> packet{};
  std::array<unsigned char, 12> header{};
  if (::recv(socket, packet.data(), packet.size(), 0) > 12) {
    std::copy_n(packet.begin(), 12, header.begin());
  }
  return header;
}

int run_codec_case(const std::string& binary, const std::string& model_path,
                   const StreamSourceCase& source_case, const std::string& url,
                   const std::string& label) {
  const std::string output_dir = create_test_output_dir(kExampleName, "test_" + label);
  if (output_dir.empty()) {
    return 1;
  }
  // This test is the Insight receiver, so the example publishes to loopback.
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const int metadata_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100);
  const fs::path config_path =
      write_config(output_dir, model_path,
                   "  type: rtsp\n  source: " + url + "\n  codec: " + source_case.codec +
                       "\n  width: 0\n  height: 0\n  fps: 0\n  rtsp:\n    tcp: true\n"
                       "    latency_ms: 100\n",
                   video_port, metadata_port);
  ProcessResult result;
  if (!run_ok(binary, {"--config", config_path.string(), "--calibrate"}, label, result)) {
    remove_dir(output_dir);
    return 1;
  }

  MetadataJsonListenerOptions options;
  options.base_port = metadata_port;
  options.timeout_ms = 5000;
  options.require_all_ports = true;
  options.metadata_type = "segmentation";
  options.data_array_key = "segments";
  MetadataJsonListener listener(options);
  if (!listener.ok()) {
    std::cerr << "[FAIL] metadata listener failed: " << listener.error() << "\n";
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
    return 1;
  }
  const bool ran = run_ok(binary, {"--config", config_path.string()}, label, result);
  const auto header = first_video_header(video_socket);
  ::close(video_socket);
  remove_dir(output_dir);
  if (!ran) {
    return 1;
  }
  const auto metadata = listener.wait_for_messages();
  const auto rest = listener.drain_pending();
  std::smatch stream;
  std::smatch summary;
  if (!metadata.success ||
      !std::regex_search(result.stdout_text, stream, std::regex(R"(stream=(\d+)x(\d+)@)")) ||
      !std::regex_search(
          result.stdout_text, summary,
          std::regex("processed=" + std::to_string(kFrames) + R"( flagged=(\d+) )"))) {
    std::cerr << "[FAIL] " << label << " sent no segmentation metadata: " << metadata.error
              << "\nstdout:\n"
              << result.stdout_text << "\n";
    return 1;
  }
  const int width = std::stoi(stream[1]);
  const int height = std::stoi(stream[2]);
  const int first = valid_segments(metadata.messages, width, height);
  const int later = valid_segments(rest.messages, width, height);
  // The test streams show no part from the nominal set, so the model flags them.
  if (std::stoi(summary[1]) == 0 || first < 0 || later < 0 || first + later == 0) {
    std::cerr << "[FAIL] " << label << " sent no valid defect region\nstdout:\n"
              << result.stdout_text << "\n";
    return 1;
  }
  // VideoSender forwards the stream as received: RTP H.264 (96), H.265 (98), or JPEG (26).
  const int payload_type = source_case.codec == "h265"    ? 98
                           : source_case.codec == "mjpeg" ? 26
                                                          : 96;
  if ((header[0] >> 6) != 2 || (header[1] & 0x7F) != payload_type) {
    std::cerr << "[FAIL] " << label << " did not publish RTP video with payload type "
              << payload_type << " to Insight\n";
    return 1;
  }
  std::cout << "[OK] " << label << " sent " << kFrames << " frames of regions and Insight video\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  // Absolute, because the example resolves a relative model.path from its config's folder.
  const std::string models_dir =
      fs::absolute(models_dir_raw != nullptr ? models_dir_raw : "models").string();
  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("configured PatchCore model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  int images_rc = 0;
  for (const std::string& model_path : model_paths) {
    images_rc |=
        run_image_case(binary, model_path, "images",
                       {"scratch_0.png", "scratch_1.png", "scratch_2.png", "scratch_3.png"});
    // Known-good images the bank never saw score below the threshold.
    images_rc |= run_image_case(binary, model_path, "held_out_normal", {});
  }

  const int cases_rc = run_single_stream_source_cases(
      "PatchCore anomaly detector",
      {
          {"rtsp_h264", "SIMANEAT_TEST_RTSP_H264_URL", "rtsp", "h264", 0, true},
          {"rtsp_h265", "SIMANEAT_TEST_RTSP_H265_URL", "rtsp", "h265", 0, true},
          {"rtsp_mjpeg", "SIMANEAT_TEST_RTSP_MJPEG_URL", "rtsp", "mjpeg", 0, true},
      },
      [&](const StreamSourceCase& source_case, const std::string& url) {
        int case_rc = 0;
        for (const std::string& model_path : model_paths) {
          const std::string label =
              model_case_label(source_case.name, model_path, model_paths.size());
          if (run_codec_case(binary, model_path, source_case, url, label) != 0) {
            case_rc = 1;
          }
        }
        return case_rc;
      });
  return rc != 0 || images_rc != 0 ? 1 : cases_rc;
}
