// Unit test for single-stream-object-detector: CLI handling, the ffprobe
// command builder, the configuration rules --validate-config-only can
// reach without a model or a stream, and the NV16 saved-frame conversion.
#include "support/runtime/example_utils.h"
#include "support/runtime/ffprobe_command.h"
#include "support/runtime/pull_status.h"
#include "support/testing/test_checks.h"
#include "support/testing/test_process.h"

#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::build_ffprobe_rtsp_stream_info_command;
using sima_examples::nv16_to_bgr;
using sima_examples::pull_status_has_sample;
using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_not_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::validate_config_body;

namespace {

constexpr const char* kExampleName = "single-stream-object-detector";

// The smallest config the validator accepts, with the source and model lines
// supplied by the caller so each test states exactly the keys it is about.
// Everything else takes its documented default.
std::string minimal_config(const std::string& source_lines, const std::string& model_lines = "") {
  return "model:\n"
         "  path: models/model.tar.gz\n" +
         model_lines + "source:\n" + source_lines +
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
}

bool test_ffprobe_tcp_probe_selects_tcp_once_before_the_url() {
  const std::string command = build_ffprobe_rtsp_stream_info_command("rtsp://camera/live", true);
  const auto first = command.find("-rtsp_transport tcp");
  return expect_true(first != std::string::npos && first == command.rfind("-rtsp_transport tcp") &&
                         first < command.find("'rtsp://camera/live'"),
                     "TCP RTSP probe selects TCP exactly once before the URL");
}

bool test_ffprobe_default_probe_omits_tcp_and_quotes_the_url() {
  const std::string command =
      build_ffprobe_rtsp_stream_info_command("rtsp://camera/stream's", false);
  return expect_true(command.find("-rtsp_transport") == std::string::npos &&
                         command.find("'rtsp://camera/stream'\\''s'") != std::string::npos,
                     "default RTSP probe omits TCP and shell-quotes the URL");
}

bool test_help_runs(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(r.exit_code == 0, "help exits with code 0") &&
         expect_contains(r.stdout_text, "Usage", "help prints usage");
}

bool test_missing_config_file_fails(const std::string& binary) {
  const auto r =
      spawn_and_wait(binary, {"--config", "/nonexistent/single-rtsp-config.yaml"}, 20000);
  return expect_true(r.exit_code != 0, "missing config exits non-zero") &&
         expect_contains(r.stderr_text, "failed to open config", "missing config names the config");
}

bool test_unknown_flag_fails(const std::string& binary) {
  const auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
  return expect_true(r.exit_code != 0, "unknown flag exits non-zero") &&
         expect_contains(r.stderr_text, "unknown argument", "unknown flag names the argument");
}

// Only the section source.type names is read. The validated line reports its key, never the URL,
// which can carry credentials.
bool test_rtsp_section_supplies_the_url(const std::string& binary) {
  const auto r =
      validate_config_body(kExampleName, binary, "rtsp_section_url",
                           minimal_config("  rtsp:\n"
                                          "    url: rtsp://user:secret@127.0.0.1:8554/src1\n"));
  return expect_true(r.exit_code == 0, "source.rtsp.url validates") &&
         expect_contains(r.stdout_text, "source=source.rtsp.url",
                         "rtsp source reports source.rtsp.url") &&
         expect_not_contains(r.stdout_text, "rtsp://", "validated line does not echo the URL");
}

bool test_flat_source_url_is_not_read(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "flat_source_url",
                                      minimal_config("  url: rtsp://127.0.0.1:8554/src1\n"));
  return expect_true(r.exit_code != 0, "a flat source.url is not read") &&
         expect_contains(r.stderr_text, "source.rtsp.url must be set",
                         "missing url names source.rtsp.url");
}

bool test_http_source_reads_its_own_section(const std::string& binary) {
  const auto r = validate_config_body(kExampleName, binary, "http_section",
                                      minimal_config("  type: http\n"
                                                     "  rtsp:\n"
                                                     "    url: rtsp://127.0.0.1:8554/src1\n"));
  return expect_true(r.exit_code != 0, "http source does not read the rtsp section") &&
         expect_contains(r.stderr_text, "source.http.url must be set",
                         "missing url names source.http.url");
}

// The Python loader once accepted this because Path("") is "."; the C++ side
// is held to the rule so the two implementations cannot drift apart.
bool test_empty_labels_is_rejected(const std::string& binary) {
  const auto r = validate_config_body(
      kExampleName, binary, "empty_labels",
      minimal_config("  rtsp:\n    url: rtsp://127.0.0.1:8554/src1\n", "  labels: \"\"\n"));
  return expect_true(r.exit_code != 0, "empty model.labels is rejected") &&
         expect_contains(r.stderr_text, "model.labels must be set",
                         "empty labels error names the key");
}

// source.type=usb names a camera instead of a URL; the validated line reports the key, not the ID.
bool test_usb_source_validates_without_a_url(const std::string& binary) {
  const auto r =
      validate_config_body(kExampleName, binary, "usb_source",
                           minimal_config("  type: usb\n"
                                          "  usb:\n"
                                          "    camera: /base/pci@7000000000-3.1:1.0-046d:08e5\n"
                                          "    width: 1280\n"
                                          "    height: 720\n"
                                          "    fps: 30\n"));
  return expect_true(r.exit_code == 0, "usb source with a camera validates") &&
         expect_contains(r.stdout_text, "source=source.usb.camera",
                         "usb source reports source.usb.camera");
}

bool test_usb_source_requires_camera_and_fps(const std::string& binary) {
  const auto no_camera = validate_config_body(kExampleName, binary, "usb_no_camera",
                                              minimal_config("  type: usb\n"
                                                             "  usb:\n"
                                                             "    fps: 30\n"));
  const auto no_fps = validate_config_body(kExampleName, binary, "usb_no_fps",
                                           minimal_config("  type: usb\n"
                                                          "  usb:\n"
                                                          "    camera: cam0\n"
                                                          "    fps: 0\n"));
  return expect_true(no_camera.exit_code != 0, "usb source without a camera is rejected") &&
         expect_contains(no_camera.stderr_text, "source.usb.camera must be set",
                         "missing camera names source.usb.camera") &&
         expect_true(no_fps.exit_code != 0, "usb source without a frame rate is rejected") &&
         expect_contains(no_fps.stderr_text, "source.usb.fps must be > 0",
                         "missing fps names source.usb.fps");
}

// The camera's JPEG is forwarded as RTP/JPEG, which carries 8-pixel blocks up to 2040 per side.
bool test_usb_source_rejects_sizes_rtp_jpeg_cannot_carry(const std::string& binary) {
  bool ok = true;
  for (const std::string& size :
       {"    width: 3840\n    height: 2160\n", "    width: 1284\n    height: 720\n"}) {
    const auto r = validate_config_body(kExampleName, binary, "usb_rtp_jpeg_size",
                                        minimal_config("  type: usb\n"
                                                       "  usb:\n"
                                                       "    camera: cam0\n"
                                                       "    fps: 30\n" +
                                                       size));
    ok &= expect_true(r.exit_code != 0, "usb size RTP/JPEG cannot carry is rejected") &&
          expect_contains(r.stderr_text, "multiples of 8 and at most 2040",
                          "rejected size names the RTP/JPEG limit");
  }
  return ok;
}

bool test_closed_output_is_terminal() {
  using simaai::neat::PullStatus;
  simaai::neat::PullError pull_error;
  pull_error.message = "queue torn down";
  simaai::neat::PullError end_of_stream;
  end_of_stream.message = "input source reached end of stream";
  const auto thrown_message = [](PullStatus status, const simaai::neat::PullError& detail,
                                 const std::string& run_error) -> std::string {
    try {
      (void)pull_status_has_sample(status, "detections", detail, run_error);
    } catch (const std::runtime_error& error) {
      return error.what();
    }
    return "";
  };
  return expect_true(thrown_message(PullStatus::Closed, pull_error, "source reached EOS") ==
                         "detections output closed unexpectedly: source reached EOS",
                     "closed output ends the run with the runtime's reason") &&
         expect_true(thrown_message(PullStatus::Closed, end_of_stream, "") ==
                         "detections output closed unexpectedly: input source reached end of "
                         "stream",
                     "closed output without a run error keeps the pull's reason") &&
         expect_true(thrown_message(PullStatus::Error, pull_error, "source reached EOS") ==
                         "failed to pull detections: queue torn down",
                     "pull error ends the run with its message") &&
         expect_true(!pull_status_has_sample(PullStatus::Timeout, "detections", pull_error, ""),
                     "timeout is not a sample") &&
         expect_true(pull_status_has_sample(PullStatus::Ok, "detections", pull_error, ""),
                     "successful pull is a sample");
}

// A 4:2:2 frame whose chroma changes on every row, in padded planes: each row must keep its own
// chroma, which a 4:2:0 conversion would drop.
bool test_nv16_frame_keeps_each_rows_chroma() {
  constexpr int kWidth = 8;
  constexpr int kHeight = 4;
  constexpr int kStride = 16;
  std::vector<uint8_t> bytes(2 * kHeight * kStride, 0);
  for (int row = 0; row < kHeight; ++row) {
    const uint8_t u = row % 2 ? 240 : 90;
    const uint8_t v = row % 2 ? 90 : 240;
    for (int x = 0; x < kWidth; ++x) {
      bytes[row * kStride + x] = 81;
      bytes[(kHeight + row) * kStride + x] = x % 2 ? v : u;
    }
  }
  simaai::neat::Tensor tensor = simaai::neat::Tensor::from_vector(
      bytes, {static_cast<int64_t>(bytes.size())}, simaai::neat::TensorMemory::CPU);
  tensor.semantic.image = simaai::neat::ImageSpec{simaai::neat::ImageSpec::PixelFormat::NV16, ""};
  tensor.planes = {
      {simaai::neat::PlaneRole::Y, {kHeight, kWidth}, {kStride, 1}, 0},
      {simaai::neat::PlaneRole::UV, {kHeight, kWidth}, {kStride, 1}, kHeight * kStride},
  };
  cv::Mat bgr;
  std::string err;
  if (!expect_true(nv16_to_bgr(tensor, bgr, err), "NV16 converts: " + err))
    return false;
  auto expected = [](uint8_t u, uint8_t v) {
    cv::Mat pair(1, 2, CV_8UC2);
    pair.at<cv::Vec2b>(0, 0) = cv::Vec2b(81, u);
    pair.at<cv::Vec2b>(0, 1) = cv::Vec2b(81, v);
    cv::Mat out;
    cv::cvtColor(pair, out, cv::COLOR_YUV2BGR_YUY2);
    return out.at<cv::Vec3b>(0, 0);
  };
  bool ok = expect_true(bgr.rows == kHeight && bgr.cols == kWidth, "NV16 keeps its size");
  for (int row = 0; row < kHeight && ok; ++row) {
    const cv::Vec3b want = row % 2 ? expected(240, 90) : expected(90, 240);
    for (int x = 0; x < kWidth; ++x)
      ok &= bgr.at<cv::Vec3b>(row, x) == want;
  }
  return expect_true(ok, "every NV16 row keeps its own chroma") &&
         expect_true(expected(90, 240) != expected(240, 90), "the two rows differ in colour");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  bool ok = true;
  ok &= test_ffprobe_tcp_probe_selects_tcp_once_before_the_url();
  ok &= test_ffprobe_default_probe_omits_tcp_and_quotes_the_url();
  ok &= test_help_runs(binary);
  ok &= test_missing_config_file_fails(binary);
  ok &= test_unknown_flag_fails(binary);
  ok &= test_rtsp_section_supplies_the_url(binary);
  ok &= test_flat_source_url_is_not_read(binary);
  ok &= test_http_source_reads_its_own_section(binary);
  ok &= test_empty_labels_is_rejected(binary);
  ok &= test_usb_source_validates_without_a_url(binary);
  ok &= test_usb_source_requires_camera_and_fps(binary);
  ok &= test_usb_source_rejects_sizes_rtp_jpeg_cannot_carry(binary);
  ok &= test_closed_output_is_terminal();
  ok &= test_nv16_frame_keeps_each_rows_chroma();
  return ok ? 0 : 1;
}
