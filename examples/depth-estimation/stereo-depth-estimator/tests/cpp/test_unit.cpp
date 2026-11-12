// Unit test for stereo-depth-estimator: CLI handling and the configuration rules the binary
// checks before it loads the model or opens a camera. Each case exits with code 2 and names the
// key.
#include "support/testing/test_checks.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::example_common_config;
using sima_examples::testing::example_common_config_path;
using sima_examples::testing::expect_contains;
using sima_examples::testing::expect_true;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_scratch_config;

namespace {

constexpr const char* kExampleName = "stereo-depth-estimator";

// A config the validator accepts, with the caller's source section and `rest` appended. The model
// file does not exist, so a config that passes validation stops at the model check.
std::string config_with(const std::string& source, const std::string& rest = "") {
  return "model:\n"
         "  path: models/lightstereo.tar.gz\n"
         "source:\n" +
         source + rest +
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
}

const std::string kRtsp = "  type: rtsp\n"
                          "  rtsp:\n"
                          "    left_url: rtsp://127.0.0.1:8554/src1\n"
                          "    right_url: rtsp://127.0.0.1:8554/src2\n";

struct ConfigCase {
  std::string name;
  std::string body;
  std::string message;
};

bool expect_config_error(const std::string& binary, const ConfigCase& config_case) {
  const auto path = write_scratch_config(kExampleName, config_case.name, config_case.body);
  const auto result = spawn_and_wait(binary, {"--config", path.string()}, 20000);
  remove_dir(path.parent_path().string());
  return expect_true(result.exit_code == 2, config_case.name + " exits with code 2") &&
         expect_contains(result.stderr_text, "Error: " + config_case.message,
                         config_case.name + " reports: " + config_case.message);
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  bool ok = true;

  const auto help = spawn_and_wait(binary, {"--help"}, 20000);
  ok &= expect_true(help.exit_code == 0, "--help exits with code 0") &&
        expect_contains(help.stdout_text, "Usage", "--help prints usage");

  const auto unknown = spawn_and_wait(binary, {"--bogus"}, 20000);
  ok &= expect_true(unknown.exit_code == 2, "unknown argument exits with code 2") &&
        expect_contains(unknown.stderr_text, "unknown argument", "unknown argument is named");

  const auto missing_value = spawn_and_wait(binary, {"--config"}, 20000);
  ok &= expect_true(missing_value.exit_code == 2, "--config without a path exits with code 2") &&
        expect_contains(missing_value.stderr_text, "--config requires a path",
                        "--config without a path is reported");

  const auto missing_config = spawn_and_wait(binary, {"--config", "/does/not/exist.yaml"}, 20000);
  ok &= expect_true(missing_config.exit_code == 2, "missing config exits with code 2") &&
        expect_contains(missing_config.stderr_text, "Error:", "missing config is reported");

  // The shipped config parses; its model is prepared separately, so the model check stops it.
  const auto shipped = spawn_and_wait(
      binary, {"--config", example_common_config_path(kExampleName).string()}, 20000);
  ok &= expect_true(shipped.exit_code == 2, "shipped config without a model exits with code 2") &&
        expect_contains(shipped.stderr_text,
                        "Error: model does not exist: " +
                            example_common_config(kExampleName).string_or("model.path", ""),
                        "shipped config names its missing model");

  const std::vector<ConfigCase> cases = {
      {"valid_config_missing_model", config_with(kRtsp),
       "model does not exist: models/lightstereo.tar.gz"},
      {"unknown_source", config_with("  type: file\n"), "source.type must be rtsp, usb or mipi"},
      {"unknown_codec", config_with(kRtsp + "    codec: vp9\n"),
       "source.rtsp.codec must be h264, h265 or mjpeg"},
      {"missing_right_url",
       config_with("  type: rtsp\n"
                   "  rtsp:\n"
                   "    left_url: rtsp://127.0.0.1:8554/src1\n"),
       "source.rtsp.left_url and source.rtsp.right_url must be set"},
      {"same_camera_twice",
       config_with("  type: usb\n"
                   "  usb:\n"
                   "    left_camera: cam0\n"
                   "    right_camera: cam0\n"
                   "    width: 1280\n"
                   "    height: 720\n"
                   "    fps: 30\n"),
       "source.usb.left_camera and source.usb.right_camera must differ"},
      {"missing_mipi_mode",
       config_with("  type: mipi\n"
                   "  mipi:\n"
                   "    left_camera: cam0\n"
                   "    right_camera: cam1\n"),
       "source.mipi.width, source.mipi.height and source.mipi.fps must be > 0"},
      {"negative_latency", config_with(kRtsp + "    latency_ms: -1\n"),
       "source.rtsp.latency_ms must be >= 0"},
      {"negative_frames", config_with(kRtsp, "inference:\n  frames: -1\n"),
       "inference.frames must be >= 0"},
      // The right camera and the depth map take the next two ports.
      {"video_port_out_of_range", config_with(kRtsp) + "    video_port: 65534\n",
       "output.insight.video_port must be in [1, 65533]"},
      {"no_disparity_range", config_with(kRtsp) + "  max_disparity: 0\n",
       "output.max_disparity must be > 0"},
  };
  for (const auto& config_case : cases) {
    ok &= expect_config_error(binary, config_case);
  }
  return ok ? 0 : 1;
}
