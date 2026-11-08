#include "examples/feature-extraction/superpoint-feature-extractor/src/cpp/feature_points.h"
#include "support/testing/test_process.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::spawn_and_wait;
using superpoint_app::FrameSize;
using superpoint_app::source_points;

namespace {

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
  }
  return condition;
}

int source_point_failures() {
  const FrameSize model{640, 480};
  const FrameSize source{1280, 720};
  int failures = 0;
  failures +=
      !expect(source_points({600.0F, 400.0F}, model, source) == std::vector<float>{1200.0F, 600.0F},
              "model pixels scale per axis onto the source frame");
  failures +=
      !expect(source_points({10.0F, 20.0F}, model, model) == std::vector<float>{10.0F, 20.0F},
              "a model-sized source frame keeps the decoder coordinates");
  failures += !expect(source_points({600.0F, 400.0F, 640.0F, 200.0F, 20.0F, 480.0F, NAN, 10.0F},
                                    model, source) == std::vector<float>{1200.0F, 600.0F},
                      "points on or beyond a source edge and nonfinite points are dropped");
  return failures;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = source_point_failures();

  const auto help = spawn_and_wait(binary, {"--help"}, 20000);
  if (help.exit_code != 0 || help.stdout_text.find("Usage") == std::string::npos) {
    std::cerr << "[FAIL] --help\n";
    ++failures;
  }

  const auto unknown = spawn_and_wait(binary, {"--bogus"}, 20000);
  if (unknown.exit_code == 0 || unknown.stderr_text.find("unknown argument") == std::string::npos) {
    std::cerr << "[FAIL] unknown argument\n";
    ++failures;
  }

  const auto missing_value = spawn_and_wait(binary, {"--config"}, 20000);
  if (missing_value.exit_code == 0 ||
      missing_value.stderr_text.find("--config requires a path") == std::string::npos) {
    std::cerr << "[FAIL] missing --config value\n";
    ++failures;
  }

  const auto missing_config = spawn_and_wait(binary, {"--config", "/does/not/exist.yaml"}, 20000);
  if (missing_config.exit_code == 0 ||
      missing_config.stderr_text.find("Error:") == std::string::npos) {
    std::cerr << "[FAIL] missing config\n";
    ++failures;
  }

  return failures == 0 ? 0 : 1;
}
