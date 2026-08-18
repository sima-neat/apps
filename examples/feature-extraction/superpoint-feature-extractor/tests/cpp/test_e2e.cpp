#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <regex>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

bool has_valid_summary(const std::string& text) {
  const std::regex summary(R"(frames=8 average_points=([0-9]+(?:\.[0-9]+)?) descriptor_dim=256 )");
  std::smatch match;
  return std::regex_search(text, match, summary) && std::stod(match[1].str()) > 0.0 &&
         std::stod(match[1].str()) <= 600.0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";
  const fs::path model = configured_model_path("superpoint-feature-extractor", models_dir);
  const fs::path input = "assets/datasets/tum-rgbd/freiburg1-desk.mp4";
  if (!fs::is_regular_file(model)) {
    return skip_or_fail("qualified SuperPoint model not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }
  if (!fs::is_regular_file(input)) {
    return skip_or_fail("TUM RGB-D SuperPoint input video is missing");
  }

  const std::string insight_host = env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       ? env_or_null("SIMANEAT_APPS_TEST_INSIGHT_HOST")
                                       : "127.0.0.1";
  const int video_port = env_int_or_default("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000);
  const fs::path test_dir = create_test_output_dir("superpoint-feature-extractor", "pipeline");
  const fs::path config = test_dir / "config.yaml";
  write_e2e_config("superpoint-feature-extractor", config,
                   {{"model.path", fs::absolute(model).string()},
                    {"io.input", fs::absolute(input).string()},
                    {"output.insight.host", insight_host},
                    {"output.insight.video_port", std::to_string(video_port)},
                    {"output.insight.channel", "0"},
                    {"runtime.frames", "8"}});

  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);
  const auto result = spawn_and_wait(argv[1], {"--config", config.string()}, timeout);

  int rc = 0;
  if (result.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << result.exit_code << "\n" << result.stderr_text << "\n";
    rc = 1;
  } else if (!has_valid_summary(result.stdout_text)) {
    std::cerr << "[FAIL] missing or invalid feature summary\n" << result.stdout_text << "\n";
    rc = 1;
  } else if (result.stdout_text.find("video_sender=" + insight_host + ":" +
                                     std::to_string(video_port)) == std::string::npos) {
    std::cerr << "[FAIL] missing Insight video sender summary\n" << result.stdout_text << "\n";
    rc = 1;
  }

  remove_dir(test_dir.string());
  return rc;
}
