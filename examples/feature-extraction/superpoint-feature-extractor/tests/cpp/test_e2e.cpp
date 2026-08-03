#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
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

bool is_h264_mp4(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  std::string bytes(128, '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  bytes.resize(static_cast<std::size_t>(input.gcount()));
  return bytes.find("ftyp") != std::string::npos && bytes.find("avc1") != std::string::npos;
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

  const fs::path output_dir = create_test_output_dir("superpoint-feature-extractor", "pipeline");
  const fs::path output = output_dir / "annotated.mp4";
  const fs::path config = output_dir.parent_path() / "config.yaml";
  write_e2e_config("superpoint-feature-extractor", config,
                   {{"model.path", fs::absolute(model).string()},
                    {"io.input", fs::absolute(input).string()},
                    {"io.output", fs::absolute(output).string()},
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
  } else if (!fs::is_regular_file(output) || fs::file_size(output) == 0) {
    std::cerr << "[FAIL] annotated H.264 video was not written\n";
    rc = 1;
  } else if (!is_h264_mp4(output)) {
    std::cerr << "[FAIL] output is not an H.264 MP4\n";
    rc = 1;
  }

  remove_dir(output_dir.string());
  return rc;
}
