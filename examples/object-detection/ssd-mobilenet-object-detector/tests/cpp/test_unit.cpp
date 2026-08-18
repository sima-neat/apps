// Unit test for ssd-mobilenet-object-detector: validates CLI and decode configuration.
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  int failures = 0;

  {
    const ProcessResult result = spawn_and_wait(binary, {"--help"}, 20000);
    if (result.exit_code != 0 || result.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] --help did not print usage\n";
      ++failures;
    }
  }

  {
    const ProcessResult result = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (result.exit_code != 2 || result.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown argument was not rejected\n";
      ++failures;
    }
  }

  {
    const ProcessResult result = spawn_and_wait(binary, {"--config"}, 20000);
    if (result.exit_code != 2 ||
        result.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] missing config path was not rejected\n";
      ++failures;
    }
  }

  {
    const fs::path config = fs::temp_directory_path() / "ssd-blank-model.yaml";
    std::ofstream(config) << "model:\n  path: '   '\n";
    const ProcessResult result = spawn_and_wait(binary, {"--config", config.string()}, 20000);
    std::error_code error;
    fs::remove(config, error);
    if (result.exit_code != 2 ||
        result.stderr_text.find("model.path must be a nonempty path") == std::string::npos) {
      std::cerr << "[FAIL] blank model path was not rejected\n";
      ++failures;
    }
  }

  {
    const fs::path config = fs::temp_directory_path() / "ssd-missing-model.yaml";
    std::ofstream(config) << "io:\n  input_dir: /tmp\n";
    const ProcessResult result = spawn_and_wait(binary, {"--config", config.string()}, 20000);
    std::error_code error;
    fs::remove(config, error);
    if (result.exit_code != 2 ||
        result.stderr_text.find("model.path must be a nonempty path") == std::string::npos) {
      std::cerr << "[FAIL] missing model path was not rejected\n";
      ++failures;
    }
  }

  {
    const fs::path config = fs::temp_directory_path() / "ssd-invalid-threshold.yaml";
    std::ofstream(config) << "model:\n  path: /tmp/missing-model.tar.gz\n"
                             "decode:\n  score_threshold: 1.5\n";
    const ProcessResult result = spawn_and_wait(binary, {"--config", config.string()}, 20000);
    std::error_code error;
    fs::remove(config, error);
    if (result.exit_code != 2 ||
        result.stderr_text.find("decode.score_threshold") == std::string::npos) {
      std::cerr << "[FAIL] invalid score threshold was not rejected\n";
      ++failures;
    }
  }

  return failures == 0 ? 0 : 1;
}
