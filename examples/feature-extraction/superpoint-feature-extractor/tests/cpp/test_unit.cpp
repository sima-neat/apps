#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

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
