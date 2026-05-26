// Unit test for single-stream-object-detector: validates CLI arg handling.
#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  {
    auto r = spawn_and_wait(binary, {"--help"}, 10000);
    if (r.exit_code != 0 || r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] help should pass and print usage\n";
      ++failures;
    }
  }

  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent/single-rtsp-config.yaml"}, 10000);
    if (r.exit_code == 0 || r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config should fail and mention config\n";
      ++failures;
    }
  }

  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 10000);
    if (r.exit_code == 0 || r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown flag should fail and mention unknown argument\n";
      ++failures;
    }
  }

  return failures > 0 ? 1 : 0;
}
