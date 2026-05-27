// Unit test for human-pose-estimator: validates CLI arg handling.
#include "support/testing/test_process.h"

#include <iostream>
#include <string>

using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  // Test 1: help should exit 0 and mention --config
  {
    auto r = spawn_and_wait(binary, {"--help"}, 10000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("--config") == std::string::npos) {
      std::cerr << "[FAIL] help: stdout does not mention --config\n";
      ++failures;
    } else {
      std::cout << "[OK] help exits with code 0\n";
    }
  }

  // Test 2: missing config file should exit 2
  {
    auto r = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] missing config: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config correctly rejected\n";
    }
  }

  // Test 3: unknown arg should exit 2
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] unknown flag: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
