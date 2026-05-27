// Unit test for yolov8-object-detector: validates CLI arg handling.
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

  // Test 1: help exits successfully and prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 10000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] help prints usage\n";
    }
  }

  // Test 2: unknown flag is rejected before model/runtime startup.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] unknown flag: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown flag: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  // Test 3: missing config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] missing config path: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] missing config path: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config path correctly rejected\n";
    }
  }

  // Test 4: missing config file is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent/simple-det-config.yaml"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] bad config: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
