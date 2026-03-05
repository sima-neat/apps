// Unit test for multistream-rtsp-detection-pipeline: validates CLI arg handling.
// This example uses die() which throws std::runtime_error for missing args,
// caught by main() and printed to stderr. The exit code is nonzero.
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

  // Test 1: no args → nonzero exit, error mentions --model
  {
    auto r = spawn_and_wait(binary, {}, 10000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] no args: expected nonzero exit, got 0\n";
      ++failures;
    } else if (r.stderr_text.find("model") == std::string::npos &&
               r.stderr_text.find("--model") == std::string::npos) {
      std::cerr << "[FAIL] no args: stderr does not mention --model\n";
      ++failures;
    } else {
      std::cout << "[OK] no args correctly rejected\n";
    }
  }

  // Test 2: --model provided but no --output → nonzero exit
  {
    auto r = spawn_and_wait(binary, {"--model", "dummy.tar.gz"}, 10000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] no --output: expected nonzero exit, got 0\n";
      ++failures;
    } else if (r.stderr_text.find("output") == std::string::npos &&
               r.stderr_text.find("--output") == std::string::npos) {
      std::cerr << "[FAIL] no --output: stderr does not mention --output\n";
      ++failures;
    } else {
      std::cout << "[OK] missing --output correctly rejected\n";
    }
  }

  // Test 3: --model and --output but no --rtsp → nonzero exit
  {
    auto r = spawn_and_wait(binary,
        {"--model", "dummy.tar.gz", "--output", "/tmp/out"}, 10000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] no --rtsp: expected nonzero exit, got 0\n";
      ++failures;
    } else if (r.stderr_text.find("rtsp") == std::string::npos &&
               r.stderr_text.find("--rtsp") == std::string::npos) {
      std::cerr << "[FAIL] no --rtsp: stderr does not mention --rtsp\n";
      ++failures;
    } else {
      std::cout << "[OK] missing --rtsp correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
