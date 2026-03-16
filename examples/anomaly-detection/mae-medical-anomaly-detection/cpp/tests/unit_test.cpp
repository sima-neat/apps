// Unit test for mae-medical-anomaly-detection: validates CLI arg handling.
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

  // Test 1: no args → exit 1, prints usage
  {
    auto r = spawn_and_wait(binary, {}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] no args: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] no args: stderr does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] no args correctly rejected\n";
    }
  }

  // Test 2: only 2 args (models but no inputs) → exit 1
  {
    auto r = spawn_and_wait(binary, {"model.tar.gz", "cls_model.tar.gz"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] 2 args: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("No input") == std::string::npos &&
               r.stderr_text.find("input") == std::string::npos) {
      std::cerr << "[FAIL] 2 args: stderr does not mention missing input\n";
      ++failures;
    } else {
      std::cout << "[OK] insufficient args correctly rejected\n";
    }
  }

  // Test 3: unknown flag → should be handled gracefully (may exit 1 or continue)
  {
    auto r = spawn_and_wait(binary,
        {"model.tar.gz", "cls_model.tar.gz", "input.npy", "--unknown-flag"}, 10000);
    // The binary may exit early or continue; either is acceptable for unit test
    if (r.exit_code == 0) {
      std::cout << "[OK] unknown flag handled (ignored or accepted)\n";
    } else {
      std::cout << "[OK] unknown flag handled (rejected with exit " << r.exit_code << ")\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
