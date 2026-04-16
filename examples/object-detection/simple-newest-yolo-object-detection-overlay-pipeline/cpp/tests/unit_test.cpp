// Unit test for simple-newest-yolo-object-detection-overlay-pipeline: validates CLI arg handling.
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

  // Test 2: partial flags (missing --input-dir, --output-dir) → exit 1
  {
    auto r = spawn_and_wait(binary,
        {"--model", "model.tar.gz", "--labels", "labels.txt"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] partial flags: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] partial flags correctly rejected\n";
    }
  }

  // Test 3: all flags but bad input dir → exit 2
  {
    auto r = spawn_and_wait(binary,
        {"--model", "model.tar.gz", "--labels", "labels.txt",
         "--input-dir", "/nonexistent_dir_abc", "--output-dir", "/tmp/out"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] bad input dir: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] bad input dir correctly rejected\n";
    }
  }

  // Test 4: unknown flag → exit 1
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] unknown flag: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  // Test 5: --profile with bad input dir → exit 2 (validates --profile is recognized)
  {
    auto r = spawn_and_wait(binary,
        {"--model", "m.tar.gz", "--labels", "l.txt",
         "--input-dir", "/nonexistent_dir_abc", "--output-dir", "/tmp/out",
         "--profile"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] profile+bad dir: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] --profile flag recognized, bad input dir correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
