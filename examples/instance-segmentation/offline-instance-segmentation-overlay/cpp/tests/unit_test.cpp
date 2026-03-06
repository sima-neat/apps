// Unit test for offline-instance-segmentation-overlay: validates CLI arg handling.
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

  // Test 2: only 1 arg → exit 1
  {
    auto r = spawn_and_wait(binary, {"dummy_model.tar.gz"}, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] 1 arg: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] insufficient args correctly rejected\n";
    }
  }

  // Test 3: bad input dir → nonzero exit
  {
    auto r = spawn_and_wait(binary, {"dummy.tar.gz", "/nonexistent_dir_abc", "/tmp/out"}, 10000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] bad input dir: expected nonzero exit, got 0\n";
      ++failures;
    } else {
      std::cout << "[OK] bad input dir correctly rejected (exit " << r.exit_code << ")\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
