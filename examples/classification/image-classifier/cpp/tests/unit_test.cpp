// Unit test for image-classifier: validates CLI arg handling.
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

  // Test 1: --help prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 10000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] --help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] --help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] --help printed usage\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
