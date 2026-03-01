// Unit test for live-rtsp-depth-estimation: validates CLI arg handling.
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

  // Test 1: no --url and no --self-test → exit 2
  {
    auto r = spawn_and_wait(binary, {}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] no url/self-test: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] no url/self-test correctly rejected\n";
    }
  }

  // Test 2: --url provided but no --model → exit 3
  {
    auto r = spawn_and_wait(binary, {"--url", "rtsp://dummy:8554/stream"}, 10000);
    if (r.exit_code != 3) {
      std::cerr << "[FAIL] no --model: expected exit 3, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("model") == std::string::npos &&
               r.stderr_text.find("Model") == std::string::npos) {
      std::cerr << "[FAIL] no --model: stderr does not mention model\n";
      ++failures;
    } else {
      std::cout << "[OK] missing --model correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
