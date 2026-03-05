// Unit test for single-rtsp-object-detection-optiview: validates CLI arg handling.
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
    auto r = spawn_and_wait(binary, {"--mpk", "dummy.tar.gz"}, 10000);
    if (r.exit_code == 0 || r.stderr_text.find("--rtsp") == std::string::npos) {
      std::cerr << "[FAIL] missing --rtsp should fail and mention --rtsp\n";
      ++failures;
    }
  }

  {
    auto r = spawn_and_wait(binary, {"--rtsp", "rtsp://127.0.0.1:8554/stream"}, 10000);
    if (r.exit_code == 0 || r.stderr_text.find("--mpk") == std::string::npos) {
      std::cerr << "[FAIL] missing --mpk should fail and mention --mpk\n";
      ++failures;
    }
  }

  return failures > 0 ? 1 : 0;
}
