// Unit test for retinaface-face-detection: validates CLI arg handling.
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

  // Test 1: no args -> exit 2, prints usage
  {
    ProcessResult r = spawn_and_wait(binary, {}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] no args: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("Usage") == std::string::npos &&
               r.stderr_text.find("retinaface-face-detection <image_path>") == std::string::npos) {
      std::cerr << "[FAIL] no args: stderr does not contain usage hint\n";
      ++failures;
    } else {
      std::cout << "[OK] no args correctly rejected\n";
    }
  }

  // Test 2: non-existent image path -> exit 2, mentions image
  {
    ProcessResult r = spawn_and_wait(binary, {"/path/does/not/exist.png"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] bad image path: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("image does not exist") == std::string::npos) {
      std::cerr << "[FAIL] bad image path: stderr does not mention missing image\n";
      ++failures;
    } else {
      std::cout << "[OK] bad image path correctly rejected\n";
    }
  }

  // Test 3: existing image but missing model file -> exit 2, mentions model
  // We don't require a real image; the binary only checks fs::exists before reading.
  {
    // Use /tmp as a stand-in "image" that exists on most systems; adjust expectations to match error path.
    ProcessResult r = spawn_and_wait(binary, {"/tmp", "--model", "does_not_exist.tar.gz"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] missing model file: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("model does not exist") == std::string::npos &&
               r.stderr_text.find("model") == std::string::npos) {
      std::cerr << "[FAIL] missing model file: stderr does not mention model\n";
      ++failures;
    } else {
      std::cout << "[OK] missing model file correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}

