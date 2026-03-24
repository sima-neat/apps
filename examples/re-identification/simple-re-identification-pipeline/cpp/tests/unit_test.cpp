// Unit test for simple-re-identification-pipeline: validates CLI arg handling.
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

  // Test 1: no args -> exit 2.
  {
    auto r = spawn_and_wait(binary, {}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] no args: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] no args correctly rejected\n";
    }
  }

  // Test 2: only one image -> exit 2.
  {
    auto r = spawn_and_wait(binary, {"some_image.jpg"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] one image: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] missing second image correctly rejected\n";
    }
  }

  // Test 3: non-existent explicit model path -> exit 2.
  {
    auto r = spawn_and_wait(
        binary,
        {"some_image.jpg", "other_image.jpg", "--model", "does_not_exist.tar.gz"},
        10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] bad model path: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("Model file does not exist") == std::string::npos) {
      std::cerr << "[FAIL] bad model path: stderr missing expected message\n";
      ++failures;
    } else {
      std::cout << "[OK] missing model path correctly rejected\n";
    }
  }

  // Test 4: invalid metric -> exit 2.
  {
    auto r = spawn_and_wait(
        binary,
        {"a.jpg", "b.jpg", "--metric", "manhattan"},
        10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] invalid metric: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] invalid metric correctly rejected\n";
    }
  }

  // Test 5: unknown flag -> exit 2.
  {
    auto r = spawn_and_wait(binary, {"a.jpg", "b.jpg", "--bogus"}, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] unknown flag: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
