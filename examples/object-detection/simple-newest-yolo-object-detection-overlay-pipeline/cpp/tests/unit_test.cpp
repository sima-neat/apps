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

  // Helper: each optional flag should be recognized (not rejected as unknown).
  // With a bad input-dir we expect exit 2, NOT exit 1 (which would mean arg-parse failed).
  auto expect_flag_recognized = [&](const std::string& label,
                                    std::vector<std::string> extra_args) {
    std::vector<std::string> argv_vec = {
        "--model", "m.tar.gz", "--labels", "l.txt",
        "--input-dir", "/nonexistent_dir_abc", "--output-dir", "/tmp/out"};
    argv_vec.insert(argv_vec.end(), extra_args.begin(), extra_args.end());
    auto r = spawn_and_wait(binary, argv_vec, 10000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] " << label << ": expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] " << label << " flag recognized\n";
    }
  };

  // Helper: invalid value for a validated flag should exit 1 (not silently accepted).
  auto expect_flag_rejected = [&](const std::string& label,
                                  std::vector<std::string> extra_args) {
    std::vector<std::string> argv_vec = {
        "--model", "m.tar.gz", "--labels", "l.txt",
        "--input-dir", "/tmp", "--output-dir", "/tmp/out"};
    argv_vec.insert(argv_vec.end(), extra_args.begin(), extra_args.end());
    auto r = spawn_and_wait(binary, argv_vec, 10000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] " << label << ": expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] " << label << " rejected\n";
    }
  };

  // Tests 5-9: smoke tests that each optional flag is recognized.
  expect_flag_recognized("--profile", {"--profile"});
  expect_flag_recognized("--no-overlay", {"--no-overlay"});
  expect_flag_recognized("--num-runs", {"--num-runs", "3"});
  expect_flag_recognized("--min-score", {"--min-score", "0.3"});
  expect_flag_recognized("--nms-iou", {"--nms-iou", "0.5"});

  // Tests 10-14: invalid values must be rejected.
  expect_flag_rejected("--num-runs 0", {"--num-runs", "0"});
  expect_flag_rejected("--num-runs -1", {"--num-runs", "-1"});
  expect_flag_rejected("--min-score 2.0", {"--min-score", "2.0"});
  expect_flag_rejected("--min-score -0.1", {"--min-score", "-0.1"});
  expect_flag_rejected("--nms-iou 1.5", {"--nms-iou", "1.5"});

  return failures > 0 ? 1 : 0;
}
