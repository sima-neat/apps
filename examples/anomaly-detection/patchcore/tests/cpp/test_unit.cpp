// Unit test for patchcore (C++): validates CLI arg handling.
//
// The scoring stage's math (nearest-neighbor distance, the PatchCore
// neighborhood-reweighting term, greedy k-center coreset selection, threshold
// derivation, and bank/model hash pinning) is covered against fixed embeddings
// in tests/python/test_unit.py -- the C++ and Python implementations share the
// same algorithm (see support/anomaly_detection/patchcore_memory_bank.h), and
// this repo's C++ unit-test target only links against support_testing, not
// support_runtime, so it cannot call into that library directly. This test is
// scoped to what test_unit.cpp binaries in this repo test elsewhere: the CLI
// surface.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
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

  // Test 1: help exits successfully and prints usage.
  {
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] help: stdout does not contain Usage\n";
      ++failures;
    } else {
      std::cout << "[OK] help prints usage\n";
    }
  }

  // Test 2: unknown flag is rejected before model/runtime startup.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] unknown flag: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown flag: stderr does not explain the failure\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  // Test 3: missing config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config"}, 20000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] missing config path: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] missing config path: stderr does not explain the failure\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config path correctly rejected\n";
    }
  }

  // Test 4: nonexistent config path is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent/patchcore-config.yaml"}, 20000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] bad config: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config: stderr does not explain the failure\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path correctly rejected\n";
    }
  }

  // Test 5: --validate-config-only accepts the shipped default config.
  {
    auto r = spawn_and_wait(binary, {"--validate-config-only"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] validate-config-only: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else {
      std::cout << "[OK] --validate-config-only accepts the shipped default config\n";
    }
  }

  // Test 6: memory bank missing (a valid config, but --calibrate was never run)
  // is rejected with a clear, actionable message rather than a crash.
  {
    const std::string out_dir =
        sima_examples::testing::create_test_output_dir("patchcore", "test_missing_bank_rejected");
    if (out_dir.empty()) {
      ++failures;
    } else {
      const std::filesystem::path run_dir = std::filesystem::path(out_dir).parent_path();
      const std::filesystem::path config_path = run_dir / "config.yaml";
      sima_examples::testing::write_e2e_config(
          "patchcore", config_path,
          {
              {"model.path", "dummy-model.tar.gz"},
              {"memory_bank.path", (run_dir / "memory_bank.npy").string()},
              {"memory_bank.meta_path", (run_dir / "bank_meta.json").string()},
          });

      auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
      if (r.exit_code != 2) {
        std::cerr << "[FAIL] missing bank: expected exit 2, got " << r.exit_code << "\n";
        ++failures;
      } else if (r.stderr_text.find("memory bank not found") == std::string::npos) {
        std::cerr << "[FAIL] missing bank: stderr does not explain the failure\n";
        ++failures;
      } else {
        std::cout << "[OK] missing memory bank correctly rejected\n";
      }
      sima_examples::testing::remove_dir(out_dir);
    }
  }

  return failures > 0 ? 1 : 0;
}
