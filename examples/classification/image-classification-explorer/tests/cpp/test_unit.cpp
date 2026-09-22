// Unit test for image-classification-explorer: validates CLI arg handling.
#include "support/testing/test_process.h"

#include <chrono>
#include <filesystem>
#include <fstream>
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
    auto r = spawn_and_wait(binary, {"--help"}, 20000);
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

  // Test 2: a missing config file produces a nonzero exit.
  {
    auto r = spawn_and_wait(binary, {"--config", "/nonexistent/config.yaml"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] missing config: expected nonzero exit, got 0\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config produced a nonzero exit\n";
    }
  }

  // Test 3: an unrecognized flag produces a nonzero exit.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] unknown flag: expected nonzero exit, got 0\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag produced a nonzero exit\n";
    }
  }

  // Test 4: an empty profile must be rejected instead of silently ignored.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-empty-profile-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  valid:\n"
             << "    path: valid-model.tar.gz\n"
             << "  bad: {}\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.exit_code == 0 ||
        r.stderr_text.find("models.bad.path is required") == std::string::npos) {
      std::cerr << "[FAIL] empty profile: expected required-path error, got exit " << r.exit_code
                << "\n";
      ++failures;
    } else {
      std::cout << "[OK] empty model profile was rejected\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
