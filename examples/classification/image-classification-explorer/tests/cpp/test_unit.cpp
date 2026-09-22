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

  // Test 5: profiles run in declaration order even when `models:` carries an
  // inline YAML comment (the order scanner must strip comments, not fall back
  // to alphabetical order).
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-model-order-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:   # profiles, non-alphabetical on purpose\n"
             << "  zeta:  # loaded first\n"
             << "    path: /nonexistent/zeta.tar.gz\n"
             << "  alpha:\n"
             << "    path: /nonexistent/alpha.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    const auto zeta = r.stdout_text.find("Loading model 'zeta'");
    const auto alpha = r.stdout_text.find("Loading model 'alpha'");
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] model order: expected nonzero exit for missing model files\n";
      ++failures;
    } else if (zeta == std::string::npos || alpha != std::string::npos) {
      std::cerr << "[FAIL] model order: expected 'zeta' to load first (declaration order), "
                << "stdout:\n"
                << r.stdout_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] declared model order preserved with inline YAML comment\n";
    }
  }

  // Test 6: an input directory that exists but cannot be enumerated must fail
  // with a concise error, not an uncaught exception. (Skipped when running as
  // root, where permission bits do not apply.)
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto locked_dir =
        fs::temp_directory_path() / ("image-classification-explorer-locked-" + stamp);
    const auto config_path =
        fs::temp_directory_path() / ("image-classification-explorer-locked-" + stamp + ".yaml");
    fs::create_directories(locked_dir);
    fs::permissions(locked_dir, fs::perms::none);
    std::error_code probe;
    fs::directory_iterator probe_it(locked_dir, probe);
    if (!probe) {
      std::cout << "[SKIP] unreadable input directory: permissions not enforced (root?)\n";
    } else {
      {
        std::ofstream config(config_path);
        config << "io:\n"
               << "  input: " << locked_dir.string() << "\n"
               << "models:\n"
               << "  m:\n"
               << "    path: /nonexistent/m.tar.gz\n";
      }
      auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
      fs::remove(config_path);
      if (r.exit_code == 0 || r.stderr_text.find("Error:") == std::string::npos ||
          r.stderr_text.find("terminate") != std::string::npos) {
        std::cerr << "[FAIL] unreadable input directory: expected concise error, got exit "
                  << r.exit_code << "\nstderr:\n"
                  << r.stderr_text << "\n";
        ++failures;
      } else {
        std::cout << "[OK] unreadable input directory produced a concise error\n";
      }
    }
    fs::permissions(locked_dir, fs::perms::owner_all);
    fs::remove_all(locked_dir);
  }

  // Test 7: a profile name containing '.' is rejected with a clear message instead
  // of being truncated to a nonexistent profile.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-dotted-name-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  resnet.v2:\n"
             << "    path: /nonexistent/resnet_v2.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.exit_code == 0 ||
        r.stderr_text.find("models.resnet.v2: profile names must not contain '.'") ==
            std::string::npos) {
      std::cerr << "[FAIL] dotted profile name: expected rejection, got exit " << r.exit_code
                << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] dotted model profile name was rejected\n";
    }
  }

  // Test 8: output_dir is replaced as a whole, so a directory holding anything
  // other than a previous report is refused; a clean one receives the report.
  // Uses an unsupported-extension input so no model is loaded.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("image-classification-explorer-out-" + stamp);
    fs::create_directories(work / "shared");
    {
      std::ofstream(work / "shared" / "notes.txt") << "customer data\n";
    }
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    auto write_config = [&](const fs::path& output_dir) {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << output_dir.string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    };

    write_config(work / "shared");
    auto refused = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    if (refused.exit_code == 0 ||
        refused.stderr_text.find("not part of a previous report") == std::string::npos ||
        !fs::exists(work / "shared" / "notes.txt")) {
      std::cerr << "[FAIL] shared output_dir: expected refusal, got exit " << refused.exit_code
                << "\nstderr:\n"
                << refused.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] output_dir with foreign entries was refused\n";
    }

    write_config(work / "report");
    auto ok = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    auto second = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    bool leftovers = false;
    for (const auto& entry : fs::directory_iterator(work)) {
      if (entry.path().filename().string().rfind(".report.", 0) == 0)
        leftovers = true;
    }
    if (ok.exit_code != 0 || second.exit_code != 0 ||
        !fs::exists(work / "report" / "report.json") ||
        !fs::exists(work / "report" / "report.csv") ||
        !fs::exists(work / "report" / "report.html") || leftovers) {
      std::cerr << "[FAIL] dedicated output_dir: expected report files after two runs, exits "
                << ok.exit_code << "/" << second.exit_code << ", leftovers=" << leftovers
                << "\nstderr:\n"
                << ok.stderr_text << second.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] dedicated output_dir was written and replaced without leftovers\n";
    }
    fs::remove_all(work);
  }

  return failures > 0 ? 1 : 0;
}
