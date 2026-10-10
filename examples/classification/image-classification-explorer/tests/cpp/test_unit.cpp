// Unit test for image-classification-explorer: validates CLI arg handling.
#include "support/testing/test_process.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

namespace {

// The repository copy of a file shipped under src/common, for building a fake
// packaged layout in the test.
std::filesystem::path find_bundled_source(const std::string& name) {
  namespace fs = std::filesystem;
  const std::vector<fs::path> candidates = {
      fs::path("examples/classification/image-classification-explorer/src/common") / name,
      fs::path("src") / "common" / name,
      fs::path("../src/common") / name,
  };
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (fs::is_regular_file(candidate, ec) && !ec)
      return candidate;
  }
  return {};
}

// spawn_and_wait always inherits the caller's directory; this runs the binary
// somewhere else, which is the whole point of the packaged-layout check.
sima_examples::testing::ProcessResult spawn_and_wait_in(const std::string& binary,
                                                        const std::vector<std::string>& args,
                                                        int timeout_ms,
                                                        const std::string& working_directory) {
  namespace fs = std::filesystem;
  const fs::path previous = fs::current_path();
  fs::current_path(working_directory);
  auto result = sima_examples::testing::spawn_and_wait(binary, args, timeout_ms);
  fs::current_path(previous);
  return result;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  // These checks write throwaway configurations into temporary directories, so
  // the application under test can only find src/common through the repository
  // layout. ctest supplies that working directory; say so plainly rather than
  // letting a later filesystem call abort with an unhandled exception.
  if (find_bundled_source("report.css").empty()) {
    std::cerr << "[ERR] run this from the repository root: src/common was not found from "
              << std::filesystem::current_path() << "\n";
    return 2;
  }

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
    // 2 is the configuration-error code the Python entrypoint uses.
    if (r.exit_code != 2 || r.stderr_text.find("Invalid configuration") == std::string::npos) {
      std::cerr << "[FAIL] missing config: expected exit 2 and an \"Invalid configuration\" "
                   "message, got "
                << r.exit_code << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config produced a configuration error (exit 2)\n";
    }
  }

  // Test 3: an unrecognized flag produces a nonzero exit.
  {
    auto r = spawn_and_wait(binary, {"--bogus"}, 20000);
    // argparse exits 2 for a usage error; the C++ entrypoint matches it.
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] unknown flag: expected exit 2, got " << r.exit_code << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag produced a usage error (exit 2)\n";
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
      // Exit 3 is the input-error code the Python entrypoint uses for this.
      if (r.exit_code != 3 ||
          r.stderr_text.find("failed to read input directory") == std::string::npos ||
          r.stderr_text.find("terminate") != std::string::npos) {
        std::cerr << "[FAIL] unreadable input directory: expected exit 3 and a read failure, got "
                  << r.exit_code << "\nstderr:\n"
                  << r.stderr_text << "\n";
        ++failures;
      } else {
        std::cout << "[OK] unreadable input directory produced an input error (exit 3)\n";
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
        r.stderr_text.find("models.resnet.v2: profile names may only contain") ==
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

    // A customer directory without the ownership marker - even one that holds
    // a report.html - must never be swapped away.
    {
      std::ofstream(work / "shared" / "report.html") << "customer page\n";
    }
    write_config(work / "shared");
    auto refused = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    if (refused.exit_code == 0 ||
        refused.stderr_text.find("not created by this application") == std::string::npos ||
        !fs::exists(work / "shared" / "notes.txt") ||
        !fs::exists(work / "shared" / "report.html")) {
      std::cerr << "[FAIL] unowned output_dir: expected refusal, got exit " << refused.exit_code
                << "\nstderr:\n"
                << refused.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] output_dir without the ownership marker was refused\n";
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
        !fs::exists(work / "report" / "report.html") ||
        !fs::exists(work / "report" / ".image-classification-explorer-report") || leftovers) {
      std::cerr << "[FAIL] dedicated output_dir: expected report files after two runs, exits "
                << ok.exit_code << "/" << second.exit_code << ", leftovers=" << leftovers
                << "\nstderr:\n"
                << ok.stderr_text << second.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] dedicated output_dir was written and replaced without leftovers\n";
    }

    // A previous report that has since gained unrelated files is refused too.
    {
      std::ofstream(work / "report" / "notes.txt") << "customer data\n";
    }
    auto foreign = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    if (foreign.exit_code == 0 ||
        foreign.stderr_text.find("not part of a previous report") == std::string::npos ||
        !fs::exists(work / "report" / "notes.txt")) {
      std::cerr << "[FAIL] foreign entries: expected refusal, got exit " << foreign.exit_code
                << "\nstderr:\n"
                << foreign.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] previous report with foreign entries was refused\n";
    }

    // A symlinked output_dir keeps the link and replaces its target.
    fs::create_directories(work / "real_report");
    fs::create_directory_symlink(work / "real_report", work / "linked");
    write_config(work / "linked");
    auto via_link = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    auto via_link2 = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    if (via_link.exit_code != 0 || via_link2.exit_code != 0 || !fs::is_symlink(work / "linked") ||
        fs::read_symlink(work / "linked") != work / "real_report" ||
        !fs::exists(work / "real_report" / "report.json")) {
      std::cerr << "[FAIL] symlinked output_dir: expected link preserved and target replaced, "
                   "exits "
                << via_link.exit_code << "/" << via_link2.exit_code << "\nstderr:\n"
                << via_link.stderr_text << via_link2.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] symlinked output_dir kept the link and replaced its target\n";
    }
    fs::remove_all(work);
  }

  // Test 9: a blank line inside the first num_classes label-map lines is rejected
  // instead of silently shifting every later class id.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto labels_path =
        fs::temp_directory_path() / ("image-classification-explorer-labels-" + stamp + ".txt");
    const auto config_path =
        fs::temp_directory_path() / ("image-classification-explorer-labels-" + stamp + ".yaml");
    {
      std::ofstream(labels_path) << "cat\n\nbird\ndog\n";
    }
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n"
             << "    num_classes: 3\n"
             << "    label_map: " << labels_path.string() << "\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    fs::remove(labels_path);
    if (r.exit_code == 0 || r.stderr_text.find("line 2 is blank") == std::string::npos) {
      std::cerr << "[FAIL] blank label line: expected rejection, got exit " << r.exit_code
                << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] blank label-map line was rejected\n";
    }
  }

  // Test 10: a quoted YAML profile key must be reported without its quotes, so
  // C++ metadata matches what Python's YAML loader produces.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-quoted-key-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  \"resnet_50\":\n"
             << "    path: /nonexistent/resnet_50.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    const bool unquoted = r.stdout_text.find("Loading model 'resnet_50'") != std::string::npos;
    const bool quoted = r.stdout_text.find("\"resnet_50\"") != std::string::npos;
    if (!unquoted || quoted) {
      std::cerr << "[FAIL] quoted profile key: expected unquoted model name, stdout:\n"
                << r.stdout_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] quoted YAML profile key was unquoted\n";
    }
  }

  // Test 11: a missing custom label map whose basename matches the bundled one
  // must be rejected rather than silently loading the shipped ImageNet map.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto config_path =
        fs::temp_directory_path() / ("image-classification-explorer-labelmap-" + stamp + ".yaml");
    const auto missing =
        fs::temp_directory_path() / ("no-such-dir-" + stamp) / "imagenet_labels.txt";
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n"
             << "    label_map: " << missing.string() << "\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.exit_code == 0 || r.stderr_text.find("failed to open label map") == std::string::npos) {
      std::cerr << "[FAIL] missing custom label map: expected rejection, got exit " << r.exit_code
                << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] missing custom label map was rejected\n";
    }
  }

  // Test 13: a profile name containing a colon cannot be addressed through the
  // config keys, so it must be rejected by its real name rather than silently
  // truncated at the first colon.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-colon-name-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  \"resnet:50\":\n"
             << "    path: /nonexistent/resnet_50.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.exit_code == 0 ||
        r.stderr_text.find("models.resnet:50: profile names may only contain") ==
            std::string::npos) {
      std::cerr << "[FAIL] colon profile name: expected rejection naming resnet:50, got exit "
                << r.exit_code << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] colon in a model profile name was rejected by its real name\n";
    }
  }

  // Test 14: unquoted YAML scalars that are not strings (bool, null, integers in
  // any spelling) must be rejected, because PyYAML turns them into Python
  // objects whose text differs from the raw spelling C++ reads.
  {
    namespace fs = std::filesystem;
    const char* const kNonStringKeys[] = {"true", "null", "01", "1_0",
                                          "0x1",  "-1",   "+2", "2026-09-22"};
    for (const auto* key : kNonStringKeys) {
      const auto config_path =
          fs::temp_directory_path() /
          ("image-classification-explorer-nonstring-" + std::string(key) + "-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
      {
        std::ofstream config(config_path);
        config << "models:\n"
               << "  " << key << ":\n"
               << "    path: /nonexistent/m.tar.gz\n";
      }
      auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
      fs::remove(config_path);
      if (r.exit_code == 0 ||
          r.stderr_text.find("is not a string; quote it in config.yaml") == std::string::npos) {
        std::cerr << "[FAIL] unquoted non-string key '" << key << "': expected rejection, got exit "
                  << r.exit_code << "\nstderr:\n"
                  << r.stderr_text << "\n";
        ++failures;
      }
    }
    if (failures == 0)
      std::cout << "[OK] unquoted non-string profile names were rejected\n";
  }

  // Test 15: the same names are accepted when quoted, which makes them strings
  // in both implementations. (Fails later on the missing model file, not on the
  // name, so assert the name was accepted.)
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-quoted-numeric-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  \"1\":\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.stderr_text.find("is not a string") != std::string::npos ||
        r.stdout_text.find("Loading model '1'") == std::string::npos) {
      std::cerr << "[FAIL] quoted numeric name: expected it to be accepted, exit " << r.exit_code
                << "\nstdout:\n"
                << r.stdout_text << "stderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] quoted numeric profile name was accepted\n";
    }
  }

  // Test 16: a non-integral scalar must be rejected rather than truncated, so
  // both entrypoints run the same settings.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-float-topk-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n"
             << "    top_k: 1.9\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    // "nonzero" would also be satisfied by the missing model archive, so check
    // that the configuration itself was rejected, with the code Python uses.
    if (r.exit_code != 2 ||
        r.stderr_text.find("models.m.top_k must be an integer") == std::string::npos) {
      std::cerr << "[FAIL] non-integral top_k: expected exit 2 naming top_k, got " << r.exit_code
                << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] non-integral top_k was rejected\n";
    }
  }

  // Test 19: a zero or negative per-image timeout is not a usable bound.
  {
    namespace fs = std::filesystem;
    for (const char* timeout : {"0", "-1"}) {
      const auto config_path =
          fs::temp_directory_path() /
          ("image-classification-explorer-timeout-" +
           std::string(timeout == std::string("0") ? "zero" : "negative") + "-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
      {
        std::ofstream config(config_path);
        config << "runtime:\n"
               << "  timeout_ms: " << timeout << "\n"
               << "models:\n"
               << "  m:\n"
               << "    path: /nonexistent/m.tar.gz\n";
      }
      auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
      fs::remove(config_path);
      if (r.exit_code == 0 ||
          r.stderr_text.find("runtime.timeout_ms must be positive") == std::string::npos) {
        std::cerr << "[FAIL] timeout_ms " << timeout << ": expected rejection, got exit "
                  << r.exit_code << "\nstderr:\n"
                  << r.stderr_text << "\n";
        ++failures;
      }
    }
    if (failures == 0)
      std::cout << "[OK] non-positive runtime.timeout_ms was rejected\n";
  }

  // Test 23: the thumbnail digest must match Python's, so both implementations
  // name thumbnails identically and neither renames them between runs.
  {
    const std::string expected = "771220d11190d381"; // pinned in test_unit.py too
    std::uint64_t digest = 0xCBF29CE484222325ULL;
    for (unsigned char byte : std::string("/a/b.jpg")) {
      digest ^= static_cast<std::uint64_t>(byte);
      digest *= 0x100000001B3ULL;
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << digest;
    if (out.str() != expected) {
      std::cerr << "[FAIL] thumbnail digest: expected " << expected << ", got " << out.str()
                << "\n";
      ++failures;
    } else {
      std::cout << "[OK] thumbnail digest matches the Python implementation\n";
    }
  }

  // Test 24: a malformed validation block is rejected during configuration
  // loading, not after the report has been written.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-validation-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "validation:\n"
             << "  expected_class_id: abc\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.exit_code == 0 || r.stderr_text.find("validation.expected_class_id must be an integer") ==
                                std::string::npos) {
      std::cerr << "[FAIL] malformed validation: expected rejection, got exit " << r.exit_code
                << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] malformed validation.expected_class_id was rejected up front\n";
    }
  }

  // Test 25: an unreadable input path is an input error (exit 3), the code the
  // Python entrypoint uses, not a generic runtime failure.
  {
    namespace fs = std::filesystem;
    const auto config_path =
        fs::temp_directory_path() /
        ("image-classification-explorer-missing-input-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
    {
      std::ofstream config(config_path);
      config << "io:\n"
             << "  input: /nonexistent/directory/of/images\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    fs::remove(config_path);
    if (r.exit_code != 3 || r.stderr_text.find("input path does not exist") == std::string::npos) {
      std::cerr << "[FAIL] missing input: expected exit 3, got " << r.exit_code << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] missing input path produced an input error (exit 3)\n";
    }
  }

  // Test 26: `extensions: jpg` (no leading dot) must still match .jpg files.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("image-classification-explorer-ext-" + stamp);
    const auto images = work / "images";
    fs::create_directories(images);
    {
      std::ofstream(images / "a.jpg") << "not really an image\n";
    }
    // The config lives outside the scanned directory: a .yaml inside it would
    // itself be reported as an unsupported extension.
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << images.string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "  extensions: jpg\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    // The model cannot load, so the run fails - but the file must have been
    // accepted as an input rather than reported as an unsupported extension.
    const bool skipped_it =
        (r.stdout_text + r.stderr_text).find("unsupported extension") != std::string::npos;
    const bool classified = r.stdout_text.find("Classifying 1 image(s)") != std::string::npos;
    if (skipped_it || !classified) {
      std::cerr << "[FAIL] extensions without a dot: skipped=" << skipped_it
                << " classified=" << classified << "\nstdout:\n"
                << r.stdout_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] an extension without a leading dot still matched\n";
    }
    fs::remove_all(work);
  }

  // Test 28: an entry that cannot be stat'ed must not abort the scan. The
  // throwing is_regular_file() overload raised filesystem_error on a
  // self-referential symlink, so the run ended at exit 6 with no report while
  // Python skipped the entry and continued.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("image-classification-explorer-loop-" + stamp);
    const auto images = work / "images";
    fs::create_directories(images);
    {
      std::ofstream(images / "a.jpg") << "not really an image\n";
    }
    std::error_code link_ec;
    fs::create_symlink("loop", images / "loop", link_ec); // points at itself
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << images.string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto r = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    // The model cannot load, so the run still fails - but the scan must have
    // completed and found the one real image.
    const bool classified = r.stdout_text.find("Classifying 1 image(s)") != std::string::npos;
    const bool filesystem_error =
        (r.stdout_text + r.stderr_text).find("filesystem error") != std::string::npos;
    if (link_ec) {
      std::cout << "[OK] unstatable directory entry (skipped: symlinks unavailable here)\n";
    } else if (!classified || filesystem_error) {
      std::cerr << "[FAIL] unstatable directory entry: classified=" << classified
                << " filesystem_error=" << filesystem_error << " exit " << r.exit_code
                << "\nstdout:\n"
                << r.stdout_text << "\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] an unstatable directory entry was skipped, not fatal\n";
    }
    fs::remove_all(work);
  }

  // Test 29: a numeric or null-like output_dir must name the same directory
  // here as in Python. PyYAML resolves `010` to 8 and unquotes `"null"` to a
  // plain string, while ScalarConfig sees only text and cannot tell either
  // spelling apart, so both sides canonicalise. Uses an unsupported-extension
  // input so no model is loaded.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work =
        fs::temp_directory_path() / ("image-classification-explorer-dirname-" + stamp);
    fs::create_directories(work);
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    // The run happens with `work` as the working directory, so the report
    // assets are resolved through the config-file anchor rather than a
    // repo-relative path. Put them beside config.yaml, as a package does.
    for (const char* asset : {"report.css", "report.js"}) {
      const fs::path source = find_bundled_source(asset);
      if (!source.empty())
        fs::copy_file(source, work / asset, fs::copy_options::overwrite_existing);
    }
    auto run_with = [&](const std::string& output_dir_value) {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << output_dir_value << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
      config.close();
      // Absolute: spawn_and_wait_in changes the working directory, which is
      // what makes the relative output_dir below resolve inside `work`.
      return spawn_and_wait_in(fs::absolute(binary).string(),
                               {"--config", (work / "config.yaml").string()}, 20000, work.string());
    };

    // `010` is octal 8 to PyYAML, so the directory is "8" in Python.
    auto numeric = run_with("010");
    const bool canonical = fs::exists(work / "8") && !fs::exists(work / "010");
    // `"null"` is unquoted before the null test, so C++ falls back to "report";
    // Python now does the same rather than creating a directory named "null".
    auto nulled = run_with("\"null\"");
    const bool defaulted = fs::exists(work / "report") && !fs::exists(work / "null");

    if (numeric.exit_code != 0 || nulled.exit_code != 0 || !canonical || !defaulted) {
      std::cerr << "[FAIL] output_dir canonicalisation: numeric_exit=" << numeric.exit_code
                << " canonical=" << canonical << " nulled_exit=" << nulled.exit_code
                << " defaulted=" << defaulted << "\nstderr:\n"
                << numeric.stderr_text << nulled.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] numeric and null-like output_dir names match Python\n";
    }
    fs::remove_all(work);
  }

  // Test 27: the packaged layout works from an unrelated working directory.
  //
  // SIMANEAT_APPS_EXAMPLE_SOURCE_DIR is repository-relative, so a customer who
  // runs the shipped binary from their own directory with an explicit --config
  // can only find src/common through the executable. Recreate that layout - a
  // copy of the binary under src/cpp/pre-built/ beside a src/common/ - and run
  // it from somewhere else entirely.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto pkg = fs::temp_directory_path() / ("image-classification-explorer-pkg-" + stamp);
    const auto bin_dir = pkg / "src" / "cpp" / "pre-built";
    const auto common = pkg / "src" / "common";
    fs::create_directories(bin_dir);
    fs::create_directories(common);

    std::error_code ec;
    fs::copy_file(binary, bin_dir / "image-classification-explorer",
                  fs::copy_options::overwrite_existing, ec);
    if (ec) {
      std::cerr << "[SKIP] packaged layout: could not copy the binary: " << ec.message() << "\n";
    } else {
      fs::permissions(bin_dir / "image-classification-explorer",
                      fs::perms::owner_all | fs::perms::group_exec | fs::perms::others_exec,
                      fs::perm_options::add, ec);

      // The label map and the report assets both have to be found.
      for (const char* name : {"imagenet_labels.txt", "report.css", "report.js"}) {
        const fs::path source = find_bundled_source(name);
        if (!source.empty())
          fs::copy_file(source, common / name, fs::copy_options::overwrite_existing, ec);
      }

      {
        std::ofstream(pkg / "input.txt") << "not an image\n";
      }
      const auto config_path = pkg / "config.yaml";
      {
        std::ofstream config(config_path);
        config << "io:\n"
               << "  input: " << (pkg / "input.txt").string() << "\n"
               << "  output_dir: " << (pkg / "report").string() << "\n"
               << "models:\n"
               << "  m:\n"
               << "    path: /nonexistent/m.tar.gz\n"
               << "    num_classes: 1000\n"
               << "    label_map: src/common/imagenet_labels.txt\n";
      }

      // Run from a directory unrelated to the package.
      const auto elsewhere = fs::temp_directory_path();
      auto r = spawn_and_wait_in((bin_dir / "image-classification-explorer").string(),
                                 {"--config", config_path.string()}, 20000, elsewhere.string());
      const bool found_labels = r.stderr_text.find("failed to open label map") == std::string::npos;
      const bool found_assets =
          r.stderr_text.find("failed to read bundled report asset") == std::string::npos;
      // The absence of two strings is also true of a binary that never ran, so
      // require the run to have succeeded and written its report.
      const bool published = r.exit_code == 0 && fs::exists(pkg / "report" / "report.html");
      if (!found_labels || !found_assets || !published) {
        std::cerr << "[FAIL] packaged layout: labels_found=" << found_labels
                  << " assets_found=" << found_assets << " published=" << published << " exit "
                  << r.exit_code << "\nstderr:\n"
                  << r.stderr_text << "\n";
        ++failures;
      } else {
        std::cout << "[OK] packaged layout resolved src/common from another directory\n";
      }
    }
    fs::remove_all(pkg, ec);
  }

  return failures > 0 ? 1 : 0;
}
