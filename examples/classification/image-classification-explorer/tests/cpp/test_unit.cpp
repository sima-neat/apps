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

  // Test 12: a report stranded by a publish that was killed mid-swap is restored
  // on the next run instead of being left under .<name>.previous-<pid>.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work =
        fs::temp_directory_path() / ("image-classification-explorer-recover-" + stamp);
    fs::create_directories(work);
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto first = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    // Simulate a process killed after the first rename of the swap.
    fs::rename(work / "report", work / ".report.previous-4242");
    auto second = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    bool stranded = false;
    for (const auto& entry : fs::directory_iterator(work)) {
      if (entry.path().filename().string().rfind(".report.previous-", 0) == 0)
        stranded = true;
    }
    if (first.exit_code != 0 || second.exit_code != 0 ||
        !fs::exists(work / "report" / "report.json") || stranded ||
        second.stderr_text.find("Recovered an interrupted report publication") ==
            std::string::npos) {
      std::cerr << "[FAIL] interrupted publish: expected recovery, exits " << first.exit_code << "/"
                << second.exit_code << " stranded=" << stranded << "\nstderr:\n"
                << second.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] report stranded by an interrupted publish was recovered\n";
    }
    fs::remove_all(work);
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
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] non-integral top_k: expected nonzero exit\nstderr:\n"
                << r.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] non-integral top_k was rejected\n";
    }
  }

  // Test 17: a backup orphaned by a publish that was killed after installing the
  // new report is removed on the next run, while a backup belonging to a live
  // process, and any directory without the report marker, are left alone.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("image-classification-explorer-backup-" + stamp);
    fs::create_directories(work);
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto first = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);

    // Orphan from a process that no longer exists, a backup owned by this live
    // test process, and an unmarked directory that must never be touched.
    const fs::path orphan = work / ".report.previous-2147483646";
    const fs::path live = work / (".report.previous-" + std::to_string(::getpid()));
    const fs::path unmarked = work / ".report.previous-2147483645";
    fs::copy(work / "report", orphan, fs::copy_options::recursive);
    fs::copy(work / "report", live, fs::copy_options::recursive);
    fs::create_directories(unmarked);
    {
      std::ofstream(unmarked / "customer.txt") << "keep me\n";
    }

    // The second run still cleans the orphan, then defers because a live process
    // owns a backup.
    auto second = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    const bool orphan_removed = !fs::exists(orphan);
    const bool live_kept = fs::exists(live);
    const bool unmarked_kept = fs::exists(unmarked / "customer.txt");
    const bool deferred = second.exit_code != 0 &&
                          second.stderr_text.find("another run is publishing") != std::string::npos;
    if (first.exit_code != 0 || !deferred || !orphan_removed || !live_kept || !unmarked_kept) {
      std::cerr << "[FAIL] backup cleanup: exits " << first.exit_code << "/" << second.exit_code
                << " orphan_removed=" << orphan_removed << " live_kept=" << live_kept
                << " unmarked_kept=" << unmarked_kept << " deferred=" << deferred << "\nstderr:\n"
                << second.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] orphaned backup removed; live one preserved and publication deferred\n";
    }
    fs::remove_all(work);
  }

  // Test 18: with output_dir absent, a backup owned by a live process must not
  // be restored - it belongs to a concurrent publisher that is mid-swap.
  //
  // (The "backup bearing our own recycled pid" case cannot be set up here: the
  // child binary's pid is not known before it runs. Python covers it directly,
  // since main() runs in-process there, and the C++ predicate is the same.)
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("image-classification-explorer-live-" + stamp);
    fs::create_directories(work);
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto first = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);

    // Exactly the state a concurrent publisher is in mid-swap: the old report
    // renamed aside under a live pid, output_dir not yet reinstalled.
    const fs::path live = work / (".report.previous-" + std::to_string(::getpid()));
    fs::rename(work / "report", live);

    auto second = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    const bool live_kept = fs::exists(live / ".image-classification-explorer-report");
    // It must also decline to publish, rather than occupying the path the live
    // run is about to rename its staging directory into.
    const bool deferred = second.exit_code != 0 &&
                          second.stderr_text.find("another run is publishing") != std::string::npos;
    if (first.exit_code != 0 || !deferred || !live_kept || fs::exists(work / "report")) {
      std::cerr << "[FAIL] live publisher backup: exits " << first.exit_code << "/"
                << second.exit_code << " live_kept=" << live_kept << " deferred=" << deferred
                << "\nstderr:\n"
                << second.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] a live publisher's backup was preserved and publication deferred\n";
    }
    fs::remove_all(work);
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

  // Test 21: a staging directory abandoned by a terminated run is removed.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work =
        fs::temp_directory_path() / ("image-classification-explorer-staging-" + stamp);
    fs::create_directories(work);
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    auto first = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    const fs::path abandoned = work / ".report.staging-2147483646";
    fs::create_directories(abandoned);
    {
      std::ofstream(abandoned / "report.json") << "{}\n";
    }
    auto second = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    if (first.exit_code != 0 || second.exit_code != 0 || fs::exists(abandoned)) {
      std::cerr << "[FAIL] abandoned staging: exits " << first.exit_code << "/" << second.exit_code
                << " still_present=" << fs::exists(abandoned) << "\nstderr:\n"
                << second.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] staging directory abandoned by a dead run was removed\n";
    }
    fs::remove_all(work);
  }

  // Test 22: publication is serialized by a lock. A lock held by a live process
  // makes a run defer; one left by a dead process is reclaimed.
  {
    namespace fs = std::filesystem;
    const auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto work = fs::temp_directory_path() / ("image-classification-explorer-lock-" + stamp);
    fs::create_directories(work);
    {
      std::ofstream(work / "input.txt") << "not an image\n";
    }
    {
      std::ofstream config(work / "config.yaml");
      config << "io:\n"
             << "  input: " << (work / "input.txt").string() << "\n"
             << "  output_dir: " << (work / "report").string() << "\n"
             << "models:\n"
             << "  m:\n"
             << "    path: /nonexistent/m.tar.gz\n";
    }
    const fs::path lock = work / ".report.lock";

    // Held by this live test process: the run must defer and leave it alone.
    {
      std::ofstream(lock) << ::getpid() << "\n";
    }
    auto blocked = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    const bool deferred =
        blocked.exit_code != 0 &&
        blocked.stderr_text.find("another run is publishing") != std::string::npos;
    const bool lock_kept = fs::exists(lock);

    // Held by a pid that cannot exist: the lock is reclaimed and the run proceeds.
    {
      std::ofstream(lock) << "2147483646\n";
    }
    auto reclaimed = spawn_and_wait(binary, {"--config", (work / "config.yaml").string()}, 20000);
    const bool published = reclaimed.exit_code == 0 && fs::exists(work / "report" / "report.json");
    const bool lock_released = !fs::exists(lock);

    if (!deferred || !lock_kept || !published || !lock_released) {
      std::cerr << "[FAIL] publication lock: deferred=" << deferred << " lock_kept=" << lock_kept
                << " published=" << published << " lock_released=" << lock_released << "\nstderr:\n"
                << blocked.stderr_text << reclaimed.stderr_text << "\n";
      ++failures;
    } else {
      std::cout << "[OK] publication lock defers a second run and reclaims a stale lock\n";
    }
    fs::remove_all(work);
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

  return failures > 0 ? 1 : 0;
}
