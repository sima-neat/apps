// Unit test for fastflow-anomaly-detector: validates CLI arg handling and config validation.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::example_common_config_path;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_e2e_config;

namespace {

constexpr const char* kExampleName = "fastflow-anomaly-detector";

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  int failures = 0;

  // Test 1: --help exits successfully and prints usage.
  {
    const ProcessResult r = spawn_and_wait(binary, {"--help"}, 20000);
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

  // Test 2: unknown flag is rejected before model/runtime startup.
  {
    const ProcessResult r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] --bogus: expected nonzero exit\n";
      ++failures;
    } else if (r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] --bogus: stderr does not mention unknown argument\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag rejected\n";
    }
  }

  // Test 3: missing --config value is rejected.
  {
    const ProcessResult r = spawn_and_wait(binary, {"--config"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] --config without a path: expected nonzero exit\n";
      ++failures;
    } else if (r.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] --config without a path: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config path rejected\n";
    }
  }

  // Test 4: bad config path is rejected.
  {
    const ProcessResult r = spawn_and_wait(binary, {"--config", "/nonexistent_config.yaml"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] bad config: expected nonzero exit\n";
      ++failures;
    } else if (r.stderr_text.find("[ERR]") == std::string::npos) {
      std::cerr << "[FAIL] bad config: stderr does not report [ERR]\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path rejected\n";
    }
  }

  // Test 5: --validate-config-only accepts the packaged config. Its <rtsp-url> and
  // <insight-host-ip> placeholders are non-empty strings, which is all validation asks of them.
  {
    const fs::path packaged = example_common_config_path(kExampleName);
    const ProcessResult r =
        spawn_and_wait(binary, {"--config", packaged.string(), "--validate-config-only"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] validate-config-only: expected exit 0, got " << r.exit_code << "\n"
                << r.stderr_text;
      ++failures;
    } else if (r.stdout_text.find("Config validated") == std::string::npos) {
      std::cerr << "[FAIL] validate-config-only: stdout does not confirm validation\n";
      ++failures;
    } else {
      std::cout << "[OK] validate-config-only accepts the packaged config\n";
    }
  }

  // Test 6: a config that fails validation is rejected with [ERR].
  {
    const fs::path scratch = create_test_scratch_dir(kExampleName, "config");
    const fs::path invalid = scratch / "invalid.yaml";
    write_e2e_config(kExampleName, invalid, {{"model.normalize.stddev", "[1, 0, 1]"}});
    const ProcessResult r =
        spawn_and_wait(binary, {"--config", invalid.string(), "--validate-config-only"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] invalid config: expected nonzero exit\n";
      ++failures;
    } else if (r.stderr_text.find("[ERR] model.normalize.stddev must not contain zero") ==
               std::string::npos) {
      std::cerr << "[FAIL] invalid config: stderr does not explain failure\n" << r.stderr_text;
      ++failures;
    } else {
      std::cout << "[OK] invalid config rejected\n";
    }
    remove_dir(scratch.string());
  }

  // Test 7: the heatmap opacity is checked too.
  {
    const fs::path scratch = create_test_scratch_dir(kExampleName, "config_alpha");
    const fs::path invalid = scratch / "invalid_alpha.yaml";
    write_e2e_config(kExampleName, invalid, {{"output.alpha", "1.5"}});
    const ProcessResult r =
        spawn_and_wait(binary, {"--config", invalid.string(), "--validate-config-only"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] invalid alpha: expected nonzero exit\n";
      ++failures;
    } else if (r.stderr_text.find("[ERR] output.alpha must be between 0 and 1") ==
               std::string::npos) {
      std::cerr << "[FAIL] invalid alpha: stderr does not explain failure\n" << r.stderr_text;
      ++failures;
    } else {
      std::cout << "[OK] invalid alpha rejected\n";
    }
    remove_dir(scratch.string());
  }

  return failures > 0 ? 1 : 0;
}
