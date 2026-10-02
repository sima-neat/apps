// Unit test for fastflow-anomaly-detector: validates CLI arg handling and config validation.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;
using sima_examples::testing::write_e2e_config;

namespace {

constexpr const char* kExampleName = "fastflow-anomaly-detector";

// ---------------------------------------------------------------------------
// Configuration rules --validate-config-only can reach without a model or a
// stream (Refs #526). Each case is the packaged config with exactly one value
// broken, and asserts the message that names the rule; the accepted cases pin
// both ends of each range so an off-by-one cannot creep in. Tests 6 and 7
// above own the stddev and alpha rules.
// ---------------------------------------------------------------------------
struct RuleCase {
  const char* name;
  sima_examples::testing::ConfigScalars overrides;
  const char* message; // empty: the config must validate
};

int configuration_rule_failures(const std::string& binary) {
  const std::vector<RuleCase> cases = {
      {"model-path-empty", {{"model.path", ""}}, "model.path must be set"},
      {"rtsp-url-empty", {{"source.rtsp_url", ""}}, "source.rtsp_url must be set"},
      {"insight-host-empty", {{"output.insight.host", ""}}, "output.insight.host must be set"},
      {"latency-negative", {{"source.latency_ms", "-1"}}, "source.latency_ms must be >= 0"},
      {"frames-negative", {{"inference.frames", "-1"}}, "inference.frames must be >= 0"},
      {"threshold-above",
       {{"inference.threshold", "1.5"}},
       "inference.threshold must be between 0 and 1"},
      {"threshold-below",
       {{"inference.threshold", "-0.5"}},
       "inference.threshold must be between 0 and 1"},
      {"min-region-zero",
       {{"inference.min_region_px", "0"}},
       "inference.min_region_px must be > 0"},
      {"profile-interval-zero",
       {{"runtime.profile_interval", "0"}},
       "runtime.profile_interval must be > 0"},
      {"video-port-zero",
       {{"output.insight.video_port", "0"}},
       "output.insight.video_port must be in [1, 65535]"},
      {"video-port-above",
       {{"output.insight.video_port", "65536"}},
       "output.insight.video_port must be in [1, 65535]"},
      {"save-every-negative", {{"output.save_every", "-1"}}, "output.save_every must be >= 0"},
      {"heat-max-equals-threshold",
       {{"output.heat_max", "0.5"}},
       "output.heat_max must be greater than inference.threshold"},
      {"heat-max-below-threshold",
       {{"inference.threshold", "0.8"}, {"output.heat_max", "0.7"}},
       "output.heat_max must be greater than inference.threshold"},
      // accepted boundaries
      {"threshold-zero-accepted", {{"inference.threshold", "0"}}, ""},
      {"threshold-one-accepted", {{"inference.threshold", "1"}, {"output.heat_max", "1.5"}}, ""},
      {"alpha-zero-accepted", {{"output.alpha", "0"}}, ""},
      {"alpha-one-accepted", {{"output.alpha", "1"}}, ""},
      {"latency-zero-accepted", {{"source.latency_ms", "0"}}, ""},
      {"min-region-one-accepted", {{"inference.min_region_px", "1"}}, ""},
      {"profile-interval-one-accepted", {{"runtime.profile_interval", "1"}}, ""},
      {"video-port-one-accepted", {{"output.insight.video_port", "1"}}, ""},
      {"video-port-max-accepted", {{"output.insight.video_port", "65535"}}, ""},
      {"save-every-zero-accepted", {{"output.save_every", "0"}}, ""},
  };
  int failures = 0;
  const fs::path scratch = create_test_scratch_dir(kExampleName, "configuration-rules");
  if (scratch.empty()) {
    std::cerr << "[FAIL] could not create config test directory\n";
    return 1;
  }
  for (const RuleCase& c : cases) {
    const fs::path config_path = scratch / (std::string(c.name) + ".yaml");
    write_e2e_config(kExampleName, config_path, c.overrides);
    const ProcessResult r =
        spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
    const bool expect_reject = std::string(c.message).empty() == false;
    if (expect_reject) {
      if (r.exit_code == 0) {
        std::cerr << "[FAIL] " << c.name << ": expected rejection, got exit 0\n";
        ++failures;
      } else if (r.stderr_text.find(std::string("[ERR] ") + c.message) == std::string::npos) {
        std::cerr << "[FAIL] " << c.name << ": stderr does not name the rule (" << c.message
                  << ")\n"
                  << r.stderr_text;
        ++failures;
      } else {
        std::cout << "[OK] " << c.name << " is rejected by its rule\n";
      }
    } else if (r.exit_code != 0 || r.stdout_text.find("Config validated") == std::string::npos) {
      std::cerr << "[FAIL] " << c.name << ": expected the config to validate, got exit "
                << r.exit_code << "\n"
                << r.stderr_text;
      ++failures;
    } else {
      std::cout << "[OK] " << c.name << "\n";
    }
  }
  remove_dir(scratch.string());
  return failures;
}

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

  // Test 5: --validate-config-only accepts the packaged config, the binary's default. Its
  // <rtsp-url> and <insight-host-ip> placeholders are non-empty strings, which is all
  // validation asks of them. No --config, so no process logs land next to the packaged config.
  {
    const ProcessResult r = spawn_and_wait(binary, {"--validate-config-only"}, 20000);
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
    write_e2e_config(kExampleName, invalid, {{"model.normalize.stddev", "[-1, 1, 1]"}});
    const ProcessResult r =
        spawn_and_wait(binary, {"--config", invalid.string(), "--validate-config-only"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] invalid config: expected nonzero exit\n";
      ++failures;
    } else if (r.stderr_text.find("[ERR] model.normalize.stddev must be > 0") ==
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

  // Test 8: every configuration rule --validate-config-only can reach, both ends of each range.
  failures += configuration_rule_failures(binary);

  return failures > 0 ? 1 : 0;
}
