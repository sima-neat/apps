// Unit test for single-stream-thermal-face-detector: validates CLI arg handling.
#include "support/runtime/pull_status.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using sima_examples::testing::ProcessResult;
using sima_examples::testing::spawn_and_wait;

namespace {

// ---------------------------------------------------------------------------
// Configuration rules --validate-config-only can reach without a model or a
// stream (Refs #526). Each case is the minimal valid config with exactly one
// value broken, and asserts the message that names the rule, plus the
// unbroken baseline as a control.
// ---------------------------------------------------------------------------
struct RuleCase {
  const char* name;
  const char* source;         // a full source: section, or "" for the default one
  const char* model_extra;    // extra lines under model:
  const char* extra_sections; // extra top-level sections
  const char* insight_extra;  // extra lines under output.insight:
  const char* model_path;
  const char* host;
  const char* message;
};

std::string rule_config(const RuleCase& c) {
  const std::string source = std::string(c.source).empty()
                                 ? "source:\n  rtsp_url: rtsp://127.0.0.1:8554/src1\n"
                                 : c.source;
  return std::string("model:\n  path: '") + c.model_path + "'\n" + c.model_extra + source +
         c.extra_sections + "output:\n  insight:\n    host: '" + c.host + "'\n" + c.insight_extra;
}

int configuration_rule_failures(const std::string& binary) {
  const std::vector<RuleCase> cases = {
      {"rtsp-url-empty", "source:\n  rtsp_url: ''\n", "", "", "", "models/model.tar.gz",
       "127.0.0.1", "source.rtsp_url must be set"},
      {"model-path-empty", "", "", "", "", "", "127.0.0.1", "model.path must be set"},
      {"model-labels-empty", "", "  labels: ''\n", "", "", "models/model.tar.gz", "127.0.0.1",
       "model.labels must be set"},
      {"insight-host-empty", "", "", "", "", "models/model.tar.gz", "",
       "output.insight.host must be set"},
      {"latency-negative", "source:\n  rtsp_url: rtsp://127.0.0.1:8554/src1\n  latency_ms: -1\n",
       "", "", "", "models/model.tar.gz", "127.0.0.1", "source.latency_ms must be >= 0"},
      {"frames-negative", "", "", "inference:\n  frames: -1\n", "", "models/model.tar.gz",
       "127.0.0.1", "inference.frames must be >= 0"},
      {"min-score-above", "", "", "inference:\n  min_score: 1.5\n", "", "models/model.tar.gz",
       "127.0.0.1", "inference.min_score must be between 0 and 1"},
      {"nms-below", "", "", "inference:\n  nms_iou: -0.5\n", "", "models/model.tar.gz", "127.0.0.1",
       "inference.nms_iou must be between 0 and 1"},
      {"max-detections-zero", "", "", "inference:\n  max_detections: 0\n", "",
       "models/model.tar.gz", "127.0.0.1", "inference.max_detections must be > 0"},
      {"profile-interval-zero", "", "", "runtime:\n  profile_interval: 0\n", "",
       "models/model.tar.gz", "127.0.0.1", "runtime.profile_interval must be > 0"},
      {"video-port-zero", "", "", "", "    video_port: 0\n", "models/model.tar.gz", "127.0.0.1",
       "output.insight.video_port must be > 0"},
      {"metadata-port-zero", "", "", "", "    metadata_port: 0\n", "models/model.tar.gz",
       "127.0.0.1", "output.insight.metadata_port must be > 0"},
  };
  int failures = 0;
  const std::string temp_dir = sima_examples::testing::create_test_scratch_dir(
      "single-stream-thermal-face-detector", "configuration-rules");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] could not create config test directory\n";
    return 1;
  }
  const std::filesystem::path config_path = std::filesystem::path(temp_dir) / "config.yaml";
  for (const RuleCase& c : cases) {
    std::ofstream(config_path, std::ios::trunc) << rule_config(c);
    const auto r =
        spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
    if (r.exit_code == 0) {
      std::cerr << "[FAIL] " << c.name << ": expected rejection, got exit 0\n";
      ++failures;
    } else if (r.stderr_text.find(c.message) == std::string::npos) {
      std::cerr << "[FAIL] " << c.name << ": stderr does not name the rule (" << c.message << ")\n";
      ++failures;
    } else {
      std::cout << "[OK] " << c.name << " is rejected by its rule\n";
    }
  }
  // The control: the same minimal config with nothing broken validates.
  std::ofstream(config_path, std::ios::trunc)
      << rule_config({"baseline", "", "", "", "", "models/model.tar.gz", "127.0.0.1", ""});
  const auto r =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  if (r.exit_code != 0 || r.stdout_text.find("Config validated") == std::string::npos) {
    std::cerr << "[FAIL] minimal config should validate\n";
    ++failures;
  } else {
    std::cout << "[OK] minimal config validates\n";
  }
  sima_examples::testing::remove_dir(temp_dir);
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
      std::cerr << "[FAIL] unknown flag: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  // Test 3: missing config path value is rejected.
  {
    auto r = spawn_and_wait(binary, {"--config"}, 20000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] missing config path: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] missing config path: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config path correctly rejected\n";
    }
  }

  // Test 4: missing config file is rejected.
  {
    auto r = spawn_and_wait(
        binary, {"--config", "/nonexistent/single-stream-thermal-face-detector-config.yaml"},
        20000);
    if (r.exit_code != 1) {
      std::cerr << "[FAIL] bad config: expected exit 1, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path correctly rejected\n";
    }
  }

  // Test 5: --validate-config-only parses the shipped default config without hardware.
  {
    auto r = spawn_and_wait(binary, {"--validate-config-only"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] validate-config-only: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Config validated") == std::string::npos) {
      std::cerr << "[FAIL] validate-config-only: stdout does not confirm validation\n";
      ++failures;
    } else {
      std::cout << "[OK] validate-config-only accepts the shipped config\n";
    }
  }

  // Test 6: every configuration rule --validate-config-only can reach fires with its message.
  failures += configuration_rule_failures(binary);

  // The pull loop treats a timeout as "try again" and a closed output or pull error as the
  // end of the run, carrying the runtime's own reason.
  {
    using sima_examples::pull_status_has_sample;
    using simaai::neat::PullStatus;
    simaai::neat::PullError pull_error;
    pull_error.message = "queue torn down";
    const auto thrown_message = [&](PullStatus status) -> std::string {
      try {
        (void)pull_status_has_sample(status, "detections", pull_error, "source reached EOS");
      } catch (const std::runtime_error& error) {
        return error.what();
      }
      return "";
    };
    const std::string closed = thrown_message(PullStatus::Closed);
    const std::string errored = thrown_message(PullStatus::Error);
    if (closed != "detections output closed unexpectedly: source reached EOS") {
      std::cerr << "[FAIL] closed output should end the run with the reason, got: " << closed
                << "\n";
      ++failures;
    } else if (errored != "failed to pull detections: queue torn down") {
      std::cerr << "[FAIL] pull error should end the run with its message, got: " << errored
                << "\n";
      ++failures;
    } else if (pull_status_has_sample(PullStatus::Timeout, "detections", pull_error, "") ||
               !pull_status_has_sample(PullStatus::Ok, "detections", pull_error, "")) {
      std::cerr << "[FAIL] a timeout is not a sample and a successful pull is\n";
      ++failures;
    } else {
      std::cout << "[OK] closed output and pull error are terminal, timeout is not\n";
    }
  }

  return failures > 0 ? 1 : 0;
}
