// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "support/testing/test_process.h"

// The pure policy header carries no Neat/OpenCV dependencies, so it links into
// the standalone unit-test binary directly.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

bool expect_true(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[FAIL] " << message << "\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& message) {
  return expect_true(haystack.find(needle) != std::string::npos, message);
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir =
      create_test_scratch_dir("adaptive-resolution-object-detector", test_name);
  if (temp_dir.empty())
    throw std::runtime_error("failed to create temp directory");
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

// ── CLI contract (needs the example binary) ──────────────────────────────────

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only") &&
         expect_contains(result.stdout_text, "--mode", "help mentions --mode");
}

// ── mode dispatch ────────────────────────────────────────────────────────────
//
// This binary is an entry point for BOTH topologies, mirroring
// src/python/main.py. The pipelines chooser relies on the two taking identical
// flags, so a drift here breaks the language toggle rather than just this test.

bool test_rejects_unknown_mode(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--mode", "nonsense", "--help"}, 20000);
  return expect_true(result.exit_code != 0, "unknown --mode is rejected");
}

fs::path write_adaptive_config(const char* name) {
  return write_config(name,
                      "model:\n"
                      "  path: assets/models/yolo26m-det-int8-b1.tar.gz\n"
                      "streams:\n"
                      "  max_streams: 8\n"
                      "  sources:\n"
                      "    - id: cam-1\n"
                      "      rtsp_url: rtsp://127.0.0.1:8554/src1\n"
                      "output:\n"
                      "  insight:\n"
                      "    host: 127.0.0.1\n");
}

bool test_mode_adaptive_validates(const std::string& binary) {
  const fs::path config_path = write_adaptive_config("test_mode_adaptive_validates");
  const auto result = spawn_and_wait(
      binary, {"--mode", "adaptive", "--config", config_path.string(),
               "--validate-config-only"}, 20000);
  return expect_true(result.exit_code == 0, "--mode adaptive validates its config");
}


bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_rich_sources(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_rich_sources",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  max_streams: 8\n"
                   "  sources:\n"
                   "    - id: cam-1\n"
                   "      rtsp_url: rtsp://127.0.0.1:8554/src1\n"
                   "    - id: cam-2\n"
                   "      rtsp_url: rtsp://127.0.0.1:8554/src2\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "rich two-source config validates") &&
      expect_contains(result.stdout_text, "streams=2", "validate output reports stream count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_bare_list(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_bare_list",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "  - rtsp://127.0.0.1:8554/src2\n"
                   "  - rtsp://127.0.0.1:8554/src3\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "bare-list config validates") &&
      expect_contains(result.stdout_text, "streams=3", "bare-list validate reports stream count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_rejects_over_max_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_rejects_over_max_streams",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams:\n"
                   "  max_streams: 1\n"
                   "  sources:\n"
                   "    - id: cam-1\n"
                   "      rtsp_url: rtsp://127.0.0.1:8554/src1\n"
                   "    - id: cam-2\n"
                   "      rtsp_url: rtsp://127.0.0.1:8554/src2\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "over-max-streams config is rejected") &&
                  expect_contains(result.stderr_text, "max_streams",
                                  "over-max-streams error mentions max_streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_rejects_empty_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_rejects_empty_streams",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-int8-b1.tar.gz\n"
                   "streams: []\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "empty stream config is rejected") &&
                  expect_contains(result.stderr_text, "streams", "empty-stream error mentions streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

} // namespace

int main(int argc, char** argv) {
  bool ok = true;

  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];
  ok &= test_help_runs(binary);
  ok &= test_rejects_unknown_mode(binary);
  ok &= test_mode_adaptive_validates(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_rich_sources(binary);
  ok &= test_validate_bare_list(binary);
  ok &= test_validate_rejects_over_max_streams(binary);
  ok &= test_validate_rejects_empty_streams(binary);
  return ok ? 0 : 1;
}
