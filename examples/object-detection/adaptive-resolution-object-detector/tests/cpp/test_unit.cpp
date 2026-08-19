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
#include "../../src/cpp/adaptive_policy.h"

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

// ── pure policy / budget (no binary, no hardware) ────────────────────────────

bool test_tier_cost() {
  return expect_true(std::abs(adaptive::tier_cost(320, 320) - 1.0) < 1e-9, "tier_cost 320 == 1") &&
         expect_true(std::abs(adaptive::tier_cost(640, 320) - 4.0) < 1e-9, "tier_cost 640 == 4") &&
         expect_true(std::abs(adaptive::tier_cost(960, 320) - 9.0) < 1e-9, "tier_cost 960 == 9");
}

bool test_budget_degrades_with_stream_count() {
  const std::vector<int> res{320, 640, 960};
  const double budget = 12.0;
  return expect_true(adaptive::budget_allowed_index(1, res, budget) == 2,
                     "1 stream may use the top tier") &&
         expect_true(adaptive::budget_allowed_index(2, res, budget) == 1,
                     "2 streams are capped to the middle tier") &&
         expect_true(adaptive::budget_allowed_index(4, res, budget) == 0,
                     "4 streams are capped to the lowest tier") &&
         expect_true(adaptive::budget_allowed_index(99, res, budget) == 0,
                     "the lowest tier is always allowed");
}

bool test_frame_stats() {
  const std::vector<adaptive::DetBox> boxes{{10.0f, 50.0f, 0.9f}, {200.0f, 200.0f, 0.5f}};
  const auto stats = adaptive::frame_stats(boxes, 0.3f);
  const auto empty = adaptive::frame_stats({}, 0.3f);
  return expect_true(stats.object_count == 2, "frame_stats counts kept boxes") &&
         expect_true(std::abs(stats.min_object_px - 10.0f) < 1e-6, "frame_stats finds smallest side") &&
         expect_true(std::abs(stats.min_confidence - 0.5f) < 1e-6, "frame_stats finds min confidence") &&
         expect_true(empty.object_count == 0 && empty.min_confidence == 1.0f,
                     "empty scene reads as high confidence / no objects");
}

bool test_hysteresis_holds_then_commits() {
  adaptive::PolicyConfig cfg;
  cfg.hysteresis_frames = 3;
  adaptive::PolicyState state{1, 1, 0}; // start at 640
  adaptive::FrameStats small; // a tiny object => wants to step up
  small.object_count = 1;
  small.min_object_px = 10.0f;
  small.min_confidence = 0.9f;

  const int f1 = adaptive::select_tier(state, small, cfg);
  const int f2 = adaptive::select_tier(state, small, cfg);
  const int f3 = adaptive::select_tier(state, small, cfg);
  return expect_true(f1 == 1 && f2 == 1, "tier holds during hysteresis window") &&
         expect_true(f3 == 2, "tier commits up once desire persists hysteresis_frames");
}

bool test_hysteresis_resets_on_flip() {
  adaptive::PolicyConfig cfg;
  cfg.hysteresis_frames = 3;
  adaptive::PolicyState state{1, 1, 0};
  adaptive::FrameStats small;
  small.object_count = 1;
  small.min_object_px = 10.0f;
  small.min_confidence = 0.9f;
  adaptive::FrameStats steady; // big, confident, sparse => wants current
  steady.object_count = 1;
  steady.min_object_px = 300.0f;
  steady.min_confidence = 0.9f;

  adaptive::select_tier(state, small, cfg);  // vote up (count 1)
  adaptive::select_tier(state, steady, cfg); // desire == current => resets the vote
  const int after = adaptive::select_tier(state, small, cfg);
  return expect_true(after == 1 && state.pending_count == 1,
                     "a non-matching frame resets the hysteresis vote");
}

bool test_step_down_on_easy_scene() {
  adaptive::PolicyConfig cfg;
  cfg.hysteresis_frames = 2;
  adaptive::PolicyState state{2, 2, 0}; // start at 960
  adaptive::FrameStats empty = adaptive::frame_stats({}, 0.3f);
  const int f1 = adaptive::select_tier(state, empty, cfg);
  const int f2 = adaptive::select_tier(state, empty, cfg);
  return expect_true(f1 == 2, "tier holds one frame before stepping down") &&
         expect_true(f2 == 1, "easy scene steps the tier down after hysteresis");
}

bool test_effective_tier_clamped_by_budget() {
  adaptive::PolicyConfig cfg;
  cfg.hysteresis_frames = 1;
  adaptive::PolicyState state{2, 2, 0}; // content already at the top tier
  adaptive::FrameStats crowded;
  crowded.object_count = 40; // wants to stay/step up
  crowded.min_object_px = 10.0f;
  crowded.min_confidence = 0.2f;
  const int eff = adaptive::effective_tier(state, crowded, cfg, 8, 12.0); // 8 streams share budget
  return expect_true(eff == 0, "budget clamps a hot stream to the lowest tier when crowded");
}

} // namespace

int main(int argc, char** argv) {
  bool ok = true;

  // Pure logic first: these do not need the example binary.
  ok &= test_tier_cost();
  ok &= test_budget_degrades_with_stream_count();
  ok &= test_frame_stats();
  ok &= test_hysteresis_holds_then_commits();
  ok &= test_hysteresis_resets_on_flip();
  ok &= test_step_down_on_easy_scene();
  ok &= test_effective_tier_clamped_by_budget();

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
