// Unit tests for adaptive-resolution-object-detector (C++).
// Drives the built binary's command line and config validation; no model or
// stream is opened.
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::example_common_config_path;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

constexpr const char* kExampleName = "adaptive-resolution-object-detector";

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
  const bool found = haystack.find(needle) != std::string::npos;
  if (!found) {
    std::cerr << "[FAIL] " << message << " (expected '" << needle << "' in: " << haystack << ")\n";
    return false;
  }
  std::cout << "[OK] " << message << "\n";
  return true;
}

struct ConfigParts {
  std::string mode = "live";
  std::string source = "source:\n  type: rtsp\n  codec: h264\n";
  std::string streams = "streams:\n"
                        "  max_streams: 16\n"
                        "  sources:\n"
                        "    - id: cam-1\n"
                        "      url: rtsp://127.0.0.1:8554/src1\n"
                        "    - id: cam-2\n"
                        "      url: \"rtsp://127.0.0.1:8554/src2\"  # quoted, with a comment\n";
  std::string extra;
};

std::string config_text(const ConfigParts& parts) {
  std::string text;
  if (!parts.mode.empty()) {
    text += "mode: " + parts.mode + "\n";
  }
  text += "model:\n  path: models/yolo26n-det-int8-b1.tar.gz\n";
  text += parts.source;
  text += parts.streams;
  text += parts.extra;
  text += "output:\n  insight:\n    host: 127.0.0.1\n";
  return text;
}

fs::path write_config(const std::string& test_name, const std::string& body) {
  const std::string temp_dir = create_test_scratch_dir(kExampleName, test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

struct Validation {
  int exit_code = -1;
  std::string out;
  std::string err;
};

Validation validate(const std::string& binary, const std::string& test_name,
                    const ConfigParts& parts) {
  const fs::path config_path = write_config(test_name, config_text(parts));
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  return {result.exit_code, result.stdout_text, result.stderr_text};
}

bool expect_rejected(const std::string& binary, const std::string& test_name,
                     const ConfigParts& parts, const std::string& message) {
  const Validation result = validate(binary, test_name, parts);
  return expect_true(result.exit_code == 1, test_name + " exits with code 1") &&
         expect_contains(result.err, message, test_name + " names the problem");
}

bool test_help_lists_config_options_and_no_mode_flag(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only") &&
         expect_true(result.stdout_text.find("--mode") == std::string::npos,
                     "help does not offer a --mode flag");
}

bool test_mode_flag_is_rejected(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--mode", "scale"}, 20000);
  return expect_true(result.exit_code == 1, "--mode exits with code 1") &&
         expect_contains(result.stderr_text, "unknown argument: --mode",
                         "--mode is reported as an unknown argument");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_reports_each_mode(const std::string& binary) {
  bool ok = true;
  for (const std::string mode : {"live", "scale", "group", "LIVE"}) {
    ConfigParts parts;
    parts.mode = mode;
    const Validation result = validate(binary, "validate_mode_" + mode, parts);
    std::string expected = mode == "LIVE" ? "live" : mode;
    ok &= expect_true(result.exit_code == 0, mode + " config validates") &&
          expect_contains(result.out, "mode=" + expected, mode + " validation reports the mode") &&
          expect_contains(result.out, "streams=2", mode + " validation reports two streams");
  }
  return ok;
}

bool test_packaged_config_validates(const std::string& binary) {
  // A scratch copy: the test harness writes its process logs next to --config.
  std::ifstream packaged(example_common_config_path(kExampleName));
  std::stringstream body;
  body << packaged.rdbuf();
  const fs::path config_path = write_config("packaged_config", body.str());
  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  remove_dir(config_path.parent_path().string());
  return expect_true(result.exit_code == 0, "packaged config validates") &&
         expect_contains(result.stdout_text, "mode=live", "packaged config runs live mode");
}

bool test_mode_is_required(const std::string& binary) {
  ConfigParts missing;
  missing.mode.clear();
  ConfigParts unknown;
  unknown.mode = "fused";
  return expect_rejected(binary, "missing_mode", missing,
                         "mode must be set to one of: live, scale, group") &&
         expect_rejected(binary, "unknown_mode", unknown,
                         "mode must be one of: live, scale, group");
}

bool test_only_h264_rtsp_is_accepted(const std::string& binary) {
  bool ok = true;
  for (const std::string codec : {"avc", "H.264"}) {
    ConfigParts parts;
    parts.source = "source:\n  codec: " + codec + "\n";
    ok &= expect_true(validate(binary, "codec_alias", parts).exit_code == 0,
                      "codec alias " + codec + " is accepted");
  }
  for (const std::string codec : {"h265", "mjpeg"}) {
    ConfigParts parts;
    parts.source = "source:\n  codec: " + codec + "\n";
    ok &= expect_rejected(binary, "codec_" + codec, parts, "source.codec must be h264");
  }
  ConfigParts http;
  http.source = "source:\n  type: http\n";
  ok &= expect_rejected(binary, "source_type_http", http, "source.type must be rtsp");
  return ok;
}

bool test_stream_list_rules(const std::string& binary) {
  ConfigParts bare;
  bare.streams = "streams:\n  - rtsp://127.0.0.1:8554/src1\n";
  ConfigParts scalar_entry;
  scalar_entry.streams = "streams:\n  sources:\n    - rtsp://127.0.0.1:8554/src1\n";
  ConfigParts missing_id;
  missing_id.streams = "streams:\n  sources:\n    - url: rtsp://127.0.0.1:8554/src1\n";
  ConfigParts missing_url;
  missing_url.streams = "streams:\n  sources:\n    - id: cam-1\n      rtsp_url: rtsp://a/1\n";
  ConfigParts duplicate;
  duplicate.streams = "streams:\n  sources:\n"
                      "    - id: cam-1\n      url: rtsp://a/1\n"
                      "    - id: cam-1\n      url: rtsp://a/2\n";
  ConfigParts too_many;
  too_many.streams = "streams:\n  max_streams: 1\n  sources:\n"
                     "    - id: cam-1\n      url: rtsp://a/1\n"
                     "    - id: cam-2\n      url: rtsp://a/2\n";
  return expect_rejected(binary, "bare_stream_list", bare, "streams must be a mapping") &&
         expect_rejected(binary, "scalar_source_entry", scalar_entry,
                         "streams.sources[0] must be a mapping with id and url") &&
         expect_rejected(binary, "source_missing_id", missing_id,
                         "streams.sources[0].id must be set") &&
         expect_rejected(binary, "source_missing_url", missing_url,
                         "streams.sources[0].url must be set") &&
         expect_rejected(binary, "duplicate_ids", duplicate, "ids must be unique") &&
         expect_rejected(binary, "more_than_max_streams", too_many,
                         "more than streams.max_streams (1)");
}

bool test_empty_stream_list_depends_on_mode(const std::string& binary) {
  bool ok = true;
  ConfigParts live;
  live.streams = "streams:\n  sources: []\n";
  const Validation live_result = validate(binary, "live_empty_sources", live);
  ok &= expect_true(live_result.exit_code == 0, "live mode may start with no streams") &&
        expect_contains(live_result.out, "streams=0", "live validation reports zero streams");
  for (const std::string mode : {"scale", "group"}) {
    ConfigParts parts;
    parts.mode = mode;
    parts.streams = "streams:\n  sources: []\n";
    ok &= expect_rejected(binary, mode + "_empty_sources", parts,
                          "streams.sources must not be empty in " + mode + " mode");
  }
  return ok;
}

bool test_invalid_values_are_rejected(const std::string& binary) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"group:\n  size: 0\n", "group.size must be >= 1"},
      {"inference:\n  min_score: 1.5\n", "inference.min_score must be between 0 and 1"},
      {"inference:\n  max_detections: 0\n", "inference.max_detections must be > 0"},
      {"inference:\n  fps: -1\n", "inference.fps must be >= 0"},
      {"runtime:\n  config_watch_seconds: 0\n", "runtime.config_watch_seconds must be > 0"},
      {"runtime:\n  warmup_frames: -1\n", "runtime.warmup_frames must be >= 0"},
  };
  bool ok = true;
  int index = 0;
  for (const auto& [extra, message] : cases) {
    ConfigParts parts;
    parts.extra = extra;
    ok &= expect_rejected(binary, "invalid_value_" + std::to_string(index++), parts, message);
  }
  ConfigParts negative_latency;
  negative_latency.source = "source:\n  latency_ms: -1\n";
  ok &= expect_rejected(binary, "negative_latency", negative_latency,
                        "source.latency_ms must be >= 0");
  return ok;
}

} // namespace

// A group worker is told its streams by the supervisor, not by config order. A
// malformed assignment must stop that worker with a clear error before it opens
// anything.
bool test_malformed_group_worker_assignment_is_rejected(const std::string& binary) {
  bool ok = true;
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"no_streams", "0:3:0:"},
      {"missing_url", "0:3:0:5:cam-1"},
      {"truncated_url", "0:3:4:5:cam-140:rtsp://127.0.0.1:8554/src1"},
      {"bad_header", "group-0"},
  };
  for (const auto& [name, role] : cases) {
    ConfigParts parts;
    parts.mode = "group";
    const fs::path config_path = write_config("group_worker_" + name, config_text(parts));
    setenv("ADAPTIVE_DETECTOR_GROUP_WORKER", role.c_str(), 1);
    const auto result = spawn_and_wait(binary, {"--config", config_path.string()}, 20000);
    unsetenv("ADAPTIVE_DETECTOR_GROUP_WORKER");
    remove_dir(config_path.parent_path().string());
    ok &= expect_true(result.exit_code == 1, "group worker " + name + " exits with code 1") &&
          expect_contains(result.stderr_text, "invalid ADAPTIVE_DETECTOR_GROUP_WORKER",
                          "group worker " + name + " names the problem");
  }
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_lists_config_options_and_no_mode_flag(binary);
  ok &= test_mode_flag_is_rejected(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_reports_each_mode(binary);
  ok &= test_packaged_config_validates(binary);
  ok &= test_mode_is_required(binary);
  ok &= test_only_h264_rtsp_is_accepted(binary);
  ok &= test_stream_list_rules(binary);
  ok &= test_empty_stream_list_depends_on_mode(binary);
  ok &= test_invalid_values_are_rejected(binary);
  ok &= test_malformed_group_worker_assignment_is_rejected(binary);
  return ok ? 0 : 1;
}
