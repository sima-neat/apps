#include "support/testing/test_process.h"
#include "support/runtime/example_utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

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
  const std::string temp_dir = create_test_scratch_dir("multi-stream-object-detector", test_name);
  if (temp_dir.empty()) {
    throw std::runtime_error("failed to create temp directory");
  }
  const fs::path config_path = fs::path(temp_dir) / "config.yaml";
  std::ofstream out(config_path);
  out << body;
  return config_path;
}

bool test_help_runs(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--help"}, 20000);
  return expect_true(result.exit_code == 0, "help exits with code 0") &&
         expect_contains(result.stdout_text, "--config", "help mentions --config") &&
         expect_contains(result.stdout_text, "--validate-config-only",
                         "help mentions --validate-config-only");
}

bool test_missing_config_file_fails_cleanly(const std::string& binary) {
  const auto result = spawn_and_wait(binary, {"--config", "does-not-exist.yaml"}, 20000);
  return expect_true(result.exit_code == 2, "missing config exits with code 2") &&
         expect_contains(result.stderr_text, "config file not found",
                         "missing config error mentions config file not found");
}

bool test_validate_config_only_accepts_four_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_accepts_four_streams",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "  - rtsp://127.0.0.1:8554/src2\n"
                   "  - rtsp://127.0.0.1:8554/src3\n"
                   "  - rtsp://127.0.0.1:8554/src4\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 0, "four-stream config validates") &&
      expect_contains(result.stdout_text, "streams=4", "validate output reports stream count");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_too_many_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_too_many_streams",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz\n"
                   "streams:\n"
                   "  - rtsp://127.0.0.1:8554/src1\n"
                   "  - rtsp://127.0.0.1:8554/src2\n"
                   "  - rtsp://127.0.0.1:8554/src3\n"
                   "  - rtsp://127.0.0.1:8554/src4\n"
                   "  - rtsp://127.0.0.1:8554/src5\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok = expect_true(result.exit_code == 1, "five-stream config is rejected") &&
                  expect_contains(result.stderr_text, "up to four streams",
                                  "too-many-stream error mentions four-stream phase limit");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_validate_config_only_rejects_empty_streams(const std::string& binary) {
  const fs::path config_path =
      write_config("test_validate_config_only_rejects_empty_streams",
                   "model:\n"
                   "  path: assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz\n"
                   "streams: []\n"
                   "output:\n"
                   "  insight:\n"
                   "    host: 127.0.0.1\n");

  const auto result =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 20000);
  const bool ok =
      expect_true(result.exit_code == 1, "empty stream config is rejected") &&
      expect_contains(result.stderr_text, "streams", "empty-stream error mentions streams");
  remove_dir(config_path.parent_path().string());
  return ok;
}

bool test_object_metadata_json_contract() {
  sima_examples::MetadataBox box;
  box.id = "obj_1";
  box.label = "person";
  box.confidence = 0.75F;
  box.x = 10.0F;
  box.y = 20.0F;
  box.w = 30.0F;
  box.h = 40.0F;

  const auto data =
      nlohmann::json::parse(sima_examples::metadata_boxes_data_json("objects", {box}));
  const auto& objects = data["objects"];
  const auto& object = objects[0];

  return expect_true(objects.is_array(), "metadata data contains objects array") &&
         expect_true(objects.size() == 1, "metadata data contains one object") &&
         expect_true(object["id"] == "obj_1", "metadata object has id") &&
         expect_true(object["label"] == "person", "metadata object has label") &&
         expect_true(object["confidence"].get<float>() == 0.75F,
                     "metadata object has confidence") &&
         expect_true(object["bbox"].is_array(), "metadata object has bbox array") &&
         expect_true(object["bbox"].size() == 4, "metadata bbox uses xywh shape") &&
         expect_true(object["bbox"][0].get<float>() == 10.0F, "metadata bbox has x") &&
         expect_true(object["bbox"][1].get<float>() == 20.0F, "metadata bbox has y") &&
         expect_true(object["bbox"][2].get<float>() == 30.0F, "metadata bbox has width") &&
         expect_true(object["bbox"][3].get<float>() == 40.0F, "metadata bbox has height");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  bool ok = true;
  ok &= test_help_runs(binary);
  ok &= test_missing_config_file_fails_cleanly(binary);
  ok &= test_validate_config_only_accepts_four_streams(binary);
  ok &= test_validate_config_only_rejects_too_many_streams(binary);
  ok &= test_validate_config_only_rejects_empty_streams(binary);
  ok &= test_object_metadata_json_contract();
  return ok ? 0 : 1;
}
