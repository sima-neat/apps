// Starter e2e smoke test for multi-stream-object-detector.
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using sima_examples::testing::create_test_output_dir;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  const std::string temp_dir = create_test_output_dir("multi-stream-object-detector",
                                                      "test_validate_config_only_smoke_runs");
  if (temp_dir.empty()) {
    std::cerr << "[FAIL] failed to create temp directory\n";
    return 1;
  }

  const std::filesystem::path config_path =
      std::filesystem::path(temp_dir).parent_path() / "config.yaml";
  std::ofstream out(config_path);
  out << "model:\n"
         "  path: assets/models/yolo_v8m_mpk.tar.gz\n"
         "streams:\n"
         "  - rtsp://127.0.0.1:8554/src1\n"
         "runtime:\n"
         "  worker_count: 2\n"
         "output:\n"
         "  insight:\n"
         "    host: 127.0.0.1\n";
  out.close();

  const auto r =
      spawn_and_wait(binary, {"--config", config_path.string(), "--validate-config-only"}, 10000);
  remove_dir(temp_dir);
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] starter smoke expected exit 0, got " << r.exit_code << "\n";
    if (!r.stderr_text.empty()) {
      std::cerr << r.stderr_text << "\n";
    }
    return 1;
  }
  return 0;
}
