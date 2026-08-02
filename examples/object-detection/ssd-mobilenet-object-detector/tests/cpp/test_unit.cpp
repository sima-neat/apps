// Unit test for ssd-mobilenet-object-detector: CLI arg handling and decode config validation.
#include "support/testing/test_process.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using sima_examples::testing::create_test_scratch_dir;
using sima_examples::testing::ProcessResult;
using sima_examples::testing::remove_dir;
using sima_examples::testing::spawn_and_wait;

namespace {

// Write a config whose decode section carries the supplied value for `key`.
fs::path write_decode_config(const fs::path& dir, const std::string& name, const std::string& key,
                             const std::string& value) {
  const fs::path path = dir / name;
  std::ofstream out(path);
  out << "model:\n"
      << "  path: /nonexistent/ssd_model.tar.gz\n"
      << "io:\n"
      << "  input_dir: assets/datasets/coco\n"
      << "  output_dir: sandbox/ssd-mobilenet-object-detector\n"
      << "decode:\n"
      << "  " << key << ": " << value << "\n";
  return path;
}

bool expect_config_rejected(const std::string& binary, const fs::path& config,
                            const std::string& expected_message, const std::string& label) {
  ProcessResult r = spawn_and_wait(binary, {"--config", config.string()}, 20000);
  if (r.exit_code != 2) {
    std::cerr << "[FAIL] " << label << ": expected exit 2, got " << r.exit_code << "\n";
    return false;
  }
  if (r.stderr_text.find(expected_message) == std::string::npos) {
    std::cerr << "[FAIL] " << label << ": stderr does not mention '" << expected_message
              << "'\nstderr: " << r.stderr_text << "\n";
    return false;
  }
  std::cout << "[OK] " << label << " correctly rejected\n";
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }

  const std::string binary = argv[1];
  int failures = 0;

  {
    ProcessResult r = spawn_and_wait(binary, {"--help"}, 20000);
    if (r.exit_code != 0) {
      std::cerr << "[FAIL] help: expected exit 0, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stdout_text.find("Usage") == std::string::npos) {
      std::cerr << "[FAIL] help: stdout does not contain usage hint\n";
      ++failures;
    } else {
      std::cout << "[OK] help prints usage\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--bogus"}, 20000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] unknown flag: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("unknown argument") == std::string::npos) {
      std::cerr << "[FAIL] unknown flag: stderr does not mention unknown argument\n";
      ++failures;
    } else {
      std::cout << "[OK] unknown flag correctly rejected\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--config"}, 20000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] missing config path: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("--config requires a path") == std::string::npos) {
      std::cerr << "[FAIL] missing config path: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] missing config path correctly rejected\n";
    }
  }

  {
    ProcessResult r = spawn_and_wait(binary, {"--config", "/nonexistent/ssd-config.yaml"}, 20000);
    if (r.exit_code != 2) {
      std::cerr << "[FAIL] bad config: expected exit 2, got " << r.exit_code << "\n";
      ++failures;
    } else if (r.stderr_text.find("failed to open config") == std::string::npos) {
      std::cerr << "[FAIL] bad config: stderr does not explain failure\n";
      ++failures;
    } else {
      std::cout << "[OK] bad config path correctly rejected\n";
    }
  }

  // The decode section drives the model-managed BoxDecode stage, so out-of-range values must be
  // rejected before the pipeline is built rather than silently forwarded to the kernel.
  const std::string scratch =
      create_test_scratch_dir("ssd-mobilenet-object-detector", "test_decode_config");
  if (scratch.empty()) {
    std::cerr << "[FAIL] could not create scratch directory for config tests\n";
    return 1;
  }
  const fs::path scratch_dir(scratch);

  if (!expect_config_rejected(
          binary, write_decode_config(scratch_dir, "score.yaml", "score_threshold", "1.5"),
          "decode.score_threshold must be in [0.0, 1.0]", "out-of-range score_threshold")) {
    ++failures;
  }
  if (!expect_config_rejected(binary,
                              write_decode_config(scratch_dir, "iou.yaml", "nms_iou", "-0.1"),
                              "decode.nms_iou must be in [0.0, 1.0]", "out-of-range nms_iou")) {
    ++failures;
  }
  if (!expect_config_rejected(
          binary, write_decode_config(scratch_dir, "topk.yaml", "max_detections", "0"),
          "decode.max_detections must be >= 1", "non-positive max_detections")) {
    ++failures;
  }

  {
    const fs::path model = scratch_dir / "model.tar.gz";
    const fs::path input_dir = scratch_dir / "input";
    const fs::path image = input_dir / "frame.jpg";
    fs::create_directories(input_dir);
    std::ofstream(model).put('\n');
    std::ofstream(image).put('\n');
    const fs::path config = scratch_dir / "detections_alias.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << (scratch_dir / "output").string() << "\n"
        << "  detections_json: " << image.string() << "\n"
        << "output:\n"
        << "  overlay: false\n";
    out.close();
    if (!expect_config_rejected(binary, config,
                                "io.detections_json must not overwrite an input image",
                                "input-image detections report alias")) {
      ++failures;
    }
  }

  {
    const fs::path model = scratch_dir / "model.tar.gz";
    const fs::path input_dir = scratch_dir / "input";
    const fs::path missing_labels = scratch_dir / "custom" / "coco_labels.txt";
    const fs::path config = scratch_dir / "missing_custom_labels.yaml";
    std::ofstream out(config);
    out << "model:\n"
        << "  path: " << model.string() << "\n"
        << "  labels: " << missing_labels.string() << "\n"
        << "io:\n"
        << "  input_dir: " << input_dir.string() << "\n"
        << "  output_dir: " << (scratch_dir / "output").string() << "\n";
    out.close();
    if (!expect_config_rejected(binary, config,
                                "labels file does not exist: " + missing_labels.string(),
                                "missing custom coco_labels path")) {
      ++failures;
    }
  }

  remove_dir(scratch);
  return failures > 0 ? 1 : 0;
}
