// E2E test: run the binary and validate annotated images plus decoded detections.
#include "support/testing/test_process.h"
#include "support/testing/test_config.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

nlohmann::json load_json(const fs::path& path, std::string& err) {
  std::ifstream in(path);
  if (!in) {
    err = "failed to open " + path.string();
    return {};
  }
  try {
    return nlohmann::json::parse(in);
  } catch (const std::exception& e) {
    err = "failed to parse " + path.string() + ": " + e.what();
    return {};
  }
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  const std::string models_dir = models_dir_raw ? models_dir_raw : "models";

  const std::string model_path = configured_model_path("ssd-mobilenet-object-detector", models_dir);

  if (model_path.empty() || !fs::exists(model_path)) {
    return skip_or_fail(
        "SSD-MobileNetV2 model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/datasets-test/coco";

  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory containing test images (defaults to assets/datasets-test/coco)");
    return kSkipCode;
  }

  const std::string out_dir =
      create_test_output_dir("ssd-mobilenet-object-detector", "test_full_pipeline");
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  const fs::path detections_path = run_dir / "detections.json";
  write_e2e_config("ssd-mobilenet-object-detector", config_path,
                   {{"model.path", model_path},
                    {"io.input_dir", input_dir},
                    {"io.output_dir", out_dir},
                    {"io.detections_json", detections_path.string()}});
  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const int output_files = count_output_files(out_dir);
  if (output_files == 0) {
    std::cerr << "[FAIL] expected annotated output images but output directory is empty\n";
    remove_dir(out_dir);
    return 1;
  }
  if (!all_output_files_nonempty(out_dir)) {
    std::cerr << "[FAIL] some annotated output images are empty\n";
    remove_dir(out_dir);
    return 1;
  }

  std::string err;
  const nlohmann::json reported = load_json(detections_path, err);
  if (!err.empty()) {
    std::cerr << "[FAIL] reported detections: " << err << "\n";
    remove_dir(out_dir);
    return 1;
  }

  int detection_count = 0;
  try {
    const auto& images = reported.at("images");
    if (!images.is_array() || static_cast<int>(images.size()) != output_files) {
      throw std::runtime_error("report image count does not match annotated output count");
    }
    for (const auto& image : images) {
      const int width = image.at("width").get<int>();
      const int height = image.at("height").get<int>();
      if (width <= 0 || height <= 0 || image.at("image").get<std::string>().empty()) {
        throw std::runtime_error("invalid image record");
      }
      for (const auto& detection : image.at("detections")) {
        const auto& box = detection.at("box");
        const double x1 = box.at(0).get<double>();
        const double y1 = box.at(1).get<double>();
        const double x2 = box.at(2).get<double>();
        const double y2 = box.at(3).get<double>();
        const double score = detection.at("score").get<double>();
        const int class_id = detection.at("class_id").get<int>();
        if (!(0.0 <= x1 && x1 < x2 && x2 <= width && 0.0 <= y1 && y1 < y2 && y2 <= height &&
              0.0 <= score && score <= 1.0 && class_id > 0 && class_id < 91 &&
              !detection.at("label").get<std::string>().empty())) {
          throw std::runtime_error("invalid detection record");
        }
        ++detection_count;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] detections report: " << e.what() << "\n";
    remove_dir(out_dir);
    return 1;
  }

  if (detection_count == 0) {
    std::cerr << "[FAIL] expected at least one decoded detection\n";
    remove_dir(out_dir);
    return 1;
  }

  remove_dir(out_dir);
  std::cout << "[OK] ssd-mobilenet-object-detector produced " << output_files
            << " output files and " << detection_count << " valid detections\n";
  return 0;
}
