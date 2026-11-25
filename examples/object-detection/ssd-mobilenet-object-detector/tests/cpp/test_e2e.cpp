// E2E test: run the binary and validate annotated images plus decoded detections.
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

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

// An empty app.detections_json turns the report off: the run still saves one PNG per image and
// writes no JSON file.
bool test_empty_detections_json_writes_no_report(const std::string& binary,
                                                 const std::string& model_path,
                                                 const std::string& input_dir, int timeout) {
  const std::string out_dir = create_test_output_dir("ssd-mobilenet-object-detector",
                                                     "test_empty_detections_json_writes_no_report");
  if (out_dir.empty()) {
    return false;
  }
  const fs::path run_dir = fs::path(out_dir).parent_path();
  const fs::path config_path = run_dir / "config.yaml";
  write_e2e_config("ssd-mobilenet-object-detector", config_path,
                   {{"model.path", model_path},
                    {"input.source", input_dir},
                    {"output.dir", out_dir},
                    {"app.detections_json", ""}});
  const auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  std::string problem;
  if (r.exit_code != 0) {
    problem = "exit code " + std::to_string(r.exit_code) + "\nstderr:\n" + r.stderr_text;
  } else {
    problem = saved_frames_problem(out_dir, supported_image_count(input_dir));
  }
  for (const auto& entry : fs::recursive_directory_iterator(run_dir)) {
    if (problem.empty() && entry.path().extension() == ".json") {
      problem = "an empty app.detections_json wrote " + entry.path().string();
    }
  }
  if (problem.empty() && r.stdout_text.find("detections in") != std::string::npos) {
    problem = "an empty app.detections_json still names a report:\n" + r.stdout_text;
  }
  remove_dir(out_dir);
  if (!problem.empty()) {
    std::cerr << "[FAIL] " << problem << "\n";
    return false;
  }
  std::cout << "[OK] an empty app.detections_json wrote no report\n";
  return true;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERR] usage: " << argv[0] << " <example-binary>\n";
    return 2;
  }
  const std::string binary = argv[1];

  const char* models_dir_raw = env_or_null("SIMANEAT_APPS_TEST_MODELS_DIR");
  // The application reads a relative model.path from the config file's folder.
  const std::string models_dir = fs::absolute(models_dir_raw ? models_dir_raw : "models").string();

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
                    {"input.source", input_dir},
                    {"output.dir", out_dir},
                    {"app.detections_json", detections_path.string()}});
  int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);

  if (r.exit_code != 0) {
    std::cerr << "[FAIL] exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const int output_files = count_output_files(out_dir);
  const std::string output_problem =
      saved_frames_problem(out_dir, supported_image_count(input_dir));
  if (!output_problem.empty()) {
    std::cerr << "[FAIL] " << output_problem << "\n";
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

  const auto config = sima_examples::Config::load(config_path);
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  const double min_score = config.number("inference.min_score");
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
        // Class 0 is the background, which the SSD decoder never reports.
        if (!(0.0 <= x1 && x1 < x2 && x2 <= width && 0.0 <= y1 && y1 < y2 && y2 <= height &&
              min_score <= score && score <= 1.0 && class_id > 0 &&
              class_id < static_cast<int>(labels.size()) &&
              detection.at("label").get<std::string>() == labels[static_cast<size_t>(class_id)])) {
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

  const bool report_off_ok =
      test_empty_detections_json_writes_no_report(binary, model_path, input_dir, timeout);
  return report_off_ok ? 0 : 1;
}
