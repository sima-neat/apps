// E2E test for depth-estimator.
// Runs the binary with each scoped model and the test images, and verifies it writes one
// side-by-side PNG per image whose depth half is a spatially coherent depth map.
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "depth-estimator";
// Every depth_anything_v2_small variant in tests/test-scope.yaml predicts a 518x686 (height x
// width) depth map, and each saved PNG puts the resized input beside the depth colormap.
constexpr int kModelHeight = 518;
constexpr int kModelWidth = 686;
// A depth map changes gradually between neighbouring pixels. Reading the model output with the
// wrong layout or stride scrambles it, which raises the mean neighbour difference to about half
// the spread of the values or more; a real depth map stays near 0.02.
constexpr double kMaxNeighbourDifferenceRatio = 0.1;
constexpr double kMinDepthSpread = 10.0;

// Returns an empty string when the saved PNG holds a coherent depth map, else the problem.
std::string depth_map_problem(const fs::path& path) {
  const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  const std::string name = path.filename().string();
  if (image.empty()) {
    return name + " does not decode";
  }
  if (image.rows != kModelHeight || image.cols != 2 * kModelWidth) {
    std::ostringstream out;
    out << name << " is " << image.cols << "x" << image.rows << ", expected the input and a "
        << kModelWidth << "x" << kModelHeight << " depth map side by side";
    return out.str();
  }

  cv::Mat depth;
  cv::cvtColor(image(cv::Rect(kModelWidth, 0, kModelWidth, kModelHeight)), depth,
               cv::COLOR_BGR2GRAY);
  depth.convertTo(depth, CV_32F);
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(depth, mean, stddev);
  const double spread = stddev[0];
  if (spread < kMinDepthSpread) {
    std::ostringstream out;
    out << name << " has a flat depth map (std " << spread << ")";
    return out.str();
  }

  cv::Mat across;
  cv::Mat down;
  cv::absdiff(depth.colRange(1, depth.cols), depth.colRange(0, depth.cols - 1), across);
  cv::absdiff(depth.rowRange(1, depth.rows), depth.rowRange(0, depth.rows - 1), down);
  const double neighbour = std::max(cv::mean(across)[0], cv::mean(down)[0]);
  if (neighbour / spread > kMaxNeighbourDifferenceRatio) {
    std::ostringstream out;
    out << name << " depth map is not spatially coherent: neighbour difference " << neighbour
        << " is " << neighbour / spread << " of its spread " << spread;
    return out.str();
  }
  return "";
}

int run_model_case(const std::string& binary, const std::string& model_path,
                   const std::string& input_dir, const std::string& label) {
  const std::string out_dir = create_test_output_dir(kExampleName, label);
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(out_dir).parent_path() / (label + "_config.yaml");
  write_e2e_config(
      kExampleName, config_path,
      {{"model.path", model_path}, {"io.input_dir", input_dir}, {"io.output_dir", out_dir}});
  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 180000);

  const auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] " << label << ": exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const std::string output_problem =
      saved_frames_problem(out_dir, supported_image_count(input_dir));
  if (!output_problem.empty()) {
    std::cerr << "[FAIL] " << label << ": " << output_problem << "\n";
    remove_dir(out_dir);
    return 1;
  }

  std::vector<fs::path> outputs;
  for (const auto& entry : fs::directory_iterator(out_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".png") {
      outputs.push_back(entry.path());
    }
  }
  std::sort(outputs.begin(), outputs.end());
  for (const fs::path& output : outputs) {
    const std::string problem = depth_map_problem(output);
    if (!problem.empty()) {
      std::cerr << "[FAIL] " << label << ": " << problem << "\n";
      remove_dir(out_dir);
      return 1;
    }
  }

  remove_dir(out_dir);
  std::cout << "[OK] " << label << ": " << outputs.size() << " coherent depth maps\n";
  return 0;
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

  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("depth model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* images_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = images_raw ? images_raw : "assets/datasets-test/coco";
  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory with test images (assets/datasets-test/coco is empty or missing)");
    return kSkipCode;
  }

  for (const std::string& model_path : model_paths) {
    const std::string label =
        model_case_label("test_full_pipeline", model_path, model_paths.size());
    if (run_model_case(binary, model_path, input_dir, label) != 0) {
      rc = 1;
    }
  }
  return rc;
}
