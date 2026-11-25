// E2E test for depth-estimator.
// Runs the binary with each scoped model and the test images, and verifies it writes one
// side-by-side PNG per image whose depth half is a spatially coherent depth map that agrees with
// the FP32 reference depth of the same image.
#include "support/testing/source_cases.h"
#include "support/testing/test_config.h"
#include "support/testing/test_process.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "depth-estimator";
// Every depth_anything_v2_small variant in tests/test-scope.yaml predicts a 518x686 (height x
// width) depth map, and each saved PNG puts the resized input beside the depth colormap.
constexpr int kModelHeight = 518;
constexpr int kModelWidth = 686;

// fixtures/reference_depth/<image>.png holds the FP32 depth of each assets/datasets-test/coco
// image, from the upstream ONNX the Model Registry compiles (onnx-community/depth-anything-v2-small
// at 4472b736, sha256 afb6a5c2...), fed the registry's preprocessing: RGB / 255, cv2 INTER_CUBIC
// stretch to 686x518, ImageNet mean and std. Each map is shrunk with INTER_AREA to 172x130 and
// stored min-max scaled as 16-bit PNG. The Python e2e test applies the same checks.
constexpr int kReferenceWidth = 172;
constexpr int kReferenceHeight = 130;
// The test images are letterboxed. The model calls the black bars nearest, which would dominate
// the correlation, so only the image content is compared: pixels brighter than this gray level.
constexpr int kContentThreshold = 16;
// On the host, the FP32 model through the app's 8-bit colormap correlates at 1.00 with the
// reference, and bilinear instead of bicubic resizing or a square 518x518 input still gives 0.99
// or more. Feeding raw 0-255 pixels without normalization peaks at 0.46, and reading a row-major
// map column-major at 0.04, so these floors pass quantized builds and fail the wrong input or
// layout.
constexpr double kMinImageCorrelation = 0.8;
constexpr double kMinMeanCorrelation = 0.9;
// A depth map changes gradually between neighbouring pixels. Reading the model output with the
// wrong layout or stride scrambles it, which raises the mean neighbour difference to about half
// the spread of the values or more; a real depth map stays near 0.02.
constexpr double kMaxNeighbourDifferenceRatio = 0.1;
constexpr double kMinDepthSpread = 10.0;

fs::path reference_dir() {
  // example_common_config_path() is <example>/src/common/config.yaml.
  return example_common_config_path(kExampleName).parent_path().parent_path().parent_path() /
         "tests" / "fixtures" / "reference_depth";
}

uint32_t color_key(const cv::Vec3b& bgr) {
  return (static_cast<uint32_t>(bgr[0]) << 16) | (static_cast<uint32_t>(bgr[1]) << 8) | bgr[2];
}

// Recovers the 8-bit depth the app colored from the right half of its saved PNG, or returns the
// problem.
std::string colormap_depth(const fs::path& path, cv::Mat& depth) {
  const std::string name = path.filename().string();
  const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    return name + " does not decode";
  }
  if (image.rows != kModelHeight || image.cols != 2 * kModelWidth) {
    std::ostringstream out;
    out << name << " is " << image.cols << "x" << image.rows << ", expected the input and a "
        << kModelWidth << "x" << kModelHeight << " depth map side by side";
    return out.str();
  }

  cv::Mat ramp(1, 256, CV_8UC1);
  for (int i = 0; i < 256; ++i) {
    ramp.at<uint8_t>(0, i) = static_cast<uint8_t>(i);
  }
  cv::Mat lut;
  cv::applyColorMap(ramp, lut, cv::COLORMAP_INFERNO);
  std::unordered_map<uint32_t, float> level_of;
  for (int i = 0; i < 256; ++i) {
    level_of.emplace(color_key(lut.at<cv::Vec3b>(0, i)), static_cast<float>(i));
  }

  const cv::Mat colored = image(cv::Rect(kModelWidth, 0, kModelWidth, kModelHeight));
  depth.create(kModelHeight, kModelWidth, CV_32FC1);
  for (int y = 0; y < kModelHeight; ++y) {
    for (int x = 0; x < kModelWidth; ++x) {
      const auto it = level_of.find(color_key(colored.at<cv::Vec3b>(y, x)));
      if (it == level_of.end()) {
        return name + " depth half is not an INFERNO colormap";
      }
      depth.at<float>(y, x) = it->second;
    }
  }
  return "";
}

std::string coherence_problem(const std::string& name, const cv::Mat& depth) {
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

// Pearson correlation of the app's depth with the FP32 reference over the image content, or the
// problem. Leaves has_reference false, with no problem, when this image has no reference depth:
// the fixtures cover the bundled assets/datasets-test/coco images, and a custom
// SIMANEAT_APPS_TEST_INPUT_DIR holds other images, to which only the coherence checks apply. A
// reference that is present but not the expected size is a broken fixture, and is a problem.
std::string reference_correlation(const fs::path& image_path, const cv::Mat& depth,
                                  double& correlation, bool& has_reference) {
  has_reference = false;
  const fs::path reference_path = reference_dir() / (image_path.stem().string() + ".png");
  if (!fs::is_regular_file(reference_path)) {
    return "";
  }
  const cv::Mat reference = cv::imread(reference_path.string(), cv::IMREAD_UNCHANGED);
  if (reference.empty() || reference.rows != kReferenceHeight ||
      reference.cols != kReferenceWidth) {
    std::ostringstream out;
    out << reference_path.string() << " is not a " << kReferenceWidth << "x" << kReferenceHeight
        << " reference depth map";
    return out.str();
  }

  const cv::Mat gray = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
  if (gray.empty()) {
    return image_path.filename().string() + " does not decode";
  }
  std::vector<cv::Point> content;
  cv::findNonZero(gray > kContentThreshold, content);
  const cv::Rect box = cv::boundingRect(content);
  // Shrink the content box onto the reference grid, rounding inward so no bar pixels remain.
  const int x0 = (box.x * kReferenceWidth + gray.cols - 1) / gray.cols;
  const int x1 = (box.x + box.width) * kReferenceWidth / gray.cols;
  const int y0 = (box.y * kReferenceHeight + gray.rows - 1) / gray.rows;
  const int y1 = (box.y + box.height) * kReferenceHeight / gray.rows;
  if (x1 <= x0 || y1 <= y0) {
    return image_path.filename().string() + " has no content to compare";
  }

  cv::Mat small;
  cv::resize(depth, small, cv::Size(kReferenceWidth, kReferenceHeight), 0.0, 0.0, cv::INTER_AREA);
  const cv::Rect crop(x0, y0, x1 - x0, y1 - y0);
  cv::Mat a;
  cv::Mat b;
  small(crop).convertTo(a, CV_64F);
  reference(crop).convertTo(b, CV_64F);
  a -= cv::mean(a)[0];
  b -= cv::mean(b)[0];
  correlation = a.dot(b) / std::sqrt(a.dot(a) * b.dot(b));
  has_reference = true;
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

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (entry.is_regular_file() &&
        (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());

  std::ostringstream report;
  report << std::fixed << std::setprecision(4);
  double total = 0.0;
  double lowest = 1.0;
  int compared = 0;
  std::string problem;
  for (const fs::path& image_path : images) {
    const fs::path output = fs::path(out_dir) / (image_path.stem().string() + ".png");
    cv::Mat depth;
    problem = colormap_depth(output, depth);
    if (problem.empty()) {
      problem = coherence_problem(output.filename().string(), depth);
    }
    double correlation = 0.0;
    bool has_reference = false;
    if (problem.empty()) {
      problem = reference_correlation(image_path, depth, correlation, has_reference);
    }
    if (!problem.empty()) {
      break;
    }
    if (!has_reference) {
      continue;
    }
    report << " " << image_path.filename().string() << "=" << correlation;
    total += correlation;
    ++compared;
    lowest = std::min(lowest, correlation);
    if (correlation < kMinImageCorrelation) {
      std::ostringstream out;
      out << "depth map for " << image_path.filename().string()
          << " disagrees with the FP32 reference: correlation " << correlation << " < "
          << kMinImageCorrelation;
      problem = out.str();
      break;
    }
  }
  remove_dir(out_dir);
  if (problem.empty() && images.empty()) {
    problem = "no input images to compare";
  }
  if (!problem.empty()) {
    std::cerr << "[FAIL] " << label << ": " << problem << "\n";
    return 1;
  }

  // A custom SIMANEAT_APPS_TEST_INPUT_DIR can hold images with no reference depth. The coherence
  // checks above still ran on every one; only the FP32 floors need a reference.
  if (compared == 0) {
    std::cout << "[DEPTH] " << fs::path(model_path).filename().string() << ": " << images.size()
              << " coherent depth maps. None of these images has a reference depth in "
              << reference_dir().filename().string() << ", so the FP32 comparison was skipped\n";
    std::cout << "[OK] " << label << ": " << images.size() << " coherent depth maps\n";
    return 0;
  }

  const double mean = total / static_cast<double>(compared);
  std::cout << "[DEPTH] " << fs::path(model_path).filename().string()
            << ": correlation with the FP32 reference mean=" << std::fixed << std::setprecision(4)
            << mean << " min=" << lowest << report.str() << "\n";
  if (mean < kMinMeanCorrelation) {
    std::cerr << "[FAIL] " << label << ": mean correlation with the FP32 reference " << mean
              << " < " << kMinMeanCorrelation << "\n";
    return 1;
  }
  std::cout << "[OK] " << label << ": " << compared << " of " << images.size()
            << " coherent depth maps agree with the FP32 reference\n";
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

  // The fixtures are committed, so a missing set is a broken checkout rather than a custom image
  // directory. Per-image absence is handled in reference_correlation.
  bool any_reference = false;
  if (fs::is_directory(reference_dir())) {
    for (const auto& entry : fs::directory_iterator(reference_dir())) {
      if (entry.is_regular_file() && entry.path().extension() == ".png") {
        any_reference = true;
        break;
      }
    }
  }
  if (!any_reference) {
    std::cerr << "[FAIL] no reference depth fixtures at " << reference_dir().string() << "\n";
    return 1;
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
