// E2E test for detr-object-detector.
// Runs the binary with each scoped DETR model and a local image folder, and verifies it exits
// successfully, produces annotated output images, and detects objects in every image. A second
// case does the same for a photo larger than Neat's default 1920x1080 input limit.
#include "sima_examples/testing/source_cases.h"
#include "sima_examples/testing/test_config.h"
#include "sima_examples/testing/test_process.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sima_examples::testing;

namespace {

constexpr const char* kExampleName = "detr-object-detector";

int run_model_case(const std::string& binary, const std::string& model_path,
                   const std::string& input_dir, const std::string& label) {
  const std::string out_dir = create_test_output_dir(kExampleName, label);
  if (out_dir.empty()) {
    return 1;
  }

  const fs::path config_path = fs::path(out_dir).parent_path() / (label + "_config.yaml");
  write_e2e_config(
      kExampleName, config_path,
      {{"model.path", model_path}, {"input.source", input_dir}, {"output.dir", out_dir}});
  const int timeout = env_int_or_default("SIMANEAT_APPS_TEST_TIMEOUT_MS", 30000);

  const auto r = spawn_and_wait(binary, {"--config", config_path.string()}, timeout);
  if (r.exit_code != 0) {
    std::cerr << "[FAIL] " << label << ": exit code " << r.exit_code << "\n";
    std::cerr << "stderr:\n" << r.stderr_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  const int output_files = count_output_files(out_dir);
  const std::string output_problem =
      saved_frames_problem(out_dir, supported_image_count(input_dir));
  if (!output_problem.empty()) {
    std::cerr << "[FAIL] " << label << ": " << output_problem << "\n";
    remove_dir(out_dir);
    return 1;
  }

  // Every packaged COCO test image shows at least one object above the threshold.
  const std::regex detection_line(R"(\[\d+/\d+\] (\S+): (\d+) detections -> )");
  int reported_images = 0;
  for (std::sregex_iterator it(r.stdout_text.begin(), r.stdout_text.end(), detection_line), end;
       it != end; ++it) {
    ++reported_images;
    if (std::stoi((*it)[2].str()) == 0) {
      std::cerr << "[FAIL] " << label << ": no detections for " << (*it)[1].str() << "\nstdout:\n"
                << r.stdout_text << "\n";
      remove_dir(out_dir);
      return 1;
    }
  }
  if (reported_images != supported_image_count(input_dir)) {
    std::cerr << "[FAIL] " << label << ": expected a detection line per image, got "
              << reported_images << "\nstdout:\n"
              << r.stdout_text << "\n";
    remove_dir(out_dir);
    return 1;
  }

  remove_dir(out_dir);
  std::cout << "[OK] " << label << ": produced " << output_files
            << " output files with detections in every image\n";
  return 0;
}

// Writes a 4032x3024 copy of the first test image into its own folder and returns the folder,
// or an empty string when no test image can be read.
std::string write_large_photo_dir(const std::string& input_dir) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  for (const fs::path& image_path : images) {
    cv::Mat photo = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (photo.empty()) {
      continue;
    }
    const std::string photo_dir = create_test_output_dir(kExampleName, "large_photo_input");
    cv::resize(photo, photo, cv::Size(4032, 3024));
    if (photo_dir.empty() ||
        !cv::imwrite((fs::path(photo_dir) / "photo_4032x3024.jpg").string(), photo)) {
      return "";
    }
    return photo_dir;
  }
  return "";
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

  int rc = 0;
  const std::vector<std::string> model_paths =
      available_model_paths(configured_model_paths(kExampleName, models_dir), rc);
  if (model_paths.empty()) {
    return skip_or_fail("DETR model (.tar.gz) not found under SIMANEAT_APPS_TEST_MODELS_DIR");
  }

  const char* input_dir_raw = env_or_null("SIMANEAT_APPS_TEST_INPUT_DIR");
  const std::string input_dir = input_dir_raw ? input_dir_raw : "assets/datasets-test/coco";

  if (!fs::exists(input_dir) || fs::is_empty(input_dir)) {
    env_or_skip("SIMANEAT_APPS_TEST_INPUT_DIR",
                "directory containing test images (defaults to assets/datasets-test/coco)");
    return kSkipCode;
  }

  for (const std::string& model_path : model_paths) {
    const std::string label =
        model_case_label("test_full_pipeline", model_path, model_paths.size());
    if (run_model_case(binary, model_path, input_dir, label) != 0) {
      rc = 1;
    }
  }

  const std::string photo_dir = write_large_photo_dir(input_dir);
  if (photo_dir.empty()) {
    std::cerr << "[FAIL] test_photo_larger_than_1080p: could not write the 4032x3024 photo\n";
    return 1;
  }
  for (const std::string& model_path : model_paths) {
    const std::string label =
        model_case_label("test_photo_larger_than_1080p", model_path, model_paths.size());
    if (run_model_case(binary, model_path, photo_dir, label) != 0) {
      rc = 1;
    }
  }
  remove_dir(photo_dir);
  return rc;
}
