// Unit tests for the DETR object detector: the checks the application leaves to its config, and
// the image size and annotated names it handles. tests/python/test_unit.py runs the same checks in
// Python. This file also tests the shared C++ image_folder, as agents/rules.md "Tests" says;
// tests/scripts/test_image_folder.py runs the same cases in Python.

#include "../../src/cpp/utils/images.h"
#include "sima_examples/config.h"
#include "sima_examples/image_folder.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

// The fields draw_detections reads, as Neat's Box has them.
struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = 0;
};

bool is_inside(const fs::path& path, const fs::path& folder) {
  const fs::path relative =
      fs::weakly_canonical(path).lexically_relative(fs::weakly_canonical(folder));
  return !relative.empty() && *relative.begin() != "..";
}

// A fresh, empty folder under the system temp folder.
fs::path empty_dir(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / ("detr-object-detector-" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_labels_and_folders() {
  const fs::path common_dir = "examples/object-detection/detr-object-detector/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const fs::path model_path = common_dir / config.string("model.path");
  bool ok = expect_true(model_path.filename() == "detr_resnet_50_w16a16_mpk.tar.gz",
                        "model.path names the DETR ResNet-50 package");
  // The packaged path reaches prebuilt-apps/models from the config file.
  ok &= expect_true(is_inside(model_path, "models"), "model.path is under models/");
  // DETR scores the 91 COCO class IDs; ID 0 and the other unused IDs are N/A.
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  ok &= expect_true(labels.size() == 91U && labels[0] == "N/A" && labels[1] == "person" &&
                        labels[90] == "toothbrush",
                    "model.labels lists the 91 COCO class IDs");
  // Neat's DETR decoder keeps the package's own threshold and cap when either is 0.
  ok &= expect_true(config.number("inference.min_score") > 0.0 &&
                        config.number("inference.min_score") <= 1.0,
                    "the shipped score threshold is above 0 and at most 1");
  ok &= expect_true(config.integer("inference.max_detections") > 0,
                    "the shipped detection cap is positive");
  // The annotated PNGs go to their own folder under sandbox/, never into the input folder.
  const fs::path input_dir = config.string("input.source");
  const fs::path output_dir = config.string("output.dir");
  ok &= expect_true(!sima_examples::image_paths(input_dir).empty(), "input.source holds images");
  ok &= expect_true(fs::weakly_canonical(input_dir) != fs::weakly_canonical(output_dir),
                    "output.dir is not input.source");
  ok &= expect_true(is_inside(output_dir, "sandbox"), "output.dir is under sandbox/");
  return ok;
}

// An empty scratch folder holding an empty file for each of names.
fs::path scratch_folder(const std::string& test_name, const std::vector<std::string>& names) {
  const fs::path folder = empty_dir(test_name);
  for (const auto& name : names) {
    std::ofstream(folder / name).put('\0');
  }
  return folder;
}

std::vector<std::string> file_names(const std::vector<fs::path>& paths) {
  std::vector<std::string> names;
  for (const auto& path : paths) {
    names.push_back(path.filename().string());
  }
  return names;
}

// The message image_paths throws for folder, or "" when it returns.
std::string image_paths_error(const fs::path& folder) {
  try {
    (void)sima_examples::image_paths(folder);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "";
}

bool test_image_paths_matches_suffixes_in_any_case() {
  const fs::path folder = scratch_folder(
      "image-suffixes", {"a.jpg", "b.JPEG", "c.Png", "d.BMP", "e.gif", "f.txt", "jpg"});
  const auto names = file_names(sima_examples::image_paths(folder));
  fs::remove_all(folder);
  return expect_true(names == std::vector<std::string>{"a.jpg", "b.JPEG", "c.Png", "d.BMP"},
                     "image_paths matches .jpg, .jpeg, .png, and .bmp in any case");
}

bool test_image_paths_lists_images_in_file_name_order() {
  // The order is by character code, so 10.jpg comes before 2.jpg and B.png before a.png.
  const fs::path folder =
      scratch_folder("image-order", {"c.bmp", "b.jpg", "a.png", "B.png", "2.jpg", "10.jpg"});
  const auto names = file_names(sima_examples::image_paths(folder));
  fs::remove_all(folder);
  return expect_true(
      names == std::vector<std::string>{"10.jpg", "2.jpg", "B.png", "a.png", "b.jpg", "c.bmp"},
      "image_paths lists the images in file-name order");
}

bool test_image_paths_skips_subfolders_and_their_files() {
  const fs::path folder = scratch_folder("image-subfolders", {"a.jpg"});
  // A folder is not an image, whatever its name ends with, and images inside one are skipped.
  fs::create_directories(folder / "archive.jpg");
  fs::create_directories(folder / "more");
  std::ofstream(folder / "more" / "b.jpg").put('\0');
  const auto names = file_names(sima_examples::image_paths(folder));
  fs::remove_all(folder);
  return expect_true(names == std::vector<std::string>{"a.jpg"},
                     "image_paths skips subfolders and the files in them");
}

bool test_image_paths_names_the_folder_when_it_has_no_images() {
  const fs::path folder = scratch_folder("image-none", {"notes.txt"});
  fs::create_directories(folder / "archive.jpg");
  const std::string error = image_paths_error(folder);
  fs::remove_all(folder);
  return expect_true(error == "no .jpg, .jpeg, .png, or .bmp files in " + folder.string(),
                     "a folder with no images is named: " + error);
}

bool test_image_paths_names_a_missing_folder() {
  const fs::path folder = fs::temp_directory_path() / "detr-object-detector-image-absent";
  fs::remove_all(folder);
  const std::string error = image_paths_error(folder);
  return expect_true(error == "image folder does not exist: " + folder.string(),
                     "a missing folder is named: " + error);
}

bool test_read_image_returns_a_bgr_image_and_names_a_file_it_cannot_read() {
  const fs::path folder = empty_dir("read-image");
  cv::imwrite((folder / "wide.png").string(), cv::Mat(30, 64, CV_8UC3, cv::Scalar(10, 20, 30)));
  std::ofstream(folder / "broken.png") << "not an image";
  const cv::Mat image = sima_examples::read_image(folder / "wide.png");
  bool ok = expect_true(image.rows == 30 && image.cols == 64 &&
                            image.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30),
                        "read_image returns the file's BGR pixels");
  std::string error;
  try {
    (void)sima_examples::read_image(folder / "broken.png");
  } catch (const std::runtime_error& thrown) {
    error = thrown.what();
  }
  ok &= expect_true(error == "cannot read image: " + (folder / "broken.png").string(),
                    "an unreadable file is named: " + error);
  fs::remove_all(folder);
  return ok;
}

// Neat rejects any image above its input limits, so they must fit the widest and the tallest
// image, even when those are different images.
bool test_largest_image_size_takes_the_widest_and_the_tallest_image() {
  const fs::path dir = empty_dir("largest-size");
  const fs::path landscape = dir / "landscape.png";
  const fs::path portrait = dir / "portrait.png";
  cv::imwrite(landscape.string(), cv::Mat::zeros(30, 64, CV_8UC3));
  cv::imwrite(portrait.string(), cv::Mat::zeros(50, 40, CV_8UC3));
  const cv::Size largest = images::largest_image_size({landscape, portrait});
  fs::remove_all(dir);
  return expect_true(largest == cv::Size(64, 50),
                     "largest_image_size takes the widest and the tallest image");
}

bool test_output_name_marks_the_detr_result() {
  bool ok = expect_true(sima_examples::output_name("in/street.jpg", "_detr") == "street_detr.png",
                        "output_name puts the suffix between the stem and .png");
  ok &= expect_true(sima_examples::output_name("in/a.b.jpg", "") == "a.b.png",
                    "output_name keeps a dotted stem");
  return ok;
}

bool test_draw_detections_draws_the_box_in_its_class_color_on_a_copy() {
  const cv::Mat image = cv::Mat::zeros(60, 80, CV_8UC3);
  const cv::Mat annotated = sima_examples::draw_detections(
      image, std::vector<Box>{{9.6f, 20.4f, 50.0f, 55.0f, 0.9f, 1}}, {"N/A", "person"});
  bool ok = expect_true(cv::countNonZero(image.reshape(1)) == 0, "the input image is unchanged");
  // Class 1 is blue in BGR. x1 = 9.6 rounds to 10, and a 2-pixel line there covers columns 9 to
  // 11; truncating to 9 would cover 8 to 10.
  ok &= expect_true(annotated.at<cv::Vec3b>(40, 11) == cv::Vec3b(255, 0, 0),
                    "the box edge is drawn in its class color");
  ok &= expect_true(annotated.at<cv::Vec3b>(40, 8) == cv::Vec3b(0, 0, 0),
                    "the box edge starts at the rounded x1");
  ok &= expect_true(annotated.at<cv::Vec3b>(40, 30) == cv::Vec3b(0, 0, 0),
                    "the box is not filled");
  return ok;
}

bool test_draw_detections_repeats_the_colors_every_eight_classes() {
  const cv::Mat image = cv::Mat::zeros(60, 80, CV_8UC3);
  const std::vector<std::string> labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  const cv::Mat annotated = sima_examples::draw_detections(
      image, std::vector<Box>{{10.0f, 20.0f, 50.0f, 55.0f, 0.9f, 9}}, labels);
  // Class 9 takes class 1's color.
  return expect_true(annotated.at<cv::Vec3b>(40, 10) == cv::Vec3b(255, 0, 0),
                     "class 9 is drawn in class 1's color");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_labels_and_folders();
  ok &= test_image_paths_matches_suffixes_in_any_case();
  ok &= test_image_paths_lists_images_in_file_name_order();
  ok &= test_image_paths_skips_subfolders_and_their_files();
  ok &= test_image_paths_names_the_folder_when_it_has_no_images();
  ok &= test_image_paths_names_a_missing_folder();
  ok &= test_read_image_returns_a_bgr_image_and_names_a_file_it_cannot_read();
  ok &= test_largest_image_size_takes_the_widest_and_the_tallest_image();
  ok &= test_output_name_marks_the_detr_result();
  ok &= test_draw_detections_draws_the_box_in_its_class_color_on_a_copy();
  ok &= test_draw_detections_repeats_the_colors_every_eight_classes();
  return ok ? 0 : 1;
}
