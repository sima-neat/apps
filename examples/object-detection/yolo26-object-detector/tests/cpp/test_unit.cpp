// Unit tests for the YOLO26 object detector: the checks the application leaves to its config, and
// the image folder and annotated images it handles. tests/python/test_unit.py runs the same checks
// in Python.

#include "../../src/cpp/utils/annotate.h"
#include "../../src/cpp/utils/images.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

// The fields the annotate helpers read, as Neat's Box has them.
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
  const fs::path dir = fs::temp_directory_path() / ("yolo26-object-detector-" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_labels_and_folders() {
  const fs::path common_dir = "examples/object-detection/yolo26-object-detector/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const fs::path model_path = common_dir / config.string("model.path");
  bool ok = expect_true(model_path.filename() == "yolo26m_w16a16_mpk.tar.gz",
                        "model.path names the YOLO26m w16a16 package");
  // The packaged path reaches prebuilt-apps/models from the config file.
  ok &= expect_true(is_inside(model_path, "models"), "model.path is under models/");
  // YOLO26 scores the 80 COCO classes, IDs 0 to 79.
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  ok &= expect_true(labels.size() == 80U && labels[0] == "person" && labels[79] == "toothbrush",
                    "model.labels lists the 80 COCO classes");
  // Core's YOLO26 box decode accepts a score threshold below 1 only, and at 0 it keeps noise.
  ok &= expect_true(config.number("inference.min_score") > 0.0 &&
                        config.number("inference.min_score") < 1.0,
                    "the shipped score threshold is above 0 and below 1");
  // At 0, Core's YOLO26 decoder runs no NMS.
  ok &= expect_true(config.number("inference.nms_iou") > 0.0 &&
                        config.number("inference.nms_iou") <= 1.0,
                    "the shipped NMS overlap is above 0 and at most 1");
  ok &= expect_true(config.integer("inference.max_detections") > 0,
                    "the shipped detection cap is positive");
  // The annotated PNGs go to their own folder under sandbox/, never into the input folder.
  const fs::path input_dir = config.string("input.source");
  const fs::path output_dir = config.string("output.dir");
  ok &= expect_true(!images::image_paths(input_dir).empty(), "input.source holds images");
  ok &= expect_true(fs::weakly_canonical(input_dir) != fs::weakly_canonical(output_dir),
                    "output.dir is not input.source");
  ok &= expect_true(is_inside(output_dir, "sandbox"), "output.dir is under sandbox/");
  return ok;
}

bool test_image_paths_lists_only_images_in_name_order() {
  const fs::path dir = empty_dir("image-paths");
  for (const std::string name : {"b.PNG", "a.jpg", "c.bmp", "notes.txt", "d.jpeg"}) {
    std::ofstream(dir / name).put('\0');
  }
  fs::create_directories(dir / "e.jpg");
  std::vector<std::string> names;
  for (const auto& path : images::image_paths(dir)) {
    names.push_back(path.filename().string());
  }
  fs::remove_all(dir);
  return expect_true(names == std::vector<std::string>{"a.jpg", "b.PNG", "c.bmp", "d.jpeg"},
                     "image_paths lists only the images, in file-name order");
}

bool test_image_paths_rejects_a_folder_without_images() {
  const fs::path dir = empty_dir("no-images");
  std::ofstream(dir / "notes.txt").put('\0');
  std::string error;
  try {
    (void)images::image_paths(dir);
  } catch (const std::runtime_error& thrown) {
    error = thrown.what();
  }
  fs::remove_all(dir);
  return expect_true(error.rfind("input.source has no", 0) == 0,
                     "a folder without images names input.source: " + error);
}

bool test_output_name_marks_the_yolo26_result() {
  return expect_true(annotate::output_name("in/street.jpg") == "street_yolo26.png",
                     "output_name appends _yolo26.png to the image's name");
}

bool test_draw_detections_draws_the_box_in_its_class_color_on_a_copy() {
  const cv::Mat image = cv::Mat::zeros(60, 80, CV_8UC3);
  const cv::Mat annotated = annotate::draw_detections(
      image, std::vector<Box>{{9.6f, 20.4f, 50.0f, 55.0f, 0.9f, 1}}, {"person", "bicycle"});
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

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_labels_and_folders();
  ok &= test_image_paths_lists_only_images_in_name_order();
  ok &= test_image_paths_rejects_a_folder_without_images();
  ok &= test_output_name_marks_the_yolo26_result();
  ok &= test_draw_detections_draws_the_box_in_its_class_color_on_a_copy();
  return ok ? 0 : 1;
}
