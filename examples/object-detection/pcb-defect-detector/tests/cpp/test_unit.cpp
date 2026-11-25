// Unit tests for the PCB defect detector: the checks the application leaves to its config, and
// its own logic. tests/python/test_unit.py runs the same checks in Python.

#include "../../src/cpp/utils/images.h"
#include "../../src/cpp/utils/overlay.h"
#include "sima_examples/config.h"
#include "sima_examples/image_folder.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

const fs::path kExampleDir = "examples/object-detection/pcb-defect-detector";
const fs::path kCommonDir = kExampleDir / "src" / "common";

simaai::neat::Box box(float x1, float y1, float x2, float y2, int class_id) {
  simaai::neat::Box result;
  result.x1 = x1;
  result.y1 = y1;
  result.x2 = x2;
  result.y2 = y2;
  result.score = 0.9f;
  result.class_id = class_id;
  return result;
}

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_the_model_labels_and_folders() {
  const auto config = sima_examples::Config::load(kCommonDir / "config.yaml");
  const fs::path model_path = kCommonDir / config.string("model.path");
  bool ok = expect_true(model_path.filename() == "yolo26n_plc_mpk.tar.gz",
                        "model.path names the PCB defect package");
  // The packaged path reaches prebuilt-apps/models from the config file.
  ok &=
      expect_true(fs::weakly_canonical(model_path).parent_path() == fs::weakly_canonical("models"),
                  "model.path reaches the models folder of the apps root");
  // The model's class IDs index the labels, and each class has a color.
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  ok &= expect_true(labels == std::vector<std::string>{"missing_hole", "mouse_bite", "open_circuit",
                                                       "short", "spur", "spurious_copper"},
                    "model.labels lists the six defect classes in the model's order");
  ok &= expect_true(pcb::kClassColors.size() == labels.size(), "each class has a color");
  // The shipped sample boards run as soon as the model is in place.
  ok &= expect_true(!sima_examples::image_paths(config.string("input.source")).empty(),
                    "input.source holds the sample boards");
  const std::string output_dir = config.string("output.dir");
  // An annotated image written into input.source would be read as a board on the next run.
  ok &= expect_true(output_dir.rfind("sandbox/", 0) == 0 &&
                        fs::path(output_dir) != fs::path(config.string("input.source")),
                    "output.dir is under sandbox/ and is not input.source");
  ok &= expect_true(
      config.number("inference.min_score") >= 0.0 && config.number("inference.min_score") <= 1.0 &&
          config.number("inference.nms_iou") >= 0.0 && config.number("inference.nms_iou") <= 1.0,
      "the shipped score and IoU thresholds are from 0 to 1");
  ok &= expect_true(config.integer("inference.max_detections") > 0 &&
                        config.integer("app.input_max_width") > 0 &&
                        config.integer("app.input_max_height") > 0,
                    "the shipped detection cap and largest image size are positive");
  return ok;
}

// CI downloads the package the shipped config names.
bool test_test_scope_downloads_the_configured_model() {
  const auto config = sima_examples::Config::load(kCommonDir / "config.yaml");
  const auto scope = sima_examples::Config::load(kExampleDir / "tests" / "test-scope.yaml");
  return expect_true(scope.string("models.yolo26n_plc.file") ==
                         fs::path(config.string("model.path")).filename().string(),
                     "test-scope.yaml downloads the model the config names");
}

bool test_image_problem_names_what_the_model_cannot_take() {
  const pcb::ImageSize max_size{640, 480};
  bool ok = expect_true(pcb::image_problem(cv::Mat(), max_size) == "not a readable image",
                        "an unreadable image is named");
  ok &= expect_true(pcb::image_problem(cv::Mat::zeros(480, 640, CV_8UC3), max_size).empty(),
                    "an image of the largest size fits");
  ok &=
      expect_true(pcb::image_problem(cv::Mat::zeros(480, 641, CV_8UC3), max_size) ==
                      "641x480 is larger than app.input_max_width x app.input_max_height (640x480)",
                  "a wider image is named with both sizes");
  ok &= expect_true(!pcb::image_problem(cv::Mat::zeros(481, 640, CV_8UC3), max_size).empty(),
                    "a taller image is named");
  return ok;
}

bool test_save_image_reports_a_failed_write_instead_of_throwing() {
  const fs::path folder = fs::temp_directory_path() / "pcb-save-image-test";
  fs::remove_all(folder);
  fs::create_directories(folder);
  const cv::Mat image = cv::Mat::zeros(8, 8, CV_8UC3);
  bool ok = expect_true(pcb::save_image(folder / "board_pcb.png", image), "a PNG is saved");
  // OpenCV throws for an encoder failure, such as an extension it has no writer for; one image
  // that cannot be saved must not stop the batch.
  bool unknown_saved = true;
  try {
    unknown_saved = pcb::save_image(folder / "board_pcb.unknown", image);
  } catch (const std::exception&) {
  }
  ok &= expect_true(!unknown_saved, "an encoder failure is reported as not saved");
  ok &= expect_true(!pcb::save_image(folder / "missing" / "board_pcb.png", image),
                    "a missing folder is reported as not saved");
  fs::remove_all(folder);
  return ok;
}

bool test_pixel_rounds_half_up_and_stays_in_the_image() {
  return expect_true(pcb::pixel(2.5f, 10) == 3 && pcb::pixel(2.49f, 10) == 2 &&
                         pcb::pixel(-3.0f, 10) == 0 && pcb::pixel(12.0f, 10) == 9,
                     "pixel rounds half up and clamps to the image");
}

bool test_draw_boxes_draws_each_box_in_its_class_color() {
  cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
  pcb::draw_boxes(image, {box(10, 30, 60, 80, 2)}, {"missing_hole", "mouse_bite", "open_circuit"});
  // The left edge, below the label, is the class's color; the inside stays untouched.
  const cv::Vec3b edge = image.at<cv::Vec3b>(60, 10);
  const cv::Scalar& color = pcb::kClassColors[2];
  bool ok = expect_true(edge[0] == color[0] && edge[1] == color[1] && edge[2] == color[2],
                        "the box edge has its class's color");
  ok &= expect_true(image.at<cv::Vec3b>(55, 35) == cv::Vec3b(0, 0, 0),
                    "the inside of the box is untouched");
  return ok;
}

bool test_draw_boxes_skips_a_box_with_no_area() {
  cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
  pcb::draw_boxes(image, {box(40, 30, 40.2f, 80, 0), box(120, 30, 150, 80, 0)}, {"missing_hole"});
  return expect_true(cv::countNonZero(image.reshape(1)) == 0,
                     "a box with no width or height inside the image draws nothing");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_the_model_labels_and_folders();
  ok &= test_test_scope_downloads_the_configured_model();
  ok &= test_image_problem_names_what_the_model_cannot_take();
  ok &= test_save_image_reports_a_failed_write_instead_of_throwing();
  ok &= test_pixel_rounds_half_up_and_stays_in_the_image();
  ok &= test_draw_boxes_draws_each_box_in_its_class_color();
  ok &= test_draw_boxes_skips_a_box_with_no_area();
  return ok ? 0 : 1;
}
