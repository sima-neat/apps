// Unit tests for the SSD-MobileNet object detector: the checks the application leaves to its
// config, and the annotated images and report it writes. tests/python/test_unit.py runs the same
// checks in Python.

#include "../../src/cpp/utils/annotate.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
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

// The checks the application leaves to its config: the values it reads are usable.
bool test_shipped_config_names_a_supported_model_labels_and_folders() {
  const fs::path common_dir = "examples/object-detection/ssd-mobilenet-object-detector/src/common";
  const auto config = sima_examples::Config::load(common_dir / "config.yaml");
  const fs::path model_path = common_dir / config.string("model.path");
  bool ok =
      expect_true(model_path.filename() == "ssd_mobilenet_v2_modalix_int8_tess_mla_mpk.tar.gz",
                  "model.path names the SSD-MobileNetV2 package");
  // The packaged path reaches prebuilt-apps/models from the config file.
  ok &= expect_true(is_inside(model_path, "models"), "model.path is under models/");
  // The SSD decoder scores the 91 COCO class IDs; ID 0, the background, and the unused IDs are
  // N/A.
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  ok &= expect_true(labels.size() == 91U && labels[0] == "N/A" && labels[1] == "person" &&
                        labels[90] == "toothbrush",
                    "model.labels lists the 91 COCO class IDs");
  ok &= expect_true(config.integer("inference.max_detections") > 0,
                    "the shipped detection cap is positive");
  ok &= expect_true(
      config.number("inference.min_score") >= 0.0 && config.number("inference.min_score") <= 1.0 &&
          config.number("inference.nms_iou") >= 0.0 && config.number("inference.nms_iou") <= 1.0,
      "the shipped score and IoU thresholds are from 0 to 1");
  // The annotated PNGs go to their own folder, never into the input folder.
  const fs::path input_dir = config.string("input.source");
  const fs::path output_dir = config.string("output.dir");
  ok &= expect_true(fs::weakly_canonical(input_dir) != fs::weakly_canonical(output_dir),
                    "output.dir differs from input.source");
  ok &= expect_true(is_inside(output_dir, "sandbox"), "output.dir is under sandbox/");
  // The report is off until a customer names a path for it.
  ok &= expect_true(config.string("app.detections_json").empty(),
                    "app.detections_json is empty by default");
  return ok;
}

bool test_output_name_keeps_the_source_extension() {
  bool ok = expect_true(annotate::output_name("in/frame.jpg") == "frame_jpg.png",
                        "frame.jpg -> frame_jpg.png");
  ok &= expect_true(annotate::output_name("in/frame.PNG") == "frame_PNG.png",
                    "frame.PNG -> frame_PNG.png");
  return ok;
}

bool test_detections_record_lists_each_box_with_its_label() {
  const cv::Mat image = cv::Mat::zeros(48, 64, CV_8UC3);
  const std::vector<Box> boxes = {{1.0f, 2.0f, 30.0f, 40.0f, 0.75f, 3}};
  const nlohmann::json record =
      annotate::detections_record("in/street.jpg", image, boxes, {"bg", "a", "b", "car"});
  const nlohmann::json expected = nlohmann::json::parse(R"({
      "image": "street.jpg", "width": 64, "height": 48,
      "detections": [{"class_id": 3, "label": "car", "score": 0.75, "box": [1.0, 2.0, 30.0, 40.0]}]})");
  return expect_true(record == expected, "the record lists the box: " + record.dump());
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_labels_and_folders();
  ok &= test_output_name_keeps_the_source_extension();
  ok &= test_detections_record_lists_each_box_with_its_label();
  return ok ? 0 : 1;
}
