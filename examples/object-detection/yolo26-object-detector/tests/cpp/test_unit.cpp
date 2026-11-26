// Unit tests for the YOLO26 object detector: the checks the application leaves to its config, and
// the annotated names it handles. tests/python/test_unit.py runs the same checks in Python. The
// shared C++ image_folder is tested in detr-object-detector's tests/cpp/test_unit.cpp.

#include "sima_examples/config.h"
#include "sima_examples/image_folder.h"
#include "sima_examples/labels.h"
#include "sima_examples/testing/test_checks.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

using sima_examples::testing::expect_true;

namespace {

bool is_inside(const fs::path& path, const fs::path& folder) {
  const fs::path relative =
      fs::weakly_canonical(path).lexically_relative(fs::weakly_canonical(folder));
  return !relative.empty() && *relative.begin() != "..";
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
  ok &= expect_true(!sima_examples::image_paths(input_dir).empty(), "input.source holds images");
  ok &= expect_true(fs::weakly_canonical(input_dir) != fs::weakly_canonical(output_dir),
                    "output.dir is not input.source");
  ok &= expect_true(is_inside(output_dir, "sandbox"), "output.dir is under sandbox/");
  return ok;
}

bool test_output_name_marks_the_yolo26_result() {
  return expect_true(sima_examples::output_name("in/street.jpg", "_yolo26") == "street_yolo26.png",
                     "output_name appends _yolo26.png to the image's name");
}

} // namespace

int main() {
  bool ok = test_shipped_config_names_a_supported_model_labels_and_folders();
  ok &= test_output_name_marks_the_yolo26_result();
  return ok ? 0 : 1;
}
