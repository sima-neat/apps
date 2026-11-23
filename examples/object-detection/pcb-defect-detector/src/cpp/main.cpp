// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Finds defects on printed circuit boards in a folder of images with a custom YOLO26n model.
// Each image goes to the model at its own size. Core letterboxes it on the device and returns the
// boxes in the image's pixels, which are drawn on a copy saved to output.dir.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "utils/images.h"
#include "utils/overlay.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

// Added to each annotated image's name: board.jpg is saved as board_pcb.jpg.
constexpr const char* kOutputTag = "_pcb";

fs::path parse_args(int argc, char** argv) {
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  fs::path config_path = fs::exists(adjacent_config)
                             ? adjacent_config
                             : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else {
      throw std::runtime_error("usage: " + std::string(argv[0]) + " [--config <path>]");
    }
  }
  return config_path;
}

// YOLO26 on BGR images of any size up to max_size.
neat::Model::Options model_options(const sima_examples::Config& config,
                                   const std::vector<std::string>& labels, pcb::ImageSize max_size,
                                   int max_detections) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  options.preprocess.input_max_width = max_size.first;
  options.preprocess.input_max_height = max_size.second;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = static_cast<float>(config.number("inference.min_score"));
  options.nms_iou_threshold = static_cast<float>(config.number("inference.nms_iou"));
  options.top_k = max_detections;
  // Loading fails when the label file's class count differs from the model's.
  options.num_classes = static_cast<int>(labels.size());
  return options;
}

neat::RunOptions run_options() {
  neat::RunOptions options;
  // Realtime is the only preset that applies a new input size on the first frame; the others
  // hold a frame until a second one of the same size arrives, which one image at a time never
  // sends. Its KeepLatest overflow policy drops nothing here, because runner.run waits for each
  // result before the next image.
  options.preset = neat::RunPreset::Realtime;
  // 8 queue slots, not the preset's 2. One image is in flight at a time, so a smaller depth may do
  // as well, but that is not measured.
  options.queue_depth = 8;
  return options;
}

// The defect boxes in one image, in its pixels.
std::vector<neat::Box> detect_defects(neat::Model::Runner& runner, const cv::Mat& image,
                                      int max_detections) {
  const neat::TensorList outputs = runner.run(std::vector<cv::Mat>{image});
  return neat::decode_bbox_tensor(outputs.at(0), image.cols, image.rows, max_detections,
                                  /*strict=*/true)
      .boxes;
}

std::string format_counts(const std::map<std::string, int>& counts) {
  std::string text = "{";
  for (const auto& [name, count] : counts) {
    text += (text.size() > 1 ? ", " : "") + name + ": " + std::to_string(count);
  }
  return text + "}";
}

// Saves an annotated copy of each image and prints its defects; returns the failed count.
int annotate_images(neat::Model::Runner& runner, const std::vector<fs::path>& images,
                    const std::vector<std::string>& labels, const fs::path& output_dir,
                    pcb::ImageSize max_size, int max_detections) {
  std::map<std::string, int> per_class;
  int boards_with_defects = 0;
  int total_defects = 0;
  int failed = 0;
  const auto start = std::chrono::steady_clock::now();
  for (size_t index = 0; index < images.size(); ++index) {
    const fs::path& image_path = images[index];
    cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (const std::string problem = pcb::image_problem(image, max_size); !problem.empty()) {
      std::cerr << "[ERR] " << image_path.filename().string() << ": " << problem << "\n";
      ++failed;
      continue;
    }
    const std::vector<neat::Box> boxes = detect_defects(runner, image, max_detections);
    pcb::draw_boxes(image, boxes, labels);
    const fs::path output_path =
        output_dir / (image_path.stem().string() + kOutputTag + image_path.extension().string());
    if (!pcb::save_image(output_path, image)) {
      std::cerr << "[ERR] " << output_path.string() << ": could not be written\n";
      ++failed;
      continue;
    }
    std::map<std::string, int> counts;
    for (const auto& box : boxes) {
      ++counts[labels.at(static_cast<size_t>(box.class_id))];
      ++per_class[labels.at(static_cast<size_t>(box.class_id))];
    }
    total_defects += static_cast<int>(boxes.size());
    boards_with_defects += boxes.empty() ? 0 : 1;
    std::cout << "[" << index + 1 << "/" << images.size() << "] " << image_path.filename().string()
              << " -> " << output_path.filename().string() << " (" << boxes.size() << " defects) "
              << format_counts(counts) << std::endl;
  }
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  std::cout << "Done: " << images.size() - static_cast<size_t>(failed) << "/" << images.size()
            << " images in " << cv::format("%.2f", elapsed.count())
            << " s | images_with_defects=" << boards_with_defects
            << " total_defects=" << total_defects << " failed=" << failed << "\n";
  std::cout << "Per-class totals: " << format_counts(per_class) << std::endl;
  return failed;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  const auto images = pcb::find_images(config.string("input.source"));
  const fs::path output_dir = config.string("output.dir");
  fs::create_directories(output_dir);
  // Used by both the model and the box decode.
  const int max_detections = config.integer("inference.max_detections");
  // Used by the model, the build seed, and the size check of each image.
  const pcb::ImageSize max_size{config.integer("app.input_max_width"),
                                config.integer("app.input_max_height")};
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config, labels, max_size, max_detections));
  // The seed fixes the route's input caps, so it is an image of the largest accepted size.
  const cv::Mat seed = cv::Mat::zeros(max_size.second, max_size.first, CV_8UC3);
  // The runner's Run closes when the runner is destroyed, so also when annotate_images throws.
  neat::Model::Runner runner =
      model.build(std::vector<cv::Mat>{seed}, neat::Model::RouteOptions{}, run_options());
  std::cout << "source=" << config.string("input.source") << " images=" << images.size()
            << " output=" << output_dir.string() << std::endl;
  const int failed = annotate_images(runner, images, labels, output_dir, max_size, max_detections);
  runner.close();
  return failed == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run_app(parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << "\n";
    return 1;
  }
}
