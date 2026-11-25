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

// Detects objects in a folder of images with YOLO26, saved as annotated PNGs.
// The model runs on one image at a time and decodes YOLO26's boxes itself. Each image is saved
// with its boxes drawn.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/labels.h"
#include "utils/annotate.h"
#include "utils/images.h"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

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

// OpenCV BGR images in, boxes out of Core's YOLO26 decoder.
neat::Model::Options model_options(const sima_examples::Config& config, int max_detections) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = static_cast<float>(config.number("inference.min_score"));
  options.nms_iou_threshold = static_cast<float>(config.number("inference.nms_iou"));
  options.top_k = max_detections;
  return options;
}

// The boxes the model finds in one image, in the image's pixels.
std::vector<neat::Box> detect(neat::Model::Runner& runner, const cv::Mat& image,
                              int max_detections) {
  const neat::TensorList outputs = runner.run(std::vector<cv::Mat>{image});
  // strict fails on anything but a BBOX payload, such as raw model heads.
  return neat::decode_bbox_tensor(outputs.at(0), image.cols, image.rows, max_detections,
                                  /*strict=*/true)
      .boxes;
}

// Save each image with its boxes drawn to output_dir.
void annotate_images(neat::Model::Runner& runner, const std::vector<fs::path>& paths,
                     const std::vector<std::string>& labels, int max_detections,
                     const fs::path& output_dir) {
  for (size_t index = 0; index < paths.size(); ++index) {
    const fs::path& path = paths[index];
    const cv::Mat image = images::read_image(path);
    const std::vector<neat::Box> boxes = detect(runner, image, max_detections);
    const fs::path output_path = output_dir / annotate::output_name(path);
    if (!cv::imwrite(output_path.string(), annotate::draw_detections(image, boxes, labels))) {
      throw std::runtime_error("cannot write " + output_path.string());
    }
    std::cout << "[" << index + 1 << "/" << paths.size() << "] " << path.filename().string()
              << ": " << boxes.size() << " detections -> " << output_path.filename().string()
              << "\n";
  }
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  // Used by both the model and the box decode.
  const int max_detections = config.integer("inference.max_detections");
  // YOLO26's class IDs index the labels directly.
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  const auto paths = images::image_paths(config.string("input.source"));
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config, max_detections));
  // The first image seeds the build, which plans the route for images like it.
  neat::Model::Runner runner =
      model.build(std::vector<cv::Mat>{images::read_image(paths.front())});
  const fs::path output_dir = config.string("output.dir");
  fs::create_directories(output_dir);
  // When annotate_images throws, the Run the Runner holds closes as it is destroyed.
  annotate_images(runner, paths, labels, max_detections, output_dir);
  runner.close();

  std::cout << "Done: " << paths.size() << " images processed" << std::endl;
  return 0;
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
