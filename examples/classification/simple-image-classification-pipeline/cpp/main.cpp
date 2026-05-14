/**
 * @example simple-image-classification-pipeline.cpp
 * Minimal Model usage with a ResNet50 compiled model package.
 */
#include "neat.h"
#include "support/runtime/example_utils.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  constexpr const char* kGoldfishUrl =
      "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
      "n01443537_goldfish.JPEG";
  constexpr int kGoldfishId = 1; // ILSVRC2012 0-based index for "goldfish"
  constexpr int kInferWidth = 224;
  constexpr int kInferHeight = 224;

  std::string model_path;
  std::string image_path;
  std::string goldfish_url = kGoldfishUrl;
  float min_prob = 0.2f;

  std::string tmp;
  if (sima_examples::get_arg(argc, argv, "--model", tmp))
    model_path = tmp;
  if (sima_examples::get_arg(argc, argv, "--image", tmp))
    image_path = tmp;
  if (sima_examples::get_arg(argc, argv, "--goldfish-url", tmp))
    goldfish_url = tmp;
  if (sima_examples::get_arg(argc, argv, "--min-prob", tmp))
    min_prob = std::stof(tmp);

  if (model_path.empty()) {
    std::cerr << "Missing ResNet50 compiled model package.\n";
    std::cerr << "Pass it explicitly with --model <path/to/resnet_50_mpk.tar.gz>.\n";
    return 2;
  }

  if (image_path.empty()) {
    const fs::path out_path = sima_examples::default_goldfish_path();
    if (!sima_examples::download_file(goldfish_url, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      std::cerr << "URL was: " << goldfish_url << "\n";
      return 3;
    }
    image_path = out_path.string();
  }

  std::cout << "Using model: " << model_path << "\n";
  std::cout << "Using image: " << image_path << "\n";

  cv::Mat rgb;
  try {
    rgb = sima_examples::load_rgb_resized(image_path, kInferWidth, kInferHeight);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load image: " << e.what() << "\n";
    return 4;
  }

  // [model_basic]
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  model_opt.preprocess.input_max_width = kInferWidth;
  model_opt.preprocess.input_max_height = kInferHeight;
  model_opt.preprocess.input_max_depth = 3;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  simaai::neat::Model model(model_path, model_opt);

  try {
    const auto outputs = model.run(std::vector<cv::Mat>{rgb});
    if (outputs.empty()) {
      std::cerr << "Model run returned empty output\n";
      return 6;
    }
    auto scores = sima_examples::scores_from_tensor(outputs.front(), "model");
    sima_examples::check_top1(scores, kGoldfishId, min_prob, "model");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 6;
  }
  // [model_basic]

  return 0;
}
