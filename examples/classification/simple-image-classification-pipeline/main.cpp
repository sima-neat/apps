/**
 * @example simple-image-classification-pipeline.cpp
 * Minimal Model usage with a ResNet50 MPK.
 */
#include "neat.h"
#include "support/runtime/example_utils.h"

#include <filesystem>
#include <array>
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
    std::cerr << "Missing ResNet50 MPK tarball.\n";
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

  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> stddev = {0.229f, 0.224f, 0.225f};

  // [model_basic]
  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "RGB";
  model_opt.input_max_width = kInferWidth;
  model_opt.input_max_height = kInferHeight;
  model_opt.input_max_depth = 3;
  model_opt.preproc.normalize = true;
  model_opt.preproc.channel_mean = std::array<float, 3>{mean[0], mean[1], mean[2]};
  model_opt.preproc.channel_stddev = std::array<float, 3>{stddev[0], stddev[1], stddev[2]};
  simaai::neat::Model model(model_path, model_opt);

  try {
    auto out = model.run(rgb);
    if (!out.tensor.has_value()) {
      std::cerr << "Model run returned empty output\n";
      return 6;
    }
    auto scores = sima_examples::scores_from_tensor(*out.tensor, "model");
    sima_examples::check_top1(scores, kGoldfishId, min_prob, "model");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 6;
  }
  // [model_basic]

  return 0;
}
