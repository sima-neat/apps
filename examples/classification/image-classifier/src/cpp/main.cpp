/**
 * @example image-classifier.cpp
 * Minimal Model usage with a ResNet50 compiled model package.
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Config {
  std::string model_path;
  std::string image_path;
  std::string fallback_image_url;
  int input_width = 224;
  int input_height = 224;
  int timeout_ms = 20000;
  int expected_class_id = 1;
  float min_probability = 0.2f;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "models/resnet_50_mpk.tar.gz");
  cfg.image_path = raw.string_or("io.image", "");
  cfg.fallback_image_url =
      raw.string_or("io.fallback_image_url",
                    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
                    "n01443537_goldfish.JPEG");
  cfg.input_width = raw.int_or("runtime.input_width", 224);
  cfg.input_height = raw.int_or("runtime.input_height", 224);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 20000);
  cfg.expected_class_id = raw.int_or("validation.expected_class_id", 1);
  cfg.min_probability = static_cast<float>(raw.double_or("validation.min_probability", 0.2));
  return cfg;
}

Config parse_config(int argc, char** argv) {
  fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return load_config(config_path);
}

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  const Config cfg = parse_config(argc, argv);

  if (cfg.model_path.empty()) {
    std::cerr << "Missing ResNet50 compiled model package.\n";
    std::cerr << "Set model.path in the config file.\n";
    return 2;
  }

  std::string image_path = cfg.image_path;
  if (image_path.empty()) {
    const fs::path out_path = sima_examples::default_goldfish_path();
    if (!sima_examples::download_file(cfg.fallback_image_url, out_path)) {
      std::cerr << "Failed to download goldfish image.\n";
      std::cerr << "URL was: " << cfg.fallback_image_url << "\n";
      return 3;
    }
    image_path = out_path.string();
  }

  std::cout << "Using model: " << cfg.model_path << "\n";
  std::cout << "Using image: " << image_path << "\n";

  cv::Mat rgb;
  try {
    rgb = sima_examples::load_rgb_resized(image_path, cfg.input_width, cfg.input_height);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load image: " << e.what() << "\n";
    return 4;
  }

  // [model_basic]
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  model_opt.preprocess.input_max_width = cfg.input_width;
  model_opt.preprocess.input_max_height = cfg.input_height;
  model_opt.preprocess.input_max_depth = 3;
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  simaai::neat::Model model(cfg.model_path, model_opt);

  try {
    const auto input =
        simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB);
    const auto outputs = model.run(simaai::neat::TensorList{input}, cfg.timeout_ms);
    if (outputs.empty()) {
      std::cerr << "Model run returned empty output\n";
      return 6;
    }
    auto scores = sima_examples::scores_from_tensor(outputs.front(), "model");
    sima_examples::check_top1(scores, cfg.expected_class_id, cfg.min_probability, "model");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 6;
  }
  // [model_basic]

  return 0;
}
