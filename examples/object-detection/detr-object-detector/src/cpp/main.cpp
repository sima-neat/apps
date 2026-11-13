/**
 * @example detr-object-detector.cpp
 * DETR folder detection through Neat's model-managed BoxDecodeType::Detr decoder.
 *
 * Usage: detr-object-detector [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr const char* kDefaultModelPath =
    "models/detr_resnet_50/w16a16/detr_resnet_50_w16a16_mpk.tar.gz";
constexpr int kPersonClassId = 1;

constexpr std::array<std::string_view, 91> kDetrCocoLabels = {
    "N/A",          "person",         "bicycle",
    "car",          "motorcycle",     "airplane",
    "bus",          "train",          "truck",
    "boat",         "traffic light",  "fire hydrant",
    "N/A",          "stop sign",      "parking meter",
    "bench",        "bird",           "cat",
    "dog",          "horse",          "sheep",
    "cow",          "elephant",       "bear",
    "zebra",        "giraffe",        "N/A",
    "backpack",     "umbrella",       "N/A",
    "N/A",          "handbag",        "tie",
    "suitcase",     "frisbee",        "skis",
    "snowboard",    "sports ball",    "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard",    "tennis racket",  "bottle",
    "N/A",          "wine glass",     "cup",
    "fork",         "knife",          "spoon",
    "bowl",         "banana",         "apple",
    "sandwich",     "orange",         "broccoli",
    "carrot",       "hot dog",        "pizza",
    "donut",        "cake",           "chair",
    "couch",        "potted plant",   "bed",
    "N/A",          "dining table",   "N/A",
    "N/A",          "toilet",         "N/A",
    "tv",           "laptop",         "mouse",
    "remote",       "keyboard",       "cell phone",
    "microwave",    "oven",           "toaster",
    "sink",         "refrigerator",   "N/A",
    "book",         "clock",          "vase",
    "scissors",     "teddy bear",     "hair drier",
    "toothbrush",
};

struct Config {
  fs::path model_path;
  fs::path input_dir;
  fs::path output_dir;
  float confidence_threshold = 0.7f;
  int max_draw = 50;
  bool person_only = false;
  bool profile = false;
  int num_runs = 100;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", kDefaultModelPath);
  cfg.input_dir = raw.string_or("io.input_dir", "assets/datasets/coco");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/detr-object-detector");
  cfg.confidence_threshold =
      static_cast<float>(raw.double_or("decode.confidence_threshold", cfg.confidence_threshold));
  cfg.max_draw = raw.int_or("decode.max_draw", cfg.max_draw);
  cfg.person_only = raw.bool_or("decode.person_only", cfg.person_only);
  cfg.profile = raw.bool_or("runtime.profile", cfg.profile);
  cfg.num_runs = raw.int_or("runtime.num_runs", cfg.num_runs);

  // Neat's decoder keeps the package's own threshold and top-k when either is 0.
  if (!(cfg.confidence_threshold > 0.0f && cfg.confidence_threshold <= 1.0f)) {
    throw std::runtime_error("decode.confidence_threshold must be in (0.0, 1.0]");
  }
  if (cfg.max_draw < 1) {
    throw std::runtime_error("decode.max_draw must be >= 1");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
  }
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

bool is_image_file(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> image_paths_in_dir(const fs::path& input_dir) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image_file(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

std::string class_name(int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < kDetrCocoLabels.size()) {
    const std::string_view label = kDetrCocoLabels[static_cast<size_t>(class_id)];
    if (label != "N/A") {
      return std::string(label);
    }
  }
  return "class_" + std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
  };
  return kColors[static_cast<size_t>(std::abs(class_id)) % kColors.size()];
}

// Stretches, ImageNet-normalizes, and decodes DETR on the model route.
neat::Model::Options build_model_options(const Config& cfg) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  // Core's classic DETR decoder requires Stretch (Core docs/reference/boxdecode_decode_types.md).
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  // The registry calibrates and scores DETR on Pillow bilinear resizes, which average every source
  // pixel when shrinking; plain bilinear samples only the nearest four.
  options.preprocess.resize.interpolation = neat::ResizeInterpolation::BilinearAntialiased;
  options.preprocess.preset = neat::NormalizePreset::ImageNet;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  options.decode_type = neat::BoxDecodeType::Detr;
  options.score_threshold = cfg.confidence_threshold;
  options.top_k = cfg.max_draw;
  return options;
}

cv::Mat read_image(const fs::path& image_path) {
  cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image_bgr.empty()) {
    throw std::runtime_error("failed to read image: " + image_path.string());
  }
  return image_bgr;
}

// Runs one image through the model and returns its boxes in original-image pixels.
std::vector<neat::Box> detect(neat::Model::Runner& run, const cv::Mat& image_bgr,
                              const Config& cfg) {
  const neat::TensorList outputs = run.run(std::vector<cv::Mat>{image_bgr});
  if (outputs.size() != 1) {
    throw std::runtime_error("expected one BBOX tensor from the DETR decoder, got " +
                             std::to_string(outputs.size()));
  }
  std::vector<neat::Box> boxes =
      neat::decode_bbox_tensor(outputs.front(), image_bgr.cols, image_bgr.rows, cfg.max_draw,
                               /*strict=*/false)
          .boxes;
  if (cfg.person_only) {
    std::erase_if(boxes, [](const neat::Box& b) { return b.class_id != kPersonClassId; });
  }
  return boxes;
}

cv::Mat draw_detections(const cv::Mat& image_bgr, const std::vector<neat::Box>& boxes) {
  cv::Mat annotated = image_bgr.clone();
  for (const neat::Box& b : boxes) {
    const cv::Point top_left(static_cast<int>(std::lround(b.x1)),
                             static_cast<int>(std::lround(b.y1)));
    const cv::Point bottom_right(static_cast<int>(std::lround(b.x2)),
                                 static_cast<int>(std::lround(b.y2)));
    const cv::Scalar color = class_color(b.class_id);
    const std::string text = class_name(b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(annotated, top_left, bottom_right, color, 2);
    cv::putText(annotated, text, cv::Point(top_left.x, std::max(0, top_left.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
  }
  return annotated;
}

void annotate_images(neat::Model::Runner& run, const std::vector<fs::path>& image_paths,
                     const Config& cfg) {
  fs::create_directories(cfg.output_dir);
  for (size_t i = 0; i < image_paths.size(); ++i) {
    const cv::Mat image_bgr = read_image(image_paths[i]);
    const std::vector<neat::Box> boxes = detect(run, image_bgr, cfg);

    const fs::path output_path = cfg.output_dir / (image_paths[i].stem().string() + "_detr.png");
    if (!cv::imwrite(output_path.string(), draw_detections(image_bgr, boxes))) {
      throw std::runtime_error("failed to write: " + output_path.string());
    }
    std::cout << "[" << (i + 1) << "/" << image_paths.size() << "] "
              << image_paths[i].filename().string() << ": " << boxes.size() << " detections -> "
              << output_path.filename().string() << "\n";
  }
  std::cout << "Done: " << image_paths.size() << " images processed\n";
}

// Times preprocess, inference, and decode on one image after one untimed warm-up run.
void profile_model_run(neat::Model::Runner& run, const fs::path& image_path, const Config& cfg) {
  const cv::Mat image_bgr = read_image(image_path);
  std::vector<neat::Box> boxes = detect(run, image_bgr, cfg);

  std::vector<double> run_times_s;
  run_times_s.reserve(static_cast<size_t>(cfg.num_runs));
  for (int i = 0; i < cfg.num_runs; ++i) {
    const auto start = std::chrono::steady_clock::now();
    boxes = detect(run, image_bgr, cfg);
    run_times_s.push_back(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  }

  const double total_s = std::accumulate(run_times_s.begin(), run_times_s.end(), 0.0);
  const auto [min_s, max_s] = std::minmax_element(run_times_s.begin(), run_times_s.end());
  const double runs = static_cast<double>(run_times_s.size());
  std::cout << "Profiling over " << run_times_s.size() << " runs (image='" << image_path.string()
            << "'):\n";
  std::cout << "  Model run (preprocess+infer+decode): mean=" << (total_s / runs)
            << "s, min=" << *min_s << "s, max=" << *max_s << "s, FPS=" << (runs / total_s) << "\n";
  std::cout << "Last run detections: " << boxes.size() << "\n";
  for (size_t i = 0; i < std::min<size_t>(boxes.size(), 20); ++i) {
    const neat::Box& b = boxes[i];
    std::cout << "  [" << i << "] class=" << class_name(b.class_id) << "(" << b.class_id
              << ") score=" << b.score << " box=[" << b.x1 << "," << b.y1 << "," << b.x2 << ","
              << b.y2 << "]\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config cfg = parse_config(argc, argv);
    if (!fs::exists(cfg.model_path)) {
      throw std::runtime_error("model does not exist: " + cfg.model_path.string());
    }
    if (!fs::is_directory(cfg.input_dir)) {
      throw std::runtime_error("input_dir does not exist: " + cfg.input_dir.string());
    }
    const std::vector<fs::path> image_paths = image_paths_in_dir(cfg.input_dir);
    if (image_paths.empty()) {
      throw std::runtime_error("no images found in: " + cfg.input_dir.string());
    }

    neat::Model model(cfg.model_path.string(), build_model_options(cfg));
    neat::Model::Runner run = model.build(std::vector<cv::Mat>{read_image(image_paths.front())});

    if (cfg.profile) {
      profile_model_run(run, image_paths.front(), cfg);
    } else {
      annotate_images(run, image_paths, cfg);
    }
    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
