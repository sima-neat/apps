/**
 * @example ssd-mobilenet-object-detector.cpp
 * SSD folder detection via the model-managed BoxDecodeType::Ssd pipeline. Runs TensorFlow
 * SSD-MobileNet v1/v2 (300x300) and TorchVision SSDlite-MobileNetV3 (320x320) with an explicit
 * preprocessing profile.
 *
 * Usage: ssd-mobilenet-object-detector [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kNumClasses = 91; // index 0 = background, 1..90 = COCO ids.
constexpr int kDefaultTimeoutMs = 20000;
constexpr const char* kDefaultLabelsPath =
    "examples/object-detection/ssd-mobilenet-object-detector/src/common/"
    "coco_labels.txt";

struct Config {
  std::string model;
  fs::path labels;
  fs::path input_dir;
  fs::path output_dir;
  fs::path detections_json;
  std::string preprocessing_profile = "tensorflow_ssd";
  float score_threshold = 0.55f;
  float nms_iou = 0.60f;
  int max_detections = 100;
  int timeout_ms = kDefaultTimeoutMs;
};

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

// Directory holding the running executable, or empty if it cannot be determined.
fs::path executable_dir() {
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (ec || exe.empty()) {
    return {};
  }
  return exe.parent_path();
}

// Resolve a labels asset from the configured path or the packaged src/common copy.
fs::path resolve_asset(const std::string& configured, const char* default_path,
                       const char* default_name) {
  if (!configured.empty() && fs::exists(configured)) {
    return configured;
  }
  // Substitute the packaged copy only for the empty/default reference; a missing custom
  // path is returned as-is so the caller fails instead of loading different labels.
  if (!configured.empty() &&
      fs::path(configured).lexically_normal() != fs::path(default_path).lexically_normal()) {
    return configured;
  }
  std::vector<fs::path> candidates;
  const fs::path exe_dir = executable_dir();
  if (!exe_dir.empty()) {
    candidates.push_back(exe_dir / ".." / ".." / "common" / default_name); // packaged pre-built/
    candidates.push_back(exe_dir / ".." / "common" / default_name);        // same-dir binary
  }
  for (const fs::path& candidate : candidates) {
    if (fs::exists(candidate)) {
      return candidate;
    }
  }
  return configured;
}

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or("model.path", "");
  cfg.preprocessing_profile = raw.string_or("model.preprocessing_profile", "tensorflow_ssd");
  cfg.labels =
      resolve_asset(raw.string_or("model.labels", ""), kDefaultLabelsPath, "coco_labels.txt");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/datasets/coco");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/ssd-mobilenet-object-detector");
  cfg.detections_json = raw.string_or("io.detections_json", "");
  cfg.score_threshold = static_cast<float>(raw.double_or("decode.score_threshold", 0.55));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", 0.60));
  cfg.max_detections = raw.int_or("decode.max_detections", 100);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kDefaultTimeoutMs);
  if (sima_examples::trim_copy(cfg.model).empty()) {
    throw std::runtime_error("model.path must be a nonempty path");
  }
  if (!std::isfinite(cfg.score_threshold) || cfg.score_threshold < 0.0f ||
      cfg.score_threshold > 1.0f) {
    throw std::runtime_error("decode.score_threshold must be in [0.0, 1.0]");
  }
  if (!std::isfinite(cfg.nms_iou) || cfg.nms_iou < 0.0f || cfg.nms_iou > 1.0f) {
    throw std::runtime_error("decode.nms_iou must be in [0.0, 1.0]");
  }
  if (cfg.max_detections < 1) {
    throw std::runtime_error("decode.max_detections must be >= 1");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  if (cfg.preprocessing_profile != "tensorflow_ssd" &&
      cfg.preprocessing_profile != "torchvision_ssdlite") {
    throw std::runtime_error(
        "model.preprocessing_profile must be tensorflow_ssd or torchvision_ssdlite");
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

std::vector<std::string> load_labels(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error("labels file does not exist: " + path.string());
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open labels file: " + path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    labels.push_back(line);
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + path.string());
  }
  return labels;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    const std::string& name = labels[static_cast<size_t>(class_id)];
    if (!name.empty() && name != "N/A") {
      return name;
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
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

std::vector<neat::Box> parse_detections(const neat::TensorList& outputs, int image_width,
                                        int image_height, int max_detections) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no detection tensors");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("expected one BBOX tensor from model-managed BoxDecode, got " +
                             std::to_string(outputs.size()));
  }
  return neat::decode_bbox_tensor(outputs.front(), image_width, image_height, max_detections,
                                  /*strict=*/false)
      .boxes;
}

std::string output_stem(const fs::path& image_path) {
  std::string extension = image_path.extension().string();
  if (!extension.empty() && extension.front() == '.') {
    extension.erase(extension.begin());
  }
  const std::string stem = image_path.stem().string();
  return extension.empty() ? stem : stem + "_" + extension;
}

void draw_detections(cv::Mat& bgr, const std::vector<neat::Box>& boxes,
                     const std::vector<std::string>& labels) {
  for (const neat::Box& b : boxes) {
    const int x1 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(b.x1))));
    const int y1 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(b.y1))));
    const int x2 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(b.x2))));
    const int y2 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(b.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const cv::Scalar color = class_color(b.class_id);
    const std::string text = class_name(labels, b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    cv::putText(bgr, text, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                2, cv::LINE_AA);
  }
}

// Machine-readable detection record, written when io.detections_json is set.
nlohmann::json detections_record(const fs::path& image_path, const cv::Mat& bgr,
                                 const std::vector<neat::Box>& boxes,
                                 const std::vector<std::string>& labels) {
  nlohmann::json entry;
  entry["image"] = image_path.filename().string();
  entry["width"] = bgr.cols;
  entry["height"] = bgr.rows;
  entry["detections"] = nlohmann::json::array();
  for (const neat::Box& b : boxes) {
    entry["detections"].push_back({{"class_id", b.class_id},
                                   {"label", class_name(labels, b.class_id)},
                                   {"score", b.score},
                                   {"box", {b.x1, b.y1, b.x2, b.y2}}});
  }
  return entry;
}

void write_detections_json(const fs::path& path, const nlohmann::json& images) {
  if (path.has_parent_path()) {
    fs::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write detections json: " + path.string());
  }
  out << nlohmann::json{{"images", images}}.dump(2) << "\n";
  // Catch a failed write/flush (e.g. full disk) before reporting success on a truncated file.
  out.flush();
  if (!out) {
    throw std::runtime_error("failed to write detections json: " + path.string());
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config cfg = parse_config(argc, argv);
    if (!fs::is_directory(cfg.input_dir)) {
      throw std::runtime_error("input_dir does not exist: " + cfg.input_dir.string());
    }
    if (fs::weakly_canonical(cfg.output_dir) == fs::weakly_canonical(cfg.input_dir)) {
      throw std::runtime_error("io.output_dir must differ from io.input_dir");
    }
    if (!fs::exists(cfg.model)) {
      throw std::runtime_error("model does not exist: " + cfg.model);
    }

    const std::vector<std::string> labels = load_labels(cfg.labels);

    const std::vector<fs::path> image_paths = image_paths_in_dir(cfg.input_dir);
    if (image_paths.empty()) {
      throw std::runtime_error("no images found in: " + cfg.input_dir.string());
    }
    cv::Mat seed_bgr = cv::imread(image_paths.front().string(), cv::IMREAD_COLOR);
    if (seed_bgr.empty()) {
      throw std::runtime_error("failed to read build seed image: " + image_paths.front().string());
    }

    // Core resolves the preprocess frame from the MPK and validates it against the SSD recipe.
    neat::Model::Options options;
    options.preprocess.kind = neat::InputKind::Image;
    options.preprocess.enable = neat::AutoFlag::On;
    options.preprocess.resize.enable = neat::AutoFlag::On;
    options.preprocess.resize.mode = neat::ResizeMode::Stretch;
    options.preprocess.normalize.enable = neat::AutoFlag::On;
    if (cfg.preprocessing_profile == "torchvision_ssdlite") {
      options.preprocess.normalize.mean = {0.485f, 0.456f, 0.406f};
      options.preprocess.normalize.stddev = {0.229f, 0.224f, 0.225f};
    } else {
      options.preprocess.normalize.mean = {0.5f, 0.5f, 0.5f};
      options.preprocess.normalize.stddev = {0.5f, 0.5f, 0.5f};
    }
    options.preprocess.normalize.has_explicit_stats = true;
    options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
    options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
    options.decode_type = neat::BoxDecodeType::Ssd;
    options.num_classes = kNumClasses;
    options.score_threshold = cfg.score_threshold;
    options.nms_iou_threshold = cfg.nms_iou;
    options.top_k = cfg.max_detections;
    neat::Model model(cfg.model, options);
    neat::Model::Runner run = [&] {
      try {
        return model.build(std::vector<cv::Mat>{seed_bgr}, neat::Model::RouteOptions{});
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) +
                                 "\n  hint: verify model.path points to a supported SSD model "
                                 "pack whose MLA input contract contains its model frame.");
      }
    }();
    run.run(std::vector<cv::Mat>{seed_bgr}, cfg.timeout_ms);

    fs::create_directories(cfg.output_dir);

    nlohmann::json records = nlohmann::json::array();
    int processed = 0;
    for (const fs::path& image_path : image_paths) {
      cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr_u8.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }

      neat::TensorList out = run.run(std::vector<cv::Mat>{bgr_u8}, cfg.timeout_ms);
      const std::vector<neat::Box> boxes =
          parse_detections(out, bgr_u8.cols, bgr_u8.rows, cfg.max_detections);

      cv::Mat annotated = bgr_u8.clone();
      draw_detections(annotated, boxes, labels);
      const fs::path output_path = cfg.output_dir / (output_stem(image_path) + ".png");
      if (!cv::imwrite(output_path.string(), annotated)) {
        throw std::runtime_error("failed to write: " + output_path.string());
      }

      if (!cfg.detections_json.empty()) {
        records.push_back(detections_record(image_path, bgr_u8, boxes, labels));
      }
      ++processed;
      std::cout << "[" << processed << "/" << image_paths.size() << "] "
                << image_path.filename().string() << " -> " << output_path.filename().string()
                << " (" << boxes.size() << " detections)\n";
    }
    if (!cfg.detections_json.empty()) {
      write_detections_json(cfg.detections_json, records);
      std::cout << "Wrote detections: " << cfg.detections_json.string() << "\n";
    }
    std::cout << "Done: " << processed << " images processed\n";

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
