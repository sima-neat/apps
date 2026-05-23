/**
 * @example simple-object-detection-overlay-pipeline.cpp
 * Minimal YOLOv8n sync pipeline: infer detections for every image in a folder.
 *
 * Usage: simple-object-detection-overlay-pipeline [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

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
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Config {
  std::string model_path;
  fs::path labels_path;
  fs::path input_dir;
  fs::path output_dir;
  float score_threshold = 0.55f;
  float nms_iou = 0.50f;
  int max_detections = 100;
  int timeout_ms = 5000;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "assets/models/yolo_v8n_mpk.tar.gz");
  cfg.labels_path = raw.string_or(
      "model.labels",
      "examples/object-detection/simple-object-detection-overlay-pipeline/common/coco_label.txt");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/test_images");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/simple_object_detection_overlay");
  cfg.score_threshold = static_cast<float>(raw.double_or("decode.score_threshold", 0.55));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", 0.50));
  cfg.max_detections = raw.int_or("decode.max_detections", 100);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 5000);
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

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

simaai::neat::Model::Options make_detection_options(const Config& cfg) {
  simaai::neat::Model::Options options;
  options.preprocess.kind = simaai::neat::InputKind::Image;
  options.preprocess.enable = simaai::neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  options.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  options.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  options.score_threshold = cfg.score_threshold;
  options.nms_iou_threshold = cfg.nms_iou;
  options.top_k = cfg.max_detections;
  return options;
}

std::vector<simaai::neat::Box> decode_detections(const simaai::neat::TensorList& outputs,
                                                 int image_width, int image_height,
                                                 int max_detections) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no detection tensors");
  }

  return simaai::neat::decode_bbox_tensor(outputs.front(), image_width, image_height,
                                          max_detections, /*strict=*/false)
      .boxes;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0)};
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

void draw_boxes(cv::Mat& frame, const std::vector<simaai::neat::Box>& boxes,
                const std::vector<std::string>& labels) {
  for (const auto& b : boxes) {
    const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x1))));
    const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y1))));
    const int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x2))));
    const int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color = class_color(b.class_id);
    const std::string text = class_name(labels, b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    int baseline = 0;
    const cv::Size tsz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int y_text_top = std::max(0, y1 - tsz.height - 4);
    const int y_text_bottom = std::max(0, y1);
    cv::rectangle(frame, cv::Point(x1, y_text_top), cv::Point(x1 + tsz.width, y_text_bottom), color,
                  cv::FILLED);
    cv::putText(frame, text, cv::Point(x1, std::max(0, y1 - 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  Config cfg;
  try {
    cfg = parse_config(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  if (!fs::is_directory(cfg.input_dir)) {
    std::cerr << "Input directory does not exist: " << cfg.input_dir << "\n";
    return 2;
  }
  fs::create_directories(cfg.output_dir);

  std::vector<std::string> labels;
  try {
    labels = load_labels(cfg.labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(cfg.input_dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());

  if (images.empty()) {
    std::cerr << "No images found in " << cfg.input_dir << "\n";
    return 3;
  }
  std::cout << "Found " << images.size() << " images\n";

  try {
    simaai::neat::Model model(cfg.model_path, make_detection_options(cfg));

    int processed = 0;
    for (const auto& image_path : images) {
      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
        continue;
      }

      std::vector<simaai::neat::Box> boxes;
      try {
        const auto outputs = model.run(std::vector<cv::Mat>{bgr}, cfg.timeout_ms);
        boxes = decode_detections(outputs, bgr.cols, bgr.rows, cfg.max_detections);
      } catch (const std::exception& e) {
        std::cerr << "Detection failed for " << image_path.filename() << ": " << e.what() << "\n";
        continue;
      }

      draw_boxes(bgr, boxes, labels);

      const fs::path out_path = cfg.output_dir / (image_path.stem().string() + ".png");
      if (!cv::imwrite(out_path.string(), bgr)) {
        std::cerr << "Failed to write: " << out_path << "\n";
        continue;
      }

      ++processed;
      std::cout << "[" << processed << "/" << images.size() << "] " << image_path.filename()
                << " -> " << out_path.filename() << " (" << boxes.size() << " detections)\n";
    }

    std::cout << "Done: " << processed << " images processed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
