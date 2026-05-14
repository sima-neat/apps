/**
 * @example simple-object-detection-overlay-pipeline.cpp
 * Minimal YOLOv8n sync pipeline: infer detections for every image in a folder.
 *
 * Usage: simple-object-detection-overlay-pipeline <model.tar.gz> <labels.txt> <input_dir>
 * <output_dir>
 */
#include "neat.h"

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

constexpr float kMinScore = 0.55f;
constexpr float kNmsIou = 0.50f;
constexpr int kMaxDet = 100;
constexpr int kTimeoutMs = 5000;

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

simaai::neat::Model::Options make_detection_options() {
  simaai::neat::Model::Options options;
  options.preprocess.kind = simaai::neat::InputKind::Image;
  options.preprocess.enable = simaai::neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  options.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  options.decode_type = simaai::neat::BoxDecodeType::YoloV8;
  options.score_threshold = kMinScore;
  options.nms_iou_threshold = kNmsIou;
  options.top_k = kMaxDet;
  return options;
}

std::vector<simaai::neat::Box> decode_detections(const simaai::neat::TensorList& outputs,
                                                 int image_width, int image_height) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no detection tensors");
  }

  return simaai::neat::decode_bbox_tensor(outputs.front(), image_width, image_height, kMaxDet,
                                          /*strict=*/false)
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

  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <labels.txt> <input_dir> <output_dir>\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const fs::path labels_path = argv[2];
  const fs::path input_dir = argv[3];
  const fs::path output_dir = argv[4];

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  fs::create_directories(output_dir);

  std::vector<std::string> labels;
  try {
    labels = load_labels(labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());

  if (images.empty()) {
    std::cerr << "No images found in " << input_dir << "\n";
    return 3;
  }
  std::cout << "Found " << images.size() << " images\n";

  try {
    simaai::neat::Model model(model_path, make_detection_options());

    int processed = 0;
    for (const auto& image_path : images) {
      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
        continue;
      }

      std::vector<simaai::neat::Box> boxes;
      try {
        const auto outputs = model.run(std::vector<cv::Mat>{bgr}, kTimeoutMs);
        boxes = decode_detections(outputs, bgr.cols, bgr.rows);
      } catch (const std::exception& e) {
        std::cerr << "Detection failed for " << image_path.filename() << ": " << e.what() << "\n";
        continue;
      }

      draw_boxes(bgr, boxes, labels);

      const fs::path out_path = output_dir / (image_path.stem().string() + ".png");
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
