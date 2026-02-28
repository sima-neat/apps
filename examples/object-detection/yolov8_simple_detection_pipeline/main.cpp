/**
 * @example yolov8_simple_detection_pipeline.cpp
 * Minimal YOLOv8n sync pipeline: infer detections for every image in a folder.
 *
 * Usage: yolov8_simple_detection_pipeline <model.tar.gz> <labels.txt> <input_dir> <output_dir>
 */
#include "neat.h"
#include "support/object_detection/obj_detection_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInferSize = 640;
constexpr float kMinScore = 0.52f;
constexpr float kNmsIou = 0.50f;
constexpr int kMaxDet = 100;
constexpr int kTimeoutMs = 5000;

struct TensorHWC {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;
};

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

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }

  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    for (const auto& field : sample.fields) {
      auto child = collect_tensors(field);
      out.insert(out.end(), child.begin(), child.end());
    }
    return out;
  }

  throw std::runtime_error("unexpected sample kind");
}

TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor for YOLOv8 decode");
  }

  TensorHWC out;
  if (t.shape.size() == 4) {
    if (t.shape[0] != 1) {
      throw std::runtime_error("only batch=1 is supported");
    }
    out.h = static_cast<int>(t.shape[1]);
    out.w = static_cast<int>(t.shape[2]);
    out.c = static_cast<int>(t.shape[3]);
  } else if (t.shape.size() == 3) {
    out.h = static_cast<int>(t.shape[0]);
    out.w = static_cast<int>(t.shape[1]);
    out.c = static_cast<int>(t.shape[2]);
  } else {
    throw std::runtime_error("unexpected tensor rank for HWC decode");
  }

  const size_t elems =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  const std::vector<uint8_t> bytes = t.copy_dense_bytes_tight();
  if (bytes.size() < elems * sizeof(float)) {
    throw std::runtime_error("tensor byte size is smaller than expected");
  }

  out.data.resize(elems);
  std::memcpy(out.data.data(), bytes.data(), elems * sizeof(float));
  return out;
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float dfl_distance_16(const float* logits) {
  float maxv = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < 16; ++i) {
    maxv = std::max(maxv, logits[i]);
  }

  float denom = 0.0f;
  float numer = 0.0f;
  for (int i = 0; i < 16; ++i) {
    const float e = std::exp(logits[i] - maxv);
    denom += e;
    numer += static_cast<float>(i) * e;
  }
  if (denom <= 0.0f) {
    return 0.0f;
  }
  return numer / denom;
}

float iou_xyxy(const objdet::Box& a, const objdet::Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float den = area_a + area_b - inter;
  return den > 0.0f ? (inter / den) : 0.0f;
}

std::vector<objdet::Box> decode_yolov8_boxes_from_tensors(
    const std::vector<simaai::neat::Tensor>& tensors) {
  if (tensors.size() < 6) {
    throw std::runtime_error("expected at least 6 tensors for YOLOv8 decode");
  }

  const std::array<TensorHWC, 3> regs = {tensor_to_hwc_f32(tensors[0]), tensor_to_hwc_f32(tensors[1]),
                                         tensor_to_hwc_f32(tensors[2])};
  const std::array<TensorHWC, 3> clss = {tensor_to_hwc_f32(tensors[3]),
                                         tensor_to_hwc_f32(tensors[4]),
                                         tensor_to_hwc_f32(tensors[5])};

  std::vector<objdet::Box> candidates;
  candidates.reserve(2000);

  for (size_t level = 0; level < regs.size(); ++level) {
    const auto& reg = regs[level];
    const auto& cls = clss[level];

    if (reg.h <= 0 || reg.w <= 0 || reg.c < 64) {
      continue;
    }
    if (reg.h != cls.h || reg.w != cls.w || cls.c <= 0) {
      throw std::runtime_error("unexpected YOLOv8 tensor shapes");
    }

    const float stride = static_cast<float>(kInferSize) / static_cast<float>(reg.h);
    for (int y = 0; y < reg.h; ++y) {
      for (int x = 0; x < reg.w; ++x) {
        const size_t cls_base =
            (static_cast<size_t>(y) * static_cast<size_t>(reg.w) + static_cast<size_t>(x)) *
            static_cast<size_t>(cls.c);

        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < cls.c; ++c) {
          const float score = sigmoid(cls.data[cls_base + static_cast<size_t>(c)]);
          if (score > best_score) {
            best_score = score;
            best_class = c;
          }
        }
        if (best_score < kMinScore || best_class < 0) {
          continue;
        }

        const size_t reg_base =
            (static_cast<size_t>(y) * static_cast<size_t>(reg.w) + static_cast<size_t>(x)) * 64U;
        const float l = dfl_distance_16(&reg.data[reg_base + 0]) * stride;
        const float t = dfl_distance_16(&reg.data[reg_base + 16]) * stride;
        const float r = dfl_distance_16(&reg.data[reg_base + 32]) * stride;
        const float b = dfl_distance_16(&reg.data[reg_base + 48]) * stride;

        const float cx = (static_cast<float>(x) + 0.5f) * stride;
        const float cy = (static_cast<float>(y) + 0.5f) * stride;

        objdet::Box box;
        box.x1 = std::max(0.0f, cx - l);
        box.y1 = std::max(0.0f, cy - t);
        box.x2 = std::min(static_cast<float>(kInferSize), cx + r);
        box.y2 = std::min(static_cast<float>(kInferSize), cy + b);
        box.score = best_score;
        box.class_id = best_class;

        if (box.x2 > box.x1 && box.y2 > box.y1) {
          candidates.push_back(box);
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const objdet::Box& a, const objdet::Box& b) { return a.score > b.score; });

  std::vector<objdet::Box> keep;
  keep.reserve(static_cast<size_t>(kMaxDet));
  for (const auto& box : candidates) {
    bool suppressed = false;
    for (const auto& kept : keep) {
      if (kept.class_id == box.class_id && iou_xyxy(kept, box) > kNmsIou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(box);
      if (static_cast<int>(keep.size()) >= kMaxDet) {
        break;
      }
    }
  }
  return keep;
}

std::vector<objdet::Box> scale_boxes_to_original(const std::vector<objdet::Box>& boxes, int out_w,
                                                  int out_h) {
  const float sx = static_cast<float>(out_w) / static_cast<float>(kInferSize);
  const float sy = static_cast<float>(out_h) / static_cast<float>(kInferSize);
  std::vector<objdet::Box> out;
  out.reserve(boxes.size());
  for (const auto& b : boxes) {
    out.push_back(
        objdet::Box{b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy, b.score, b.class_id});
  }
  return out;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0)};
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

void draw_boxes(cv::Mat& frame, const std::vector<objdet::Box>& boxes,
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
    std::cerr << "Usage: " << argv[0]
              << " <model.tar.gz> <labels.txt> <input_dir> <output_dir>\n";
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
    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "BGR";
    model_opt.input_max_width = kInferSize;
    model_opt.input_max_height = kInferSize;
    model_opt.input_max_depth = 3;

    simaai::neat::Model model(model_path, model_opt);

    simaai::neat::Session session;
    session.add(model.session());
    std::cout << "[BUILD] Pipeline:\n" << session.describe_backend() << "\n";

    cv::Mat dummy_bgr(kInferSize, kInferSize, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy = simaai::neat::from_cv_mat(
        dummy_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, /*read_only=*/true);
    auto run = session.build(dummy, simaai::neat::RunMode::Sync);

    int processed = 0;
    for (const auto& image_path : images) {
      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
        continue;
      }

      const int orig_w = bgr.cols;
      const int orig_h = bgr.rows;

      cv::Mat resized;
      cv::resize(bgr, resized, cv::Size(kInferSize, kInferSize), 0, 0, cv::INTER_LINEAR);
      simaai::neat::Tensor input = simaai::neat::from_cv_mat(
          resized, simaai::neat::ImageSpec::PixelFormat::BGR, /*read_only=*/true);

      simaai::neat::Sample out = run.push_and_pull(input, kTimeoutMs);

      std::vector<objdet::Box> boxes_infer;
      std::vector<uint8_t> payload;
      std::string bbox_err;
      if (objdet::extract_bbox_payload(out, payload, bbox_err)) {
        try {
          boxes_infer = objdet::parse_boxes_strict(payload, kInferSize, kInferSize, kMaxDet, false);
        } catch (const std::exception&) {
          boxes_infer = objdet::parse_boxes_lenient(payload, kInferSize, kInferSize, kMaxDet);
        }
      } else {
        try {
          const auto tensors = collect_tensors(out);
          boxes_infer = decode_yolov8_boxes_from_tensors(tensors);
        } catch (const std::exception& e) {
          std::cerr << "Decode failed for " << image_path.filename() << ": " << e.what() << "\n";
          continue;
        }
      }

      const auto boxes_orig = scale_boxes_to_original(boxes_infer, orig_w, orig_h);
      draw_boxes(bgr, boxes_orig, labels);

      const fs::path out_path = output_dir / (image_path.stem().string() + ".png");
      if (!cv::imwrite(out_path.string(), bgr)) {
        std::cerr << "Failed to write: " << out_path << "\n";
        continue;
      }

      ++processed;
      std::cout << "[" << processed << "/" << images.size() << "] " << image_path.filename()
                << " -> " << out_path.filename() << " (" << boxes_orig.size() << " detections)\n";
    }

    run.close();
    std::cout << "Done: " << processed << " images processed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
