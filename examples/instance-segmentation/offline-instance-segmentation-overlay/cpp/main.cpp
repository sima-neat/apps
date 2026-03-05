/**
 * @example offline-instance-segmentation-overlay.cpp
 * Minimal YOLOv8-seg pipeline using DetessDequant postprocess (no boxdecode).
 *
 * Usage: offline-instance-segmentation-overlay <model.tar.gz> <input_dir> <output_dir>
 */
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

std::vector<simaai::neat::Tensor> tensors_from_sample(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }

  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    for (const auto& field : sample.fields) {
      auto part = tensors_from_sample(field);
      out.insert(out.end(), part.begin(), part.end());
    }
    return out;
  }

  throw std::runtime_error("unexpected sample kind");
}

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

struct Box {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  std::array<float, 32> coeff{};
};

struct TensorHWC {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;
};

const std::vector<std::string>& coco_labels() {
  static const std::vector<std::string> kLabels = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
  return kLabels;
}

std::string class_name_for_id(int class_id) {
  const auto& labels = coco_labels();
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return "class_" + std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 12> kPalette = {
      cv::Scalar(56, 56, 255),
      cv::Scalar(151, 157, 255),
      cv::Scalar(31, 112, 255),
      cv::Scalar(29, 178, 255),
      cv::Scalar(49, 210, 207),
      cv::Scalar(10, 249, 72),
      cv::Scalar(23, 204, 146),
      cv::Scalar(134, 219, 61),
      cv::Scalar(52, 147, 26),
      cv::Scalar(187, 212, 0),
      cv::Scalar(255, 194, 0),
      cv::Scalar(168, 153, 44),
  };
  if (class_id < 0) {
    class_id = 0;
  }
  return kPalette[static_cast<size_t>(class_id) % kPalette.size()];
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float iou_xyxy(const Box& a, const Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

inline float at_hwc(const TensorHWC& t, int y, int x, int c) {
  const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(t.w) + static_cast<size_t>(x)) *
                         static_cast<size_t>(t.c) +
                     static_cast<size_t>(c);
  return t.data[idx];
}

TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
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

  const std::vector<uint8_t> bytes = t.copy_dense_bytes_tight();
  const size_t elems =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  if (bytes.size() < elems * sizeof(float)) {
    throw std::runtime_error("tensor byte size is smaller than expected");
  }

  out.data.resize(elems);
  std::memcpy(out.data.data(), bytes.data(), elems * sizeof(float));
  return out;
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
  if (denom <= 0.0f)
    return 0.0f;
  return numer / denom;
}

std::vector<Box> decode_yolov8_instances_from_detess(const std::vector<simaai::neat::Tensor>& tensors,
                                                     int infer_size, float conf_thr, float nms_iou,
                                                     int max_det, TensorHWC& proto) {
  if (tensors.size() < 10) {
    throw std::runtime_error("expected at least 10 tensors for instance-seg decode");
  }

  const TensorHWC reg80 = tensor_to_hwc_f32(tensors[0]);
  const TensorHWC reg40 = tensor_to_hwc_f32(tensors[1]);
  const TensorHWC reg20 = tensor_to_hwc_f32(tensors[2]);
  const TensorHWC cls80 = tensor_to_hwc_f32(tensors[3]);
  const TensorHWC cls40 = tensor_to_hwc_f32(tensors[4]);
  const TensorHWC cls20 = tensor_to_hwc_f32(tensors[5]);
  const TensorHWC mk80 = tensor_to_hwc_f32(tensors[6]);
  const TensorHWC mk40 = tensor_to_hwc_f32(tensors[7]);
  const TensorHWC mk20 = tensor_to_hwc_f32(tensors[8]);
  proto = tensor_to_hwc_f32(tensors[9]);
  if (proto.c != 32) {
    throw std::runtime_error("unexpected prototype channels");
  }

  struct Level {
    const TensorHWC* reg;
    const TensorHWC* cls;
    const TensorHWC* mk;
  };
  const std::array<Level, 3> levels = {
      Level{&reg80, &cls80, &mk80},
      Level{&reg40, &cls40, &mk40},
      Level{&reg20, &cls20, &mk20},
  };

  std::vector<Box> cand;
  cand.reserve(2000);

  for (const auto& level : levels) {
    const TensorHWC& reg = *level.reg;
    const TensorHWC& cls = *level.cls;
    const TensorHWC& mk = *level.mk;
    if (reg.h != cls.h || reg.w != cls.w || reg.c != 64 || cls.c <= 0) {
      throw std::runtime_error("unexpected reg/cls tensor shapes for YOLOv8 decode");
    }
    if (reg.h != mk.h || reg.w != mk.w || mk.c != 32) {
      throw std::runtime_error("unexpected mask-coeff tensor shapes for YOLOv8 decode");
    }

    const float stride = static_cast<float>(infer_size) / static_cast<float>(reg.h);
    for (int y = 0; y < reg.h; ++y) {
      for (int x = 0; x < reg.w; ++x) {
        const size_t cls_base =
            (static_cast<size_t>(y) * static_cast<size_t>(reg.w) + static_cast<size_t>(x)) *
            static_cast<size_t>(cls.c);
        int best_cls = -1;
        float best_score = 0.0f;
        for (int c = 0; c < cls.c; ++c) {
          const float s = sigmoid(cls.data[cls_base + static_cast<size_t>(c)]);
          if (s > best_score) {
            best_score = s;
            best_cls = c;
          }
        }
        if (best_score < conf_thr) {
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

        Box box;
        box.x1 = std::max(0.0f, cx - l);
        box.y1 = std::max(0.0f, cy - t);
        box.x2 = std::min(static_cast<float>(infer_size), cx + r);
        box.y2 = std::min(static_cast<float>(infer_size), cy + b);
        box.score = best_score;
        box.class_id = best_cls;
        const size_t mk_base =
            (static_cast<size_t>(y) * static_cast<size_t>(mk.w) + static_cast<size_t>(x)) * 32U;
        for (int k = 0; k < 32; ++k) {
          box.coeff[static_cast<size_t>(k)] = mk.data[mk_base + static_cast<size_t>(k)];
        }
        if (box.x2 > box.x1 && box.y2 > box.y1) {
          cand.push_back(box);
        }
      }
    }
  }

  std::sort(cand.begin(), cand.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> keep;
  keep.reserve(static_cast<size_t>(max_det));
  for (const auto& b : cand) {
    bool suppressed = false;
    for (const auto& k : keep) {
      if (k.class_id == b.class_id && iou_xyxy(k, b) > nms_iou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(b);
      if (static_cast<int>(keep.size()) >= max_det)
        break;
    }
  }
  return keep;
}

void apply_mask_overlay(cv::Mat& bgr, const std::vector<Box>& dets, const TensorHWC& proto,
                        int infer_size, float alpha = 0.65f) {
  if (proto.c != 32) {
    throw std::runtime_error("unexpected prototype channels");
  }

  for (const auto& d : dets) {
    cv::Mat mask_small(proto.h, proto.w, CV_32FC1, cv::Scalar(0));
    for (int y = 0; y < proto.h; ++y) {
      float* row = mask_small.ptr<float>(y);
      for (int x = 0; x < proto.w; ++x) {
        float v = 0.0f;
        for (int k = 0; k < 32; ++k) {
          v += at_hwc(proto, y, x, k) * d.coeff[static_cast<size_t>(k)];
        }
        row[x] = sigmoid(v);
      }
    }

    const float sx = static_cast<float>(proto.w) / static_cast<float>(infer_size);
    const float sy = static_cast<float>(proto.h) / static_cast<float>(infer_size);
    const int bx1 = std::max(0, static_cast<int>(std::floor(d.x1 * sx)));
    const int by1 = std::max(0, static_cast<int>(std::floor(d.y1 * sy)));
    const int bx2 = std::min(proto.w - 1, static_cast<int>(std::ceil(d.x2 * sx)));
    const int by2 = std::min(proto.h - 1, static_cast<int>(std::ceil(d.y2 * sy)));

    cv::Mat mask_crop(proto.h, proto.w, CV_32FC1, cv::Scalar(0));
    if (bx2 >= bx1 && by2 >= by1) {
      const cv::Rect roi(bx1, by1, bx2 - bx1 + 1, by2 - by1 + 1);
      mask_small(roi).copyTo(mask_crop(roi));
    }

    cv::Mat mask;
    cv::resize(mask_crop, mask, bgr.size(), 0, 0, cv::INTER_LINEAR);
    const cv::Scalar color = class_color(d.class_id);
    cv::Mat contour_mask(bgr.rows, bgr.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < bgr.rows; ++y) {
      const float* m = mask.ptr<float>(y);
      cv::Vec3b* pix = bgr.ptr<cv::Vec3b>(y);
      uint8_t* contour_row = contour_mask.ptr<uint8_t>(y);
      for (int x = 0; x < bgr.cols; ++x) {
        if (m[x] <= 0.5f) {
          continue;
        }
        contour_row[x] = 255;
        for (int c = 0; c < 3; ++c) {
          pix[x][c] = static_cast<uint8_t>(
              std::round((1.0f - alpha) * static_cast<float>(pix[x][c]) +
                         alpha * static_cast<float>(color[c])));
        }
      }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contour_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {
      cv::drawContours(bgr, contours, -1, color, 2, cv::LINE_8);
    }
  }
}

void draw_boxes_on_image(cv::Mat& bgr, const std::vector<Box>& boxes, int infer_size) {
  const float sx = static_cast<float>(bgr.cols) / static_cast<float>(infer_size);
  const float sy = static_cast<float>(bgr.rows) / static_cast<float>(infer_size);
  for (const auto& b : boxes) {
    const int x1 = std::max(0, static_cast<int>(std::round(b.x1 * sx)));
    const int y1 = std::max(0, static_cast<int>(std::round(b.y1 * sy)));
    const int x2 = std::min(bgr.cols - 1, static_cast<int>(std::round(b.x2 * sx)));
    const int y2 = std::min(bgr.rows - 1, static_cast<int>(std::round(b.y2 * sy)));
    if (x2 <= x1 || y2 <= y1)
      continue;
    const cv::Scalar color = class_color(b.class_id);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    const std::string label =
        class_name_for_id(b.class_id) + " s=" + std::to_string(b.score).substr(0, 4);
    cv::putText(bgr, label, cv::Point(x1, std::max(0, y1 - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv::LINE_AA);
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <input_dir> <output_dir>\n";
    return 1;
  }

  try {
    constexpr int kInferSize = 640;
    constexpr float kScoreThr = 0.6f;
    constexpr float kNmsIou = 0.45f;
    constexpr int kMaxDet = 200;

    const std::string tar_gz = argv[1];
    const fs::path input_dir = argv[2];
    const fs::path output_dir = argv[3];

    if (!fs::is_directory(input_dir)) {
      throw std::runtime_error("input directory does not exist: " + input_dir.string());
    }
    fs::create_directories(output_dir);

    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
      if (entry.is_regular_file() && is_image(entry.path())) {
        images.push_back(entry.path());
      }
    }
    std::sort(images.begin(), images.end());
    if (images.empty()) {
      throw std::runtime_error("no images found in: " + input_dir.string());
    }

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "RGB";
    model_opt.preproc.input_width = kInferSize;
    model_opt.preproc.input_height = kInferSize;
    model_opt.preproc.input_img_type = "RGB";
    model_opt.input_max_width = kInferSize;
    model_opt.input_max_height = kInferSize;
    model_opt.input_max_depth = 3;

    simaai::neat::Model model(tar_gz, model_opt);

    simaai::neat::Session session;
    session.add(simaai::neat::nodes::Input(model.input_appsrc_options(false)));
    session.add(simaai::neat::nodes::groups::Preprocess(model));
    session.add(simaai::neat::nodes::groups::Infer(model));
    session.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
    session.add(simaai::neat::nodes::Output());

    std::cout << "Pipeline:\n" << session.describe_backend() << "\n";

    cv::Mat dummy_rgb(kInferSize, kInferSize, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor input_tensor = simaai::neat::from_cv_mat(
        dummy_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, /*read_only=*/true);

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = 8;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.preset = simaai::neat::RunPreset::Balanced;

    auto run = session.build(input_tensor, simaai::neat::RunMode::Async, run_opt);
    std::cout << "Found " << images.size() << " images\n";

    int processed = 0;
    for (const auto& image_path : images) {
      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        std::cerr << "Skipping unreadable image: " << image_path.filename() << "\n";
        continue;
      }

      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
      cv::resize(rgb, rgb, cv::Size(kInferSize, kInferSize), 0, 0, cv::INTER_LINEAR);

      simaai::neat::Tensor input = simaai::neat::from_cv_mat(
          rgb, simaai::neat::ImageSpec::PixelFormat::RGB, /*read_only=*/true);
      if (!run.push(input)) {
        std::cerr << "Push failed for: " << image_path.filename() << "\n";
        continue;
      }

      auto out = run.pull(/*timeout_ms=*/1000);
      if (!out.has_value()) {
        std::cerr << "Pull failed for: " << image_path.filename() << "\n";
        continue;
      }

      const std::vector<simaai::neat::Tensor> tensors = tensors_from_sample(*out);
      std::vector<Box> boxes;
      TensorHWC proto;
      try {
        boxes = decode_yolov8_instances_from_detess(
            tensors, kInferSize, kScoreThr, kNmsIou, kMaxDet, proto);
      } catch (const std::exception& e) {
        std::cerr << "Decode failed for " << image_path.filename() << ": " << e.what() << "\n";
        continue;
      }

      cv::Mat overlay = bgr.clone();
      apply_mask_overlay(overlay, boxes, proto, kInferSize);
      draw_boxes_on_image(overlay, boxes, kInferSize);
      const fs::path out_file = output_dir / (image_path.stem().string() + "_overlay.jpg");
      if (!cv::imwrite(out_file.string(), overlay)) {
        std::cerr << "Failed to write overlay image: " << out_file << "\n";
        continue;
      }

      std::cout << "Wrote: " << out_file << " boxes=" << boxes.size() << "\n";
      ++processed;
    }

    run.close();
    std::cout << "Processed " << processed << " / " << images.size() << " images\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
