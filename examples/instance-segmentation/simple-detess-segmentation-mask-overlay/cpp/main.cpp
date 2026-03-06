/**
 * @example simple-detess-segmentation-mask-overlay.cpp
 * Minimal YOLOv5 instance-segmentation overlay from DetessDequant outputs.
 *
 * Usage:
 *   simple-detess-segmentation-mask-overlay <model.tar.gz> <input_dir> <output_dir>
 */
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

struct DenseTensor {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  std::array<float, 32> coeff{};
};

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

const std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& s) {
  std::vector<simaai::neat::Tensor> out;
  if (s.kind == simaai::neat::SampleKind::Tensor && s.tensor.has_value()) {
    out.push_back(*s.tensor);
    return out;
  }
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& f : s.fields) {
      auto t = collect_tensors(f);
      out.insert(out.end(), t.begin(), t.end());
    }
  }
  return out;
}

bool tensor_to_hwc(const simaai::neat::Tensor& t, DenseTensor& out, std::string& err) {
  if (!t.is_dense()) {
    err = "tensor is not dense";
    return false;
  }
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    err = "tensor dtype is not Float32";
    return false;
  }

  if (t.shape.size() == 4) {
    if (t.shape[0] != 1) {
      err = "only batch=1 is supported";
      return false;
    }
    out.h = static_cast<int>(t.shape[1]);
    out.w = static_cast<int>(t.shape[2]);
    out.c = static_cast<int>(t.shape[3]);
  } else if (t.shape.size() == 3) {
    out.h = static_cast<int>(t.shape[0]);
    out.w = static_cast<int>(t.shape[1]);
    out.c = static_cast<int>(t.shape[2]);
  } else {
    err = "unexpected tensor rank";
    return false;
  }

  if (out.h <= 0 || out.w <= 0 || out.c <= 0) {
    err = "invalid tensor shape";
    return false;
  }

  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  const size_t elems =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  if (raw.size() < elems * sizeof(float)) {
    err = "tensor byte size mismatch";
    return false;
  }

  out.data.resize(elems);
  std::memcpy(out.data.data(), raw.data(), elems * sizeof(float));
  return true;
}

inline float at_hwc(const DenseTensor& t, int y, int x, int ch) {
  const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(t.w) + static_cast<size_t>(x)) *
                         static_cast<size_t>(t.c) +
                     static_cast<size_t>(ch);
  return t.data[idx];
}

float iou(const Detection& a, const Detection& b) {
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

std::vector<Detection> nms_per_class(std::vector<Detection> dets, float iou_thr, int max_det) {
  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  keep.reserve(static_cast<size_t>(max_det));
  for (const auto& d : dets) {
    bool suppressed = false;
    for (const auto& k : keep) {
      if (k.class_id == d.class_id && iou(k, d) > iou_thr) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(d);
      if (static_cast<int>(keep.size()) >= max_det)
        break;
    }
  }
  return keep;
}

bool decode_yolov5_seg(const std::vector<simaai::neat::Tensor>& tensors, int infer_size,
                       std::vector<Detection>& dets, DenseTensor& proto, std::string& err) {
  if (tensors.size() < 13) {
    err = "expected 13 output tensors";
    return false;
  }

  // Expected order from MPK detessdequant:
  // 0 proto(160x160x32),
  // 1..4 stride8: xy(6), wh(6), cls+obj(243), coeff(96)
  // 5..8 stride16, 9..12 stride32.
  if (!tensor_to_hwc(tensors[0], proto, err))
    return false;
  if (proto.h != 160 || proto.w != 160 || proto.c != 32) {
    err = "unexpected proto shape";
    return false;
  }

  constexpr float kConfThr = 0.35f;
  for (int lvl = 0; lvl < 3; ++lvl) {
    DenseTensor txy, twh, tco, tmk;
    if (!tensor_to_hwc(tensors[1 + lvl * 4], txy, err) ||
        !tensor_to_hwc(tensors[2 + lvl * 4], twh, err) ||
        !tensor_to_hwc(tensors[3 + lvl * 4], tco, err) ||
        !tensor_to_hwc(tensors[4 + lvl * 4], tmk, err)) {
      return false;
    }

    if (txy.h != twh.h || txy.w != twh.w || txy.h != tco.h || txy.w != tco.w || txy.h != tmk.h ||
        txy.w != tmk.w) {
      err = "mismatched head shapes";
      return false;
    }
    if (txy.c != 6 || twh.c != 6 || tco.c != 243 || tmk.c != 96) {
      err = "unexpected channel layout for yolov5 head";
      return false;
    }

    const int gh = txy.h;
    const int gw = txy.w;
    for (int y = 0; y < gh; ++y) {
      for (int x = 0; x < gw; ++x) {
        for (int a = 0; a < 3; ++a) {
          const float tx = at_hwc(txy, y, x, a * 2 + 0);
          const float ty = at_hwc(txy, y, x, a * 2 + 1);
          const float tw = at_hwc(twh, y, x, a * 2 + 0);
          const float th = at_hwc(twh, y, x, a * 2 + 1);

          const int cls_base = a * 81;
          // DetessDequant already emits dequantized box values and scores.
          const float obj = std::clamp(at_hwc(tco, y, x, cls_base + 0), 0.0f, 1.0f);
          int best_cls = 0;
          float best_cls_score = 0.0f;
          for (int c = 0; c < 80; ++c) {
            const float p = std::clamp(at_hwc(tco, y, x, cls_base + 1 + c), 0.0f, 1.0f);
            if (p > best_cls_score) {
              best_cls_score = p;
              best_cls = c;
            }
          }

          const float score = obj * best_cls_score;
          if (score < kConfThr)
            continue;

          const float cx = tx;
          const float cy = ty;
          const float bw = std::max(0.0f, tw);
          const float bh = std::max(0.0f, th);

          Detection d;
          d.x1 = std::max(0.0f, cx - bw * 0.5f);
          d.y1 = std::max(0.0f, cy - bh * 0.5f);
          d.x2 = std::min(static_cast<float>(infer_size), cx + bw * 0.5f);
          d.y2 = std::min(static_cast<float>(infer_size), cy + bh * 0.5f);
          d.score = score;
          d.class_id = best_cls;
          if (d.x2 <= d.x1 || d.y2 <= d.y1)
            continue;

          const int mask_base = a * 32;
          for (int k = 0; k < 32; ++k) {
            d.coeff[static_cast<size_t>(k)] = at_hwc(tmk, y, x, mask_base + k);
          }
          dets.push_back(d);
        }
      }
    }
  }

  dets = nms_per_class(std::move(dets), 0.5f, 100);
  return true;
}

cv::Scalar class_color(int cid) {
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
  if (cid < 0) {
    cid = 0;
  }
  return kPalette[static_cast<size_t>(cid) % kPalette.size()];
}

std::string class_name(int cid) {
  static const char* kCoco80[] = {
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
      "hair drier",    "toothbrush",
  };
  if (cid >= 0 && cid < 80) {
    return kCoco80[cid];
  }
  return "class_" + std::to_string(cid);
}

void apply_mask_overlay(cv::Mat& bgr, const std::vector<Detection>& dets, const DenseTensor& proto,
                        int infer_size) {
  constexpr float kMaskAlpha = 0.65f;
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

    const float scale = static_cast<float>(proto.w) / static_cast<float>(infer_size); // 160/640
    const int bx1 = std::max(0, static_cast<int>(std::floor(d.x1 * scale)));
    const int by1 = std::max(0, static_cast<int>(std::floor(d.y1 * scale)));
    const int bx2 = std::min(proto.w - 1, static_cast<int>(std::ceil(d.x2 * scale)));
    const int by2 = std::min(proto.h - 1, static_cast<int>(std::ceil(d.y2 * scale)));
    for (int y = 0; y < proto.h; ++y) {
      float* row = mask_small.ptr<float>(y);
      for (int x = 0; x < proto.w; ++x) {
        if (x < bx1 || x > bx2 || y < by1 || y > by2) {
          row[x] = 0.0f;
        }
      }
    }

    cv::Mat mask;
    cv::resize(mask_small, mask, cv::Size(bgr.cols, bgr.rows), 0, 0, cv::INTER_LINEAR);
    const cv::Scalar col = class_color(d.class_id);
    bool any_mask = false;
    for (int y = 0; y < bgr.rows; ++y) {
      cv::Vec3b* pix = bgr.ptr<cv::Vec3b>(y);
      const float* mrow = mask.ptr<float>(y);
      for (int x = 0; x < bgr.cols; ++x) {
        if (mrow[x] > 0.5f) {
          any_mask = true;
          pix[x][0] = static_cast<uint8_t>((1.0f - kMaskAlpha) * pix[x][0] + kMaskAlpha * col[0]);
          pix[x][1] = static_cast<uint8_t>((1.0f - kMaskAlpha) * pix[x][1] + kMaskAlpha * col[1]);
          pix[x][2] = static_cast<uint8_t>((1.0f - kMaskAlpha) * pix[x][2] + kMaskAlpha * col[2]);
        }
      }
    }

    if (!any_mask) {
      continue;
    }

    cv::Mat contour_mask;
    cv::threshold(mask, contour_mask, 0.5, 255.0, cv::THRESH_BINARY);
    contour_mask.convertTo(contour_mask, CV_8UC1);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(contour_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty()) {
      cv::drawContours(bgr, contours, -1, col, 2, cv::LINE_8);
    }
  }
}

void draw_bboxes(cv::Mat& bgr, const std::vector<Detection>& dets, int infer_size) {
  const float sx = static_cast<float>(bgr.cols) / static_cast<float>(infer_size);
  const float sy = static_cast<float>(bgr.rows) / static_cast<float>(infer_size);
  for (const auto& d : dets) {
    const int x1 = std::max(0, static_cast<int>(std::round(d.x1 * sx)));
    const int y1 = std::max(0, static_cast<int>(std::round(d.y1 * sy)));
    const int x2 = std::min(bgr.cols - 1, static_cast<int>(std::round(d.x2 * sx)));
    const int y2 = std::min(bgr.rows - 1, static_cast<int>(std::round(d.y2 * sy)));
    if (x2 <= x1 || y2 <= y1)
      continue;
    const cv::Scalar col = class_color(d.class_id);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), col, 2);
    const std::string label = class_name(d.class_id) + " s=" + cv::format("%.2f", d.score);
    cv::putText(bgr, label, cv::Point(x1, std::max(0, y1 - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                col, 1, cv::LINE_AA);
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

  constexpr int kInputW = 640;
  constexpr int kInputH = 640;

  const std::string model_path = argv[1];
  const fs::path input_dir = argv[2];
  const fs::path output_dir = argv[3];

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
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
    std::cerr << "No images found in " << input_dir << "\n";
    return 3;
  }

  try {
    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "RGB";
    model_opt.preproc.input_width = kInputW;
    model_opt.preproc.input_height = kInputH;
    model_opt.preproc.input_img_type = "RGB";
    model_opt.input_max_width = kInputW;
    model_opt.input_max_height = kInputH;
    model_opt.input_max_depth = 3;

    simaai::neat::Model model(model_path, model_opt);

    simaai::neat::Session session;
    session.add(model.session());

    cv::Mat dummy_rgb(kInputH, kInputW, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy =
        simaai::neat::from_cv_mat(dummy_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);
    auto run = session.build(dummy, simaai::neat::RunMode::Sync);

    std::cout << "Pipeline:\n" << session.describe_backend() << "\n";
    std::cout << "Found " << images.size() << " images\n";

    int ok = 0;
    for (const auto& image_path : images) {
      cv::Mat src_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (src_bgr.empty()) {
        std::cerr << "Skipping unreadable image: " << image_path.filename() << "\n";
        continue;
      }

      cv::Mat resized_bgr;
      cv::resize(src_bgr, resized_bgr, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);

      cv::Mat resized_rgb;
      cv::cvtColor(resized_bgr, resized_rgb, cv::COLOR_BGR2RGB);
      simaai::neat::Tensor input =
          simaai::neat::from_cv_mat(resized_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);

      simaai::neat::Sample out = run.push_and_pull(input, /*timeout_ms=*/3000);
      const std::vector<simaai::neat::Tensor> tensors = collect_tensors(out);
      if (tensors.empty()) {
        std::cerr << "No tensor outputs for " << image_path.filename() << "\n";
        continue;
      }

      std::vector<Detection> dets;
      DenseTensor proto;
      std::string err;
      if (!decode_yolov5_seg(tensors, kInputW, dets, proto, err)) {
        std::cerr << "Decode failed for " << image_path.filename() << ": " << err << "\n";
        continue;
      }

      cv::Mat overlay = resized_bgr.clone();
      apply_mask_overlay(overlay, dets, proto, kInputW);
      draw_bboxes(overlay, dets, kInputW);
      const fs::path overlay_path = output_dir / (image_path.stem().string() + "_overlay.jpg");
      if (!cv::imwrite(overlay_path.string(), overlay)) {
        std::cerr << "Failed to write " << overlay_path << "\n";
        continue;
      }

      std::cout << "Wrote: " << overlay_path << " detections=" << dets.size() << "\n";
      ++ok;
    }

    run.close();
    std::cout << "Processed " << ok << " / " << images.size() << " images\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
