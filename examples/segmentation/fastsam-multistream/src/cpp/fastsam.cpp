#include "fastsam.h"

#include <pipeline/DetectionTypes.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace neat = simaai::neat;

namespace app {
namespace {

constexpr double kMaskStride = 4.0; // proto grid is stride-4, so one mask cell = 4 model px
constexpr int kProtoMaskSide = 160; // proto-mask grid side (infer_size / kMaskStride)

std::vector<float> tensor_to_floats(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("tensor_to_floats: expected Float32 tensor");
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

std::vector<std::uint8_t> tensor_to_u8(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::UInt8) {
    throw std::runtime_error("tensor_to_u8: expected UInt8 tensor");
  }
  return tensor.copy_dense_bytes_tight();
}

neat::Tensor ev74_input_tensor(const cv::Mat& rgb) {
  return neat::Tensor::from_cv_mat(rgb, neat::ImageSpec::PixelFormat::RGB,
                                   neat::TensorMemory::EV74);
}

neat::Model::Options make_model_options(const AppConfig& cfg, int frame_w, int frame_h) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::RGB;
  opt.preprocess.input_max_width = frame_w;
  opt.preprocess.input_max_height = frame_h;
  opt.preprocess.input_max_depth = 3;
  opt.preprocess.resize.enable = neat::AutoFlag::On;
  opt.preprocess.resize.width = cfg.infer_size;
  opt.preprocess.resize.height = cfg.infer_size;
  opt.preprocess.resize.mode = neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = 114;
  opt.decode_type = neat::BoxDecodeType::YoloV26Seg;
  opt.score_threshold = static_cast<float>(cfg.score_threshold);
  opt.nms_iou_threshold = static_cast<float>(cfg.nms_iou);
  opt.top_k = cfg.max_detections;
  return opt;
}

neat::Model::Runner build_runner(neat::Model& model, const neat::RunOptions& run_opt, int frame_w,
                                 int frame_h) {
  const cv::Mat dummy = cv::Mat::zeros(frame_h, frame_w, CV_8UC3);
  return model.build(neat::TensorList{ev74_input_tensor(dummy)}, neat::Model::RouteOptions{},
                     run_opt);
}

} // namespace

Fastsam::Fastsam(const AppConfig& cfg, const neat::RunOptions& run_opt, int frame_w, int frame_h)
    : model_(
          std::make_unique<neat::Model>(cfg.model_path, make_model_options(cfg, frame_w, frame_h))),
      runner_(build_runner(*model_, run_opt, frame_w, frame_h)) {}

neat::TensorList Fastsam::run(const cv::Mat& rgb, int timeout_ms) {
  neat::TensorList input{ev74_input_tensor(rgb)}; // built outside the lock
  std::lock_guard<std::mutex> lock(mu_);
  return runner_.run(input, timeout_ms);
}

void Fastsam::close() {
  std::lock_guard<std::mutex> lock(mu_);
  runner_.close();
}

Fastsam::Geometry Fastsam::get_letterbox_geometry(int orig_w, int orig_h, int infer_size) {
  const double scale = static_cast<double>(infer_size) / std::max(orig_w, orig_h);
  Geometry g;
  g.scale = scale;
  g.pad_x = (infer_size - static_cast<int>(std::lround(orig_w * scale))) / 2;
  g.pad_y = (infer_size - static_cast<int>(std::lround(orig_h * scale))) / 2;
  return g;
}

Fastsam::Segmentation Fastsam::decode(const neat::TensorList& model_out, int top_k) {
  const auto decoded = neat::decode_segmentation(model_out, 0, 0, top_k, false);
  Segmentation seg;
  constexpr std::size_t kMaskBytes = static_cast<std::size_t>(kProtoMaskSide) * kProtoMaskSide;
  for (const auto& item : decoded) {
    const auto boxes = tensor_to_floats(item.boxes);
    const auto masks = tensor_to_u8(item.masks);
    const int count = static_cast<int>(boxes.size() / 6U);
    for (int i = 0; i < count; ++i) {
      const float* row = boxes.data() + static_cast<std::size_t>(i) * 6U;
      seg.boxes.push_back({row[0], row[1], row[2], row[3], row[4], row[5]});
      // mask buffer may be short; blank fallback avoids an OOB read and keeps masks 1:1 with boxes.
      if (masks.size() >= (static_cast<std::size_t>(i) + 1U) * kMaskBytes) {
        const cv::Mat view(
            kProtoMaskSide, kProtoMaskSide, CV_8UC1,
            const_cast<std::uint8_t*>(masks.data() + static_cast<std::size_t>(i) * kMaskBytes));
        seg.masks.push_back(view.clone());
      } else {
        seg.masks.push_back(cv::Mat::zeros(kProtoMaskSide, kProtoMaskSide, CV_8UC1));
      }
    }
  }
  return seg;
}

std::vector<cv::Point> Fastsam::mask_polygon(const cv::Mat& mask, const Geometry& geom,
                                             int max_points, double eps_frac) {
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const std::vector<cv::Point>* best = nullptr;
  double best_area = -1.0;
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area > best_area) {
      best_area = area;
      best = &contour;
    }
  }
  if (best == nullptr || best_area <= 0.0) {
    return {};
  }

  std::vector<cv::Point> approx;
  cv::approxPolyDP(*best, approx, eps_frac * cv::arcLength(*best, true), true);

  std::vector<cv::Point> pts;
  if (static_cast<int>(approx.size()) > max_points) {
    for (int k = 0; k < max_points; ++k) {
      const double t =
          static_cast<double>(k) * (approx.size() - 1) / (max_points - 1); // np.linspace + int()
      pts.push_back(approx[static_cast<std::size_t>(t)]);
    }
  } else {
    pts = std::move(approx);
  }
  if (pts.size() < 3) {
    return {};
  }

  std::vector<cv::Point> out;
  out.reserve(pts.size());
  for (const auto& p : pts) {
    out.emplace_back(static_cast<int>(std::lround((p.x * kMaskStride - geom.pad_x) / geom.scale)),
                     static_cast<int>(std::lround((p.y * kMaskStride - geom.pad_y) / geom.scale)));
  }
  return out;
}

std::optional<Fastsam::Crop> Fastsam::object_crop(const cv::Mat& proto_mask,
                                                  const cv::Mat& frame_rgb, const Geometry& geom,
                                                  double min_area, double max_frac,
                                                  double max_box_frac, double margin) {
  std::vector<cv::Point> nz;
  cv::findNonZero(proto_mask, nz); // points as (x=col, y=row)
  if (nz.empty()) {
    return std::nullopt;
  }
  const int orig_h = frame_rgb.rows;
  const int orig_w = frame_rgb.cols;
  const int mh = proto_mask.rows;
  const int mw = proto_mask.cols;
  const double cnt = static_cast<double>(nz.size());
  if (cnt * orig_w * orig_h < min_area * mw * mh) { // proto pixels -> frame-area threshold
    return std::nullopt;
  }
  if (cnt > max_frac * mw * mh) {
    return std::nullopt;
  }

  int px1 = std::numeric_limits<int>::max();
  int py1 = std::numeric_limits<int>::max();
  int px2 = 0;
  int py2 = 0;
  for (const auto& p : nz) {
    px1 = std::min(px1, p.x);
    py1 = std::min(py1, p.y);
    px2 = std::max(px2, p.x);
    py2 = std::max(py2, p.y);
  }
  px2 += 1;
  py2 += 1;

  const double fx1 = (px1 * kMaskStride - geom.pad_x) / geom.scale; // square -> original frame
  const double fy1 = (py1 * kMaskStride - geom.pad_y) / geom.scale;
  const double fx2 = (px2 * kMaskStride - geom.pad_x) / geom.scale;
  const double fy2 = (py2 * kMaskStride - geom.pad_y) / geom.scale;
  // Near-frame-sized boxes are background, not objects; drop them before CLIP sees them.
  const double box_w = std::min(fx2, static_cast<double>(orig_w)) - std::max(fx1, 0.0);
  const double box_h = std::min(fy2, static_cast<double>(orig_h)) - std::max(fy1, 0.0);
  if (box_w > max_box_frac * orig_w || box_h > max_box_frac * orig_h) {
    return std::nullopt;
  }
  const double dw = (fx2 - fx1) * margin;
  const double dh = (fy2 - fy1) * margin;
  const int x1 = std::max(0, static_cast<int>(fx1 - dw)); // int() truncates toward zero
  const int y1 = std::max(0, static_cast<int>(fy1 - dh));
  const int x2 = std::min(orig_w, static_cast<int>(std::lround(fx2 + dw)));
  const int y2 = std::min(orig_h, static_cast<int>(std::lround(fy2 + dh)));
  if (x2 <= x1 || y2 <= y1) {
    return std::nullopt;
  }

  Crop crop;
  crop.window = frame_rgb(cv::Rect(x1, y1, x2 - x1, y2 - y1)); // ROI view
  const double proto_to_frame = kMaskStride / geom.scale;
  const double center_offset = (proto_to_frame - 1.0) / 2.0;
  const cv::Mat transform =
      (cv::Mat_<double>(2, 3) << proto_to_frame, 0.0, -geom.pad_x / geom.scale - x1 + center_offset,
       0.0, proto_to_frame, -geom.pad_y / geom.scale - y1 + center_offset);
  cv::warpAffine(proto_mask, crop.submask, transform, crop.window.size(), cv::INTER_NEAREST,
                 cv::BORDER_CONSTANT, cv::Scalar(0));
  return crop;
}

} // namespace app
