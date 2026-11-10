#include "align.h"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace face_recog {

namespace {

// Landmark positions in the 112x112 ArcFace input, from InsightFace: left eye, right eye,
// nose tip, left mouth corner, right mouth corner.
constexpr float kTemplate[5][2] = {
    {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
    {41.5493f, 92.3655f}, {70.7299f, 92.2041f},
};

// Least-squares similarity transform (rotation, uniform scale, translation) that maps the
// landmarks onto the template:
//   x' = a*x - b*y + c
//   y' = b*x + a*y + d
cv::Mat similarity_transform(const Landmarks& landmarks) {
  constexpr int n = 5;
  double sum_x = 0, sum_y = 0, sum_u = 0, sum_v = 0;
  double sum_sq = 0, sum_dot = 0, sum_cross = 0;
  for (int i = 0; i < n; ++i) {
    const double x = landmarks[i * 2], y = landmarks[i * 2 + 1];
    const double u = kTemplate[i][0], v = kTemplate[i][1];
    sum_x += x;
    sum_y += y;
    sum_u += u;
    sum_v += v;
    sum_sq += x * x + y * y;
    sum_dot += x * u + y * v;
    sum_cross += x * v - y * u;
  }
  const double spread = sum_sq - (sum_x * sum_x + sum_y * sum_y) / n;
  if (spread <= 0.0) {
    throw std::runtime_error("face landmarks are degenerate");
  }
  const double a = (sum_dot - (sum_x * sum_u + sum_y * sum_v) / n) / spread;
  const double b = (sum_cross - (sum_x * sum_v - sum_y * sum_u) / n) / spread;
  const double c = (sum_u - a * sum_x + b * sum_y) / n;
  const double d = (sum_v - b * sum_x - a * sum_y) / n;
  return (cv::Mat_<double>(2, 3) << a, -b, c, b, a, d);
}

} // namespace

cv::Mat align_face(const cv::Mat& bgr, const Landmarks& landmarks) {
  if (bgr.empty()) {
    throw std::runtime_error("cannot align a face in an empty frame");
  }
  cv::Mat crop;
  cv::warpAffine(bgr, crop, similarity_transform(landmarks), cv::Size(kArcFaceW, kArcFaceH),
                 cv::INTER_LINEAR, cv::BORDER_REFLECT);
  return crop;
}

cv::Mat preprocess_arcface_crop(const cv::Mat& crop_bgr) {
  if (crop_bgr.cols != kArcFaceW || crop_bgr.rows != kArcFaceH) {
    throw std::runtime_error("ArcFace expects a 112x112 crop");
  }
  cv::Mat rgb;
  cv::cvtColor(crop_bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat input;
  rgb.convertTo(input, CV_32FC3, 1.0 / 127.5, -1.0);
  return input;
}

} // namespace face_recog
