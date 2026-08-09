#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/camera_motion_api.cpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace multi_stream_people_tracker {

namespace {

bool transform_is_plausible(const cv::Mat& affine, int width, int height, int inliers,
                            int matches) {
  if (affine.empty() || affine.rows != 2 || affine.cols != 3 || inliers < 8 || matches <= 0 ||
      static_cast<float>(inliers) / static_cast<float>(matches) < 0.25f) {
    return false;
  }
  const double a = affine.at<double>(0, 0);
  const double b = affine.at<double>(0, 1);
  const double tx = affine.at<double>(0, 2);
  const double c = affine.at<double>(1, 0);
  const double d = affine.at<double>(1, 1);
  const double ty = affine.at<double>(1, 2);
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(tx) || !std::isfinite(c) ||
      !std::isfinite(d) || !std::isfinite(ty)) {
    return false;
  }
  const double scale_x = std::hypot(a, c);
  const double scale_y = std::hypot(b, d);
  const double rotation = std::abs(std::atan2(c, a));
  return scale_x >= 0.85 && scale_x <= 1.15 && scale_y >= 0.85 && scale_y <= 1.15 &&
         rotation <= 0.20 && std::abs(tx) <= 0.5 * width && std::abs(ty) <= 0.5 * height;
}

} // namespace

struct FrameCameraMotionEstimator::Impl {
  explicit Impl(int requested_downscale, int requested_max_features)
      : downscale(std::max(1, requested_downscale)),
        orb(cv::ORB::create(std::max(64, requested_max_features), 1.2f, 4, 15, 0, 2,
                            cv::ORB::HARRIS_SCORE, 15, 20)),
        matcher(cv::NORM_HAMMING, false) {}

  int downscale = 2;
  cv::Ptr<cv::ORB> orb;
  cv::BFMatcher matcher;
  std::vector<cv::KeyPoint> previous_keypoints;
  cv::Mat previous_descriptors;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<std::vector<cv::DMatch>> knn_matches;
  std::vector<cv::Point2f> previous_points;
  std::vector<cv::Point2f> current_points;
};

FrameCameraMotionEstimator::FrameCameraMotionEstimator(int downscale, int max_features)
    : impl_(std::make_unique<Impl>(downscale, max_features)) {}

FrameCameraMotionEstimator::~FrameCameraMotionEstimator() = default;
FrameCameraMotionEstimator::FrameCameraMotionEstimator(FrameCameraMotionEstimator&&) noexcept =
    default;
FrameCameraMotionEstimator&
FrameCameraMotionEstimator::operator=(FrameCameraMotionEstimator&&) noexcept = default;

void FrameCameraMotionEstimator::reset() {
  impl_->previous_keypoints.clear();
  impl_->previous_descriptors.release();
}

CameraTransform FrameCameraMotionEstimator::update(const cv::Mat& gray_frame,
                                                   const std::vector<Detection>& detections) {
  if (gray_frame.empty() || gray_frame.type() != CV_8UC1) {
    reset();
    return {};
  }

  cv::Mat gray;
  if (impl_->downscale > 1) {
    cv::resize(gray_frame, gray,
               cv::Size(gray_frame.cols / impl_->downscale, gray_frame.rows / impl_->downscale),
               0.0, 0.0, cv::INTER_AREA);
  } else {
    gray = gray_frame;
  }

  // Estimate background/camera motion, not object motion. This mirrors the
  // BoT-SORT/PineSORT GMC contract by excluding detector boxes and a thin,
  // feature-unstable image border from ORB extraction.
  cv::Mat feature_mask(gray.size(), CV_8UC1, cv::Scalar(0));
  const int border_x = gray.cols >= 3 ? std::max(1, gray.cols / 50) : 0;
  const int border_y = gray.rows >= 3 ? std::max(1, gray.rows / 50) : 0;
  feature_mask(cv::Rect(border_x, border_y, gray.cols - 2 * border_x, gray.rows - 2 * border_y))
      .setTo(255);
  const cv::Rect frame_bounds(0, 0, gray.cols, gray.rows);
  for (const auto& detection : detections) {
    const float scale = static_cast<float>(impl_->downscale);
    const int x1 = static_cast<int>(std::floor(detection.x1 / scale)) - 2;
    const int y1 = static_cast<int>(std::floor(detection.y1 / scale)) - 2;
    const int x2 = static_cast<int>(std::ceil(detection.x2 / scale)) + 2;
    const int y2 = static_cast<int>(std::ceil(detection.y2 / scale)) + 2;
    if (x2 > x1 && y2 > y1) {
      const cv::Rect object_bounds = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)) & frame_bounds;
      if (!object_bounds.empty()) {
        feature_mask(object_bounds).setTo(0);
      }
    }
  }

  impl_->keypoints.clear();
  impl_->descriptors.release();
  impl_->orb->detectAndCompute(gray, feature_mask, impl_->keypoints, impl_->descriptors);

  CameraTransform result;
  if (!impl_->previous_descriptors.empty() && !impl_->descriptors.empty() &&
      impl_->previous_keypoints.size() >= 8 && impl_->keypoints.size() >= 8) {
    impl_->knn_matches.clear();
    impl_->matcher.knnMatch(impl_->previous_descriptors, impl_->descriptors, impl_->knn_matches, 2);
    impl_->previous_points.clear();
    impl_->current_points.clear();
    impl_->previous_points.reserve(impl_->knn_matches.size());
    impl_->current_points.reserve(impl_->knn_matches.size());
    for (const auto& pair : impl_->knn_matches) {
      if (pair.size() < 2 || pair[0].distance >= 0.80f * pair[1].distance) {
        continue;
      }
      impl_->previous_points.push_back(
          impl_->previous_keypoints[static_cast<std::size_t>(pair[0].queryIdx)].pt);
      impl_->current_points.push_back(
          impl_->keypoints[static_cast<std::size_t>(pair[0].trainIdx)].pt);
    }

    if (impl_->previous_points.size() >= 8) {
      cv::Mat inlier_mask;
      cv::Mat affine = cv::estimateAffinePartial2D(impl_->previous_points, impl_->current_points,
                                                   inlier_mask, cv::RANSAC, 2.5, 500, 0.99, 10);
      const int inliers = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
      if (transform_is_plausible(affine, gray.cols, gray.rows, inliers,
                                 static_cast<int>(impl_->previous_points.size()))) {
        const float scale = static_cast<float>(impl_->downscale);
        result = CameraTransform{static_cast<float>(affine.at<double>(0, 0)),
                                 static_cast<float>(affine.at<double>(0, 1)),
                                 scale * static_cast<float>(affine.at<double>(0, 2)),
                                 static_cast<float>(affine.at<double>(1, 0)),
                                 static_cast<float>(affine.at<double>(1, 1)),
                                 scale * static_cast<float>(affine.at<double>(1, 2)),
                                 true};
      }
    }
  }

  impl_->previous_keypoints = std::move(impl_->keypoints);
  impl_->previous_descriptors = impl_->descriptors.clone();
  return result;
}

} // namespace multi_stream_people_tracker
