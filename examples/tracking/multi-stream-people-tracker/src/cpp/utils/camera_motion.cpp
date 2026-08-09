#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/camera_motion_api.cpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace multi_stream_people_tracker {

namespace {

bool transform_is_plausible(const cv::Mat& affine, int width, int height, int inliers, int matches,
                            float residual) {
  if (affine.empty() || affine.rows != 2 || affine.cols != 3 || inliers < 8 || matches <= 0 ||
      static_cast<float>(inliers) / static_cast<float>(matches) < 0.25f ||
      !std::isfinite(residual) || residual > 3.0f) {
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

float reprojection_residual(const cv::Mat& affine, const std::vector<cv::Point2f>& previous,
                            const std::vector<cv::Point2f>& current, const cv::Mat& inlier_mask) {
  if (affine.empty() || inlier_mask.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  double residual_sum = 0.0;
  int count = 0;
  for (std::size_t index = 0; index < previous.size(); ++index) {
    if (inlier_mask.at<unsigned char>(static_cast<int>(index)) == 0) {
      continue;
    }
    const auto& source = previous[index];
    const auto& target = current[index];
    const double predicted_x = affine.at<double>(0, 0) * source.x +
                               affine.at<double>(0, 1) * source.y + affine.at<double>(0, 2);
    const double predicted_y = affine.at<double>(1, 0) * source.x +
                               affine.at<double>(1, 1) * source.y + affine.at<double>(1, 2);
    residual_sum += std::hypot(predicted_x - target.x, predicted_y - target.y);
    ++count;
  }
  return count > 0 ? static_cast<float>(residual_sum / count)
                   : std::numeric_limits<float>::infinity();
}

float spatial_coverage(const std::vector<cv::Point2f>& points, const cv::Mat& inlier_mask,
                       int width, int height) {
  if (points.empty() || width <= 0 || height <= 0) {
    return 0.0f;
  }
  float minimum_x = static_cast<float>(width);
  float minimum_y = static_cast<float>(height);
  float maximum_x = 0.0f;
  float maximum_y = 0.0f;
  int included = 0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    if (!inlier_mask.empty() && inlier_mask.at<unsigned char>(static_cast<int>(index)) == 0) {
      continue;
    }
    const auto& point = points[index];
    minimum_x = std::min(minimum_x, point.x);
    minimum_y = std::min(minimum_y, point.y);
    maximum_x = std::max(maximum_x, point.x);
    maximum_y = std::max(maximum_y, point.y);
    ++included;
  }
  if (included == 0) {
    return 0.0f;
  }
  return std::clamp((maximum_x - minimum_x) * (maximum_y - minimum_y) /
                        static_cast<float>(width * height),
                    0.0f, 1.0f);
}

} // namespace

struct FrameCameraMotionEstimator::Impl {
  explicit Impl(int requested_downscale, int requested_max_features)
      : downscale(std::max(1, requested_downscale)),
        max_features(std::max(32, requested_max_features)) {}

  int downscale = 4;
  int max_features = 200;
  cv::Mat previous_gray;
  std::vector<cv::Point2f> previous_points;
  std::vector<cv::Point2f> forward_points;
  std::vector<cv::Point2f> backward_points;
  std::vector<cv::Point2f> matched_previous;
  std::vector<cv::Point2f> matched_current;
  std::vector<unsigned char> forward_status;
  std::vector<unsigned char> backward_status;
  std::vector<float> forward_error;
  std::vector<float> backward_error;
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
  impl_->previous_gray.release();
  impl_->previous_points.clear();
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

  // Estimate background/camera motion, not object motion. Detector boxes and a
  // thin feature-unstable border are excluded from corner selection.
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

  CameraTransform result;
  if (!impl_->previous_gray.empty() && impl_->previous_points.size() >= 8) {
    const cv::Size kWindow(15, 15);
    constexpr int kPyramidLevels = 2;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::calcOpticalFlowPyrLK(impl_->previous_gray, gray, impl_->previous_points,
                             impl_->forward_points, impl_->forward_status, impl_->forward_error,
                             kWindow, kPyramidLevels, criteria, 0, 1.0e-4);
    cv::calcOpticalFlowPyrLK(gray, impl_->previous_gray, impl_->forward_points,
                             impl_->backward_points, impl_->backward_status, impl_->backward_error,
                             kWindow, kPyramidLevels, criteria, 0, 1.0e-4);
    impl_->matched_previous.clear();
    impl_->matched_current.clear();
    impl_->matched_previous.reserve(impl_->previous_points.size());
    impl_->matched_current.reserve(impl_->previous_points.size());
    for (std::size_t index = 0; index < impl_->previous_points.size(); ++index) {
      if (impl_->forward_status[index] == 0 || impl_->backward_status[index] == 0 ||
          cv::norm(impl_->previous_points[index] - impl_->backward_points[index]) > 1.0) {
        continue;
      }
      impl_->matched_previous.push_back(impl_->previous_points[index]);
      impl_->matched_current.push_back(impl_->forward_points[index]);
    }

    if (impl_->matched_previous.size() >= 8) {
      cv::Mat inlier_mask;
      cv::Mat affine = cv::estimateAffinePartial2D(impl_->matched_previous, impl_->matched_current,
                                                   inlier_mask, cv::RANSAC, 2.0, 300, 0.995, 10);
      const int inliers = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
      const float residual = reprojection_residual(affine, impl_->matched_previous,
                                                   impl_->matched_current, inlier_mask);
      const float coverage =
          spatial_coverage(impl_->matched_previous, inlier_mask, gray.cols, gray.rows);
      if (transform_is_plausible(affine, gray.cols, gray.rows, inliers,
                                 static_cast<int>(impl_->matched_previous.size()), residual)) {
        const float scale = static_cast<float>(impl_->downscale);
        const float inlier_ratio =
            static_cast<float>(inliers) / static_cast<float>(impl_->matched_previous.size());
        const float confidence =
            std::clamp(inlier_ratio * std::min(1.0f, coverage / 0.25f) * std::exp(-0.5f * residual),
                       0.0f, 1.0f);
        result = CameraTransform{static_cast<float>(affine.at<double>(0, 0)),
                                 static_cast<float>(affine.at<double>(0, 1)),
                                 scale * static_cast<float>(affine.at<double>(0, 2)),
                                 static_cast<float>(affine.at<double>(1, 0)),
                                 static_cast<float>(affine.at<double>(1, 1)),
                                 scale * static_cast<float>(affine.at<double>(1, 2)),
                                 true,
                                 confidence,
                                 scale * residual,
                                 inliers};
      }
    }
  }

  impl_->current_points.clear();
  cv::goodFeaturesToTrack(gray, impl_->current_points, impl_->max_features, 0.01, 4.0, feature_mask,
                          3, false, 0.04);
  impl_->previous_points = std::move(impl_->current_points);
  gray.copyTo(impl_->previous_gray);
  return result;
}

} // namespace multi_stream_people_tracker
