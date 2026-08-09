#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_api.cpp"

namespace multi_stream_people_tracker {

// Estimates a confidence-bearing partial-affine transform between consecutive
// decoded frames using sparse pyramidal optical flow on luminance.
class FrameCameraMotionEstimator {
public:
  explicit FrameCameraMotionEstimator(int downscale = 4, int max_features = 200);
  ~FrameCameraMotionEstimator();

  FrameCameraMotionEstimator(FrameCameraMotionEstimator&&) noexcept;
  FrameCameraMotionEstimator& operator=(FrameCameraMotionEstimator&&) noexcept;
  FrameCameraMotionEstimator(const FrameCameraMotionEstimator&) = delete;
  FrameCameraMotionEstimator& operator=(const FrameCameraMotionEstimator&) = delete;

  CameraTransform update(const cv::Mat& gray_frame, const std::vector<Detection>& detections = {});
  void reset();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace multi_stream_people_tracker
