#pragma once

#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/pipeline_api.cpp"
#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/tracker_api.cpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <filesystem>
#include <optional>
#include <vector>

namespace multi_camera_people_tracking {

struct QuantTessCpuPreprocState {
  int src_width = 0;
  int src_height = 0;
  int scaled_w = 0;
  int scaled_h = 0;
  int pad_x = 0;
  int pad_y = 0;
  cv::Mat quant_input;
};

std::filesystem::path sample_output_path(const std::filesystem::path& output_dir, int stream_index,
                                         int frame_index);
QuantTessCpuPreprocState build_cpu_quanttess_preproc_state(
    const QuantTessCpuPreproc& contract, int src_width, int src_height);
cv::Mat cpu_quanttess_input(const cv::Mat& frame_rgb, QuantTessCpuPreprocState& state);
cv::Scalar class_color(int track_id);
cv::Mat draw_tracked_people(cv::Mat frame, const std::vector<TrackedDetection>& tracked);
bool save_overlay_frame(const std::optional<std::filesystem::path>& output_dir, int stream_index,
                        int frame_index, const cv::Mat& frame, int save_every);

} // namespace multi_camera_people_tracking
