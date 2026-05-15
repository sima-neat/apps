#pragma once

#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/utils/tracker_api.cpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <filesystem>
#include <optional>
#include <vector>

namespace multi_camera_people_tracking {

std::filesystem::path sample_output_path(const std::filesystem::path& output_dir, int stream_index,
                                         int frame_index);
cv::Scalar class_color(int track_id);
cv::Mat draw_tracked_people(cv::Mat frame, const std::vector<TrackedDetection>& tracked);
bool save_overlay_frame(const std::optional<std::filesystem::path>& output_dir, int stream_index,
                        int frame_index, const cv::Mat& frame, int save_every);

} // namespace multi_camera_people_tracking
