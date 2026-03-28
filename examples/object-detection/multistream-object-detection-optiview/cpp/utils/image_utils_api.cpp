#pragma once

#include "sample_utils_api.cpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <filesystem>
#include <optional>
#include <vector>

namespace multistream_yolox_yolov8_optiview {

std::filesystem::path sample_output_path(const std::filesystem::path& output_dir, int stream_index,
                                         int frame_index);
cv::Scalar class_color(int class_id);
cv::Mat draw_detection_boxes(cv::Mat frame, const std::vector<Detection>& detections,
                             const std::vector<std::string>& class_labels);
bool save_debug_frame(const std::optional<std::filesystem::path>& output_dir, int stream_index,
                      int frame_index, const cv::Mat& frame, int save_every);

} // namespace multistream_yolox_yolov8_optiview
