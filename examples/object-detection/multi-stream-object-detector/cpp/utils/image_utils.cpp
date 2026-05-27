#include "image_utils_api.cpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace fs = std::filesystem;

namespace multi_stream_object_detector {
namespace {

std::string class_label(const std::vector<std::string>& class_labels, int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(class_labels.size())) {
    return class_labels[static_cast<std::size_t>(class_id)];
  }
  return std::to_string(class_id);
}

} // namespace

std::filesystem::path sample_output_path(const std::filesystem::path& output_dir, int stream_index,
                                         int frame_index) {
  return output_dir / ("stream_" + std::to_string(stream_index)) /
         ("frame_" + cv::format("%06d", frame_index) + ".jpg");
}

cv::Scalar class_color(int class_id) {
  return cv::Scalar((37 * class_id + 17) % 256, (97 * class_id + 73) % 256,
                    (53 * class_id + 191) % 256);
}

cv::Mat draw_detection_boxes(cv::Mat frame, const std::vector<Detection>& detections,
                             const std::vector<std::string>& class_labels) {
  for (const auto& det : detections) {
    const cv::Scalar color = class_color(det.class_id);
    const int x1 = static_cast<int>(det.x1);
    const int y1 = static_cast<int>(det.y1);
    const int x2 = static_cast<int>(det.x2);
    const int y2 = static_cast<int>(det.y2);
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const std::string text =
        class_label(class_labels, det.class_id) + " " + cv::format("%.2f", det.score);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::rectangle(frame, cv::Point(x1, std::max(0, y1 - text_size.height - 4)),
                  cv::Point(x1 + text_size.width, y1), color, cv::FILLED);
    cv::putText(frame, text, cv::Point(x1, std::max(0, y1 - 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
  return frame;
}

bool save_debug_frame(const std::optional<fs::path>& output_dir, int stream_index, int frame_index,
                      const cv::Mat& frame, int save_every) {
  if (!output_dir.has_value()) {
    return false;
  }
  if (save_every <= 0 || (frame_index % save_every) != 0) {
    return false;
  }
  const fs::path out_path = sample_output_path(*output_dir, stream_index, frame_index);
  fs::create_directories(out_path.parent_path());
  cv::Mat frame_bgr;
  cv::cvtColor(frame, frame_bgr, cv::COLOR_RGB2BGR);
  return cv::imwrite(out_path.string(), frame_bgr);
}

} // namespace multi_stream_object_detector
