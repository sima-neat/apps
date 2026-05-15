#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/utils/image_utils_api.cpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace fs = std::filesystem;

namespace multi_camera_people_tracking {

std::filesystem::path sample_output_path(const std::filesystem::path& output_dir, int stream_index,
                                         int frame_index) {
  return output_dir / ("stream_" + std::to_string(stream_index)) /
         ("frame_" + cv::format("%06d", frame_index) + ".jpg");
}

cv::Scalar class_color(int track_id) {
  return cv::Scalar((37 * track_id + 17) % 256, (97 * track_id + 73) % 256,
                    (53 * track_id + 191) % 256);
}

cv::Mat draw_tracked_people(cv::Mat frame, const std::vector<TrackedDetection>& tracked) {
  for (const auto& det : tracked) {
    const cv::Scalar color = class_color(det.track_id);
    const int x1 = static_cast<int>(det.x1);
    const int y1 = static_cast<int>(det.y1);
    const int x2 = static_cast<int>(det.x2);
    const int y2 = static_cast<int>(det.y2);
    const std::string text =
        "person #" + std::to_string(det.track_id) + " " + cv::format("%.2f", det.score);
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

bool save_overlay_frame(const std::optional<fs::path>& output_dir, int stream_index,
                        int frame_index, const cv::Mat& frame, int save_every) {
  if (!output_dir.has_value()) {
    return false;
  }
  if (save_every <= 0 || (frame_index % save_every) != 0) {
    return false;
  }

  const fs::path out_path = sample_output_path(*output_dir, stream_index, frame_index);
  fs::create_directories(out_path.parent_path());
  cv::Mat frame_bgr;
  // Worker frames are carried in RGB, but OpenCV image codecs expect BGR channel order.
  cv::cvtColor(frame, frame_bgr, cv::COLOR_RGB2BGR);
  return cv::imwrite(out_path.string(), frame_bgr);
}

} // namespace multi_camera_people_tracking
