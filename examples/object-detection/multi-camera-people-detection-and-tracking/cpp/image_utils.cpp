#include "examples/object-detection/multi-camera-people-detection-and-tracking/cpp/image_utils_api.cpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

namespace multi_camera_people_tracking {

std::filesystem::path sample_output_path(const std::filesystem::path& output_dir, int stream_index,
                                         int frame_index) {
  return output_dir / ("stream_" + std::to_string(stream_index)) /
         ("frame_" + cv::format("%06d", frame_index) + ".jpg");
}

QuantTessCpuPreprocState build_cpu_quanttess_preproc_state(
    const QuantTessCpuPreproc& contract, int src_width, int src_height) {
  QuantTessCpuPreprocState state;
  state.src_width = src_width;
  state.src_height = src_height;

  if (contract.aspect_ratio) {
    const double scale =
        std::min(static_cast<double>(contract.width) / src_width,
                 static_cast<double>(contract.height) / src_height);
    state.scaled_w = std::max(1, static_cast<int>(std::lround(src_width * scale)));
    state.scaled_h = std::max(1, static_cast<int>(std::lround(src_height * scale)));
  } else {
    state.scaled_w = contract.width;
    state.scaled_h = contract.height;
  }

  state.pad_x = 0;
  state.pad_y = 0;
  if (contract.padding_type == "CENTER") {
    state.pad_x = (contract.width - state.scaled_w) / 2;
    state.pad_y = (contract.height - state.scaled_h) / 2;
  }

  state.quant_input = cv::Mat::zeros(contract.height, contract.width, CV_32FC3);
  return state;
}

cv::Mat cpu_quanttess_input(const cv::Mat& frame_rgb, QuantTessCpuPreprocState& state) {
  if (frame_rgb.cols != state.src_width || frame_rgb.rows != state.src_height) {
    throw std::runtime_error("unexpected frame size for cached QuantTess preproc state");
  }

  cv::Mat resized;
  cv::resize(frame_rgb, resized, cv::Size(state.scaled_w, state.scaled_h), 0.0, 0.0,
             cv::INTER_LINEAR);
  state.quant_input.setTo(cv::Scalar(0.0f, 0.0f, 0.0f));

  cv::Mat roi =
      state.quant_input(cv::Rect(state.pad_x, state.pad_y, state.scaled_w, state.scaled_h));
  resized.convertTo(roi, CV_32FC3, 1.0 / 255.0);
  return state.quant_input;
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
    const cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
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
