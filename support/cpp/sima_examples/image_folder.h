#pragma once

// Helpers for the examples that read a folder of image files and save annotated copies: list the
// images, read one, name the annotated PNG, and draw the detections on it. Mirrors
// sima_examples.image_folder in support/python. A Box is any type with x1, y1, x2, y2, score, and
// class_id, such as Neat's.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace sima_examples {

// The .jpg, .jpeg, .png, and .bmp files directly in folder, in file-name order. A suffix matches
// in any case, such as .PNG. Subfolders and their files are skipped.
inline std::vector<std::filesystem::path> image_paths(const std::filesystem::path& folder) {
  static constexpr std::array<const char*, 4> kImageSuffixes = {".jpg", ".jpeg", ".png", ".bmp"};
  if (!std::filesystem::is_directory(folder)) {
    throw std::runtime_error("image folder does not exist: " + folder.string());
  }
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    std::string suffix = entry.path().extension().string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (entry.is_regular_file() &&
        std::find(kImageSuffixes.begin(), kImageSuffixes.end(), suffix) != kImageSuffixes.end()) {
      paths.push_back(entry.path());
    }
  }
  if (paths.empty()) {
    throw std::runtime_error("no .jpg, .jpeg, .png, or .bmp files in " + folder.string());
  }
  std::sort(paths.begin(), paths.end(),
            [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
  return paths;
}

// The image at path as a BGR cv::Mat. Throws when OpenCV cannot read it.
inline cv::Mat read_image(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("cannot read image: " + path.string());
  }
  return image;
}

// The annotated PNG's file name: the image's stem, then suffix, then .png. An example picks the
// suffix that marks its results, such as "_detr".
inline std::string output_name(const std::filesystem::path& image_path,
                               const std::string& suffix) {
  return image_path.stem().string() + suffix + ".png";
}

// Box colors in BGR, one per class, repeating every eight classes.
inline const std::array<cv::Scalar, 8> kBoxColors = {
    cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
    cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
};

// A copy of image with each box, its label, and its score drawn on it.
template <typename Box>
cv::Mat draw_detections(const cv::Mat& image, const std::vector<Box>& boxes,
                        const std::vector<std::string>& labels) {
  cv::Mat annotated = image.clone();
  for (const Box& box : boxes) {
    const cv::Scalar& color = kBoxColors[static_cast<size_t>(box.class_id) % kBoxColors.size()];
    // The decoder clamps boxes to the image, so the coordinates are not negative.
    const cv::Point top_left(static_cast<int>(box.x1 + 0.5f), static_cast<int>(box.y1 + 0.5f));
    const cv::Point bottom_right(static_cast<int>(box.x2 + 0.5f), static_cast<int>(box.y2 + 0.5f));
    cv::rectangle(annotated, top_left, bottom_right, color, 2);
    const std::string text = labels.at(static_cast<size_t>(box.class_id)) + " " +
                             cv::format("%.2f", static_cast<double>(box.score));
    cv::putText(annotated, text, cv::Point(top_left.x, std::max(0, top_left.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv::LINE_AA);
  }
  return annotated;
}

} // namespace sima_examples
