#pragma once

// Read an image file, for the examples that read images from disk. Mirrors
// sima_examples.image_io in support/python.

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <stdexcept>

namespace sima_examples {

// The image at path as a BGR cv::Mat. Throws when OpenCV cannot read it.
inline cv::Mat read_image(const std::filesystem::path& path) {
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("cannot read image: " + path.string());
  }
  return image;
}

} // namespace sima_examples
