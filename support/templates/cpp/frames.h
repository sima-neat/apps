#pragma once

// Decoded frames: fields of a combined output, and conversion to BGR images for OpenCV.

#include <neat.h>

#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <string>

namespace utils {

// The tensor's image width and height, or its first two shape dimensions. False when neither
// is known.
bool infer_dims(const simaai::neat::Tensor& t, int& w, int& h);

// A decoded NV12 frame to BGR, its planes packed first.
bool nv12_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);
// A decoded 4:2:2 frame (NV16, as most USB webcams decode) to BGR on the CPU, every row keeping
// its own chroma. Reads the Y and UV planes at their own offsets and strides.
bool nv16_to_bgr(const simaai::neat::Tensor& t, cv::Mat& out, std::string& err);
// A decoded NV12, NV16 or BGR-convertible frame as a BGR image.
cv::Mat tensor_bgr_from_decoded(const simaai::neat::Tensor& tensor);

// The field with this stream label, searching bundle fields depth first; nullptr when none.
const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                       const std::string& label);
// The field of a combined output with this stream label, else the bundle field at the index.
const simaai::neat::Sample& joined_field(const simaai::neat::Sample& sample,
                                         const std::string& label, std::size_t bundle_index);

} // namespace utils
