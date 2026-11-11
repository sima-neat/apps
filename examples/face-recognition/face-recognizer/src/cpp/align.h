#pragma once

#include "scrfd_decode.h"

#include <opencv2/core/mat.hpp>

namespace face_recog {

// ArcFace input size.
constexpr int kArcFaceW = 112;
constexpr int kArcFaceH = 112;

// Rotates, scales and shifts the face so its five landmarks land on the positions ArcFace was
// trained with, and returns the 112x112 BGR crop.
cv::Mat align_face(const cv::Mat& bgr, const Landmarks& landmarks);

// The aligned crop as the ArcFace input: RGB float32 in [-1, 1].
cv::Mat preprocess_arcface_crop(const cv::Mat& crop_bgr);

} // namespace face_recog
