#pragma once

#include "scrfd_decode.h"

#include <opencv2/core/mat.hpp>

namespace face_recog {

// Size of ArcFace-expected aligned face crop.
constexpr int kArcFaceW = 112;
constexpr int kArcFaceH = 112;

// Five canonical landmark positions in the 112×112 ArcFace coordinate frame.
// Order: left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner.
extern const float kArcFaceTemplate[5][2];

// Similarity warp of `bgr_frame` to the ArcFace 112×112 canonical template.
cv::Mat align_face(const cv::Mat& bgr_frame, const Landmarks& landmarks);

// BGR → RGB → float32 normalized to [-1, 1] (x/127.5 - 1.0).
cv::Mat preprocess_arcface_crop(const cv::Mat& crop_bgr);

} // namespace face_recog
