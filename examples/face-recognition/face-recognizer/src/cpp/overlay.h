#pragma once

#include "scrfd_decode.h"
#include "match.h"

#include <opencv2/core/mat.hpp>
#include <vector>

namespace face_recog {

struct OverlayConfig {
    bool  draw_landmarks  = true;
    bool  draw_score      = true;
    int   font_scale_x10  = 6;  // cv::FONT_HERSHEY_SIMPLEX scale × 10
    int   bbox_thickness  = 2;
    int   landmark_radius = 3;
};

// `detections` and `matches` must have the same length.
void draw_overlay(
    cv::Mat&                         bgr,
    const std::vector<Detection>&    detections,
    const std::vector<MatchResult>&  matches,
    const OverlayConfig&             cfg = {});

// NV12 variant: draws boxes, keypoints, and text labels directly on the NV12 buffer,
// avoiding any full-frame color-space conversion. `nv12` is W×H×3/2 bytes, contiguous.
void draw_overlay_nv12(
    uint8_t*                         nv12, int W, int H,
    const std::vector<Detection>&    detections,
    const std::vector<MatchResult>&  matches,
    const OverlayConfig&             cfg = {});

} // namespace face_recog
