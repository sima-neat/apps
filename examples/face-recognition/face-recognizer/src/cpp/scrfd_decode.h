#pragma once

#include "neat.h"

#include <opencv2/core/mat.hpp>

#include <array>
#include <utility>
#include <vector>

namespace face_recog {

// Five facial landmarks in source-frame pixels:
// left eye, right eye, nose tip, left mouth corner, right mouth corner.
using Landmarks = std::array<float, 10>;

struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    Landmarks landmarks{};
};

// How a source frame was letterboxed into the SCRFD input, to map detections back.
struct Letterbox {
    float scale    = 1.f;  // source pixels * scale = model-input pixels
    int   pad_left = 0;
    int   pad_top  = 0;
};

struct ScrfdConfig {
    float conf_threshold = 0.65f;
    float nms_iou        = 0.40f;
    int   top_k          = 5000;  // candidates kept before NMS
    int   keep_top_k     = 100;   // detections kept after NMS
};

// SCRFD 2.5G input size.
constexpr int kScrfdInputW = 640;
constexpr int kScrfdInputH = 640;

Letterbox letterbox_for(int frame_w, int frame_h);

// Letterboxes a BGR image to the SCRFD input (black bars, RGB, [0,1] float32), the same
// preprocessing the EV74 runs for decoded video frames. Used for still images.
std::pair<simaai::neat::Tensor, Letterbox> preprocess_scrfd(const cv::Mat& bgr);

// Decodes the nine SCRFD 2.5G output tensors (scores, boxes, landmarks at strides 8, 16
// and 32) into detections in source-frame pixels, highest score first.
std::vector<Detection> decode_scrfd(const std::vector<simaai::neat::Tensor>& tensors,
                                    const ScrfdConfig& cfg, const Letterbox& letterbox);

// Leaf tensors of a sample (Tensor, TensorSet or Bundle).
std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample);

std::vector<float> tensor_to_f32(const simaai::neat::Tensor& t);

simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32);

} // namespace face_recog
