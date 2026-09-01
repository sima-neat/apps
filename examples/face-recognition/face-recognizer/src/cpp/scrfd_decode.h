#pragma once

#include "neat.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace face_recog {

// Five detected facial landmarks in original-image pixel coordinates.
// Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth.
using Landmarks = std::array<float, 10>;

struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    Landmarks landmarks{};
};

// Preprocessing bookkeeping so we can map 640×640 output back to original pixels.
struct PadMeta {
    int orig_w  = 0, orig_h  = 0;
    int pad_w   = 0, pad_h   = 0;  // padded dimensions (before resize to 640×640)
    int pad_top = 0, pad_left = 0;
};

struct ScrfdConfig {
    float conf_threshold  = 0.50f;
    float nms_iou         = 0.40f;
    int   top_k           = 5000;
    int   keep_top_k      = 100;
    int   infer_w         = 640;
    int   infer_h         = 640;
    // SCRFD 2.5G: 2 anchors/location, 1 class logit per anchor (sigmoid).
    // Set to 2 if your model was exported with 2-class softmax (RetinaFace-style).
    int   cls_per_anchor  = 1;
    int   num_anchors     = 2;
    // If true, multiply bbox/kps distance predictions by stride before decode.
    // Set false if the model already pre-scales by stride.
    bool  scale_by_stride = true;
    // Tensor index mapping: which tensor index holds each output type at each scale.
    // Matches scrfd_2.5g_bnkps.mla_mpk DetessDequant output (NHWC, fine→coarse):
    //   [0,1,2]=cls  [3,4,5]=box  [6,7,8]=kps  (stride 8,16,32 within each group)
    // Indices are for: fine(stride8)=idx[0], mid(stride16)=idx[1], coarse(stride32)=idx[2].
    int   cls_tensor_idx[3] = {0, 1, 2};
    int   box_tensor_idx[3] = {3, 4, 5};
    int   kps_tensor_idx[3] = {6, 7, 8};
};

// Pad-and-letterbox bgr_u8 to infer_w×infer_h (letterbox + normalize /255, RGB FP32).
std::pair<simaai::neat::Tensor, PadMeta>
preprocess_scrfd(const cv::Mat& bgr_u8, int infer_w = 640, int infer_h = 640);

// Fused NV12 → letterbox-resize → RGB FP32 preprocessing for RTSP sources.
// Skips the NV12→BGR intermediate frame entirely; on the common 2:1 downscale
// (e.g. 1280×720 → 640×360 content in 640×640 output) a NEON 2×2 box filter
// is used — same quality as bilinear for an exact power-of-2 scale.
// Falls back to cvtColor + preprocess_scrfd on non-NEON or non-2× sources.
std::pair<simaai::neat::Tensor, PadMeta>
preprocess_scrfd_nv12(const uint8_t* nv12, int src_w, int src_h,
                      int infer_w = 640, int infer_h = 640);

// Compute letterbox padding metadata from source frame dimensions only — no
// actual resize or normalize.  Used when EV74 CVU Preproc handles preprocessing
// so the CPU only needs the pad offsets for coordinate un-mapping.
PadMeta compute_pad_meta_only(int frame_w, int frame_h, int infer_w, int infer_h);

// Tensor layout: [cls×3, box×3, kps×3] at strides 8,16,32 — matches
//   scrfd_2.5g_bnkps.mla_mpk DetessDequant NHWC output order.
std::vector<Detection> decode_scrfd(
    const std::vector<simaai::neat::Tensor>& tensors,
    const ScrfdConfig& cfg,
    const PadMeta& meta);

// Collect all leaf tensors from a Neat Sample (handles Tensor/TensorSet/Bundle).
std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample);

std::vector<float> tensor_to_f32(const simaai::neat::Tensor& t);
simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32);

} // namespace face_recog
