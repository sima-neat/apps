#include "scrfd_decode.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace face_recog {

namespace {

// scrfd_2.5g_bnkps: three strides, two anchors per cell, one sigmoid score per anchor.
// Output order: scores for strides 8/16/32, then boxes, then landmarks.
constexpr int kStrides[3]      = {8, 16, 32};
constexpr int kAnchorsPerCell  = 2;
constexpr int kBoxValues       = 4;
constexpr int kLandmarkValues  = 10;

float iou(const Detection& a, const Detection& b) {
    const float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1) + 1.f);
    const float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1) + 1.f);
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1 + 1.f) * (a.y2 - a.y1 + 1.f);
    const float area_b = (b.x2 - b.x1 + 1.f) * (b.y2 - b.y1 + 1.f);
    return inter / (area_a + area_b - inter);
}

} // namespace

Letterbox letterbox_for(int frame_w, int frame_h) {
    Letterbox lb;
    lb.scale = std::min(static_cast<float>(kScrfdInputW) / frame_w,
                        static_cast<float>(kScrfdInputH) / frame_h);
    lb.pad_left = (kScrfdInputW - static_cast<int>(std::round(frame_w * lb.scale))) / 2;
    lb.pad_top  = (kScrfdInputH - static_cast<int>(std::round(frame_h * lb.scale))) / 2;
    return lb;
}

std::pair<simaai::neat::Tensor, Letterbox> preprocess_scrfd(const cv::Mat& bgr) {
    if (bgr.empty()) throw std::runtime_error("preprocess_scrfd: empty image");
    const Letterbox lb = letterbox_for(bgr.cols, bgr.rows);
    const cv::Size scaled_size(static_cast<int>(std::round(bgr.cols * lb.scale)),
                               static_cast<int>(std::round(bgr.rows * lb.scale)));
    cv::Mat scaled;
    cv::resize(bgr, scaled, scaled_size, 0, 0, cv::INTER_LINEAR);
    // SCRFD was calibrated with black letterbox bars.
    cv::Mat canvas(kScrfdInputH, kScrfdInputW, CV_8UC3, cv::Scalar(0, 0, 0));
    scaled.copyTo(canvas(cv::Rect(lb.pad_left, lb.pad_top, scaled.cols, scaled.rows)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    cv::Mat rgb_f32;
    canvas.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);
    return {tensor_from_hwc_f32(rgb_f32), lb};
}

std::vector<Detection> decode_scrfd(const std::vector<simaai::neat::Tensor>& tensors,
                                    const ScrfdConfig& cfg, const Letterbox& lb) {
    if (tensors.size() != 9)
        throw std::runtime_error("decode_scrfd: expected 9 output tensors, got " +
                                 std::to_string(tensors.size()));

    // sigmoid(logit) > t  <=>  logit > log(t / (1 - t)); skips exp() for most anchors.
    const float logit_threshold = std::log(cfg.conf_threshold / (1.f - cfg.conf_threshold));

    std::vector<Detection> candidates;
    for (int s = 0; s < 3; ++s) {
        const int stride = kStrides[s];
        const int cells_w = kScrfdInputW / stride;
        const size_t anchors =
            static_cast<size_t>(kScrfdInputH / stride) * cells_w * kAnchorsPerCell;
        const auto scores    = tensor_to_f32(tensors[s]);
        const auto boxes     = tensor_to_f32(tensors[3 + s]);
        const auto landmarks = tensor_to_f32(tensors[6 + s]);
        if (scores.size() != anchors || boxes.size() != anchors * kBoxValues ||
            landmarks.size() != anchors * kLandmarkValues)
            throw std::runtime_error("decode_scrfd: unexpected tensor size at stride " +
                                     std::to_string(stride));

        for (size_t i = 0; i < anchors; ++i) {
            if (scores[i] < logit_threshold) continue;
            // Both anchors of a cell share its top-left corner as their centre.
            const size_t cell = i / kAnchorsPerCell;
            const float cx = static_cast<float>((cell % cells_w) * stride);
            const float cy = static_cast<float>((cell / cells_w) * stride);
            // Model-input pixels back to source-frame pixels.
            auto to_x = [&](float x) { return (x - lb.pad_left) / lb.scale; };
            auto to_y = [&](float y) { return (y - lb.pad_top) / lb.scale; };

            Detection det;
            det.score = 1.f / (1.f + std::exp(-scores[i]));
            // Boxes are distances from the anchor centre, in units of the stride.
            const float* box = &boxes[i * kBoxValues];
            det.x1 = to_x(cx - box[0] * stride);
            det.y1 = to_y(cy - box[1] * stride);
            det.x2 = to_x(cx + box[2] * stride);
            det.y2 = to_y(cy + box[3] * stride);
            const float* lm = &landmarks[i * kLandmarkValues];
            for (int k = 0; k < 5; ++k) {
                det.landmarks[k * 2]     = to_x(cx + lm[k * 2] * stride);
                det.landmarks[k * 2 + 1] = to_y(cy + lm[k * 2 + 1] * stride);
            }
            candidates.push_back(det);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    if (cfg.top_k > 0 && candidates.size() > static_cast<size_t>(cfg.top_k))
        candidates.resize(cfg.top_k);

    std::vector<Detection> kept;
    for (const auto& candidate : candidates) {
        const bool overlaps = std::any_of(kept.begin(), kept.end(), [&](const Detection& k) {
            return iou(k, candidate) > cfg.nms_iou;
        });
        if (overlaps) continue;
        kept.push_back(candidate);
        if (cfg.keep_top_k > 0 && kept.size() == static_cast<size_t>(cfg.keep_top_k)) break;
    }
    return kept;
}

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
    if (sample.kind == simaai::neat::SampleKind::Tensor) {
        if (!sample.tensor.has_value()) throw std::runtime_error("tensor sample missing payload");
        return {*sample.tensor};
    }
    if (sample.kind == simaai::neat::SampleKind::TensorSet) return sample.tensors;
    if (sample.kind == simaai::neat::SampleKind::Bundle) {
        std::vector<simaai::neat::Tensor> out;
        for (const auto& field : sample.fields) {
            auto child = collect_tensors(field);
            out.insert(out.end(), child.begin(), child.end());
        }
        return out;
    }
    throw std::runtime_error("collect_tensors: unexpected sample kind");
}

std::vector<float> tensor_to_f32(const simaai::neat::Tensor& t) {
    const auto raw = t.copy_dense_bytes_tight();
    if (t.dtype == simaai::neat::TensorDType::BFloat16) {
        // BF16 is the upper half of a float32.
        std::vector<float> out(raw.size() / 2);
        for (size_t i = 0; i < out.size(); ++i) {
            uint16_t half;
            std::memcpy(&half, raw.data() + i * 2, sizeof(half));
            const uint32_t bits = static_cast<uint32_t>(half) << 16;
            std::memcpy(&out[i], &bits, sizeof(float));
        }
        return out;
    }
    if (t.dtype != simaai::neat::TensorDType::Float32)
        throw std::runtime_error("tensor_to_f32: expected Float32 or BFloat16");
    std::vector<float> out(raw.size() / sizeof(float));
    std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
    return out;
}

simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32) {
    if (hwc_f32.empty() || hwc_f32.type() != CV_32FC3)
        throw std::runtime_error("tensor_from_hwc_f32: expected a non-empty CV_32FC3 image");
    const cv::Mat continuous = hwc_f32.isContinuous() ? hwc_f32 : hwc_f32.clone();
    const int h = continuous.rows, w = continuous.cols, c = 3;
    const std::vector<float> data(continuous.ptr<float>(),
                                  continuous.ptr<float>() + static_cast<size_t>(h) * w * c);
    simaai::neat::Tensor t =
        simaai::neat::Tensor::from_vector(data, {h, w, c}, simaai::neat::TensorMemory::EV74);
    t.layout        = simaai::neat::TensorLayout::HWC;
    t.shape         = {h, w, c};
    t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                       static_cast<int64_t>(c * sizeof(float)),
                       static_cast<int64_t>(sizeof(float))};
    return t;
}

} // namespace face_recog
