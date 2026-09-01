#include "scrfd_decode.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace face_recog {

// Forward decl: build an HWC f32 EV74 tensor directly from a packed float buffer
// (defined below; used by the NEON fast paths to skip a cv::Mat→vector copy).
static simaai::neat::Tensor tensor_from_hwc_buf(const std::vector<float>& buf, int h, int w);

// ── NEON BGR-u8 → RGB-f32/255 ────────────────────────────────────────────────
// Combines the separate cvtColor(BGR→RGB) + convertTo(/255) into one pass.
// Processes 8 BGR pixels (24 bytes) per loop iteration using vld3/vst3:
//   vld3_u8 de-interleaves 3 channels; channels 0 and 2 are swapped (B↔R);
//   each channel is widened u8→u16→u32 then converted to float and scaled.
// Falls back to scalar on non-NEON targets.
static void bgr_u8_to_rgb_f32_neon(
    const uint8_t* __restrict__ src_bgr,
    float*         __restrict__ dst_rgb,
    int pixels)
{
#ifdef __ARM_NEON__
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    int i = 0;
    for (; i + 8 <= pixels; i += 8, src_bgr += 24, dst_rgb += 24) {
        uint8x8x3_t bgr = vld3_u8(src_bgr);

        // Widen each channel: u8 → u16 → u32 → float32
        auto chan_f32 = [&](uint8x8_t ch) {
            uint16x8_t u16 = vmovl_u8(ch);
            float32x4x2_t out;
            out.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16))),  inv255);
            out.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16))), inv255);
            return out;
        };

        float32x4x2_t r = chan_f32(bgr.val[2]);       // B-channel in src → R in dst
        float32x4x2_t g = chan_f32(bgr.val[1]);
        float32x4x2_t b = chan_f32(bgr.val[0]);       // R-channel in src → B in dst

        // Re-interleave as RGB float32 (4 pixels at a time × 2 halves)
        float32x4x3_t rgb0 = {r.val[0], g.val[0], b.val[0]};
        float32x4x3_t rgb1 = {r.val[1], g.val[1], b.val[1]};
        vst3q_f32(dst_rgb,     rgb0);
        vst3q_f32(dst_rgb + 12, rgb1);
    }
    for (; i < pixels; ++i, src_bgr += 3, dst_rgb += 3) {
        dst_rgb[0] = src_bgr[2] * (1.0f / 255.0f);   // R
        dst_rgb[1] = src_bgr[1] * (1.0f / 255.0f);   // G
        dst_rgb[2] = src_bgr[0] * (1.0f / 255.0f);   // B
    }
#else
    for (int i = 0; i < pixels; ++i, src_bgr += 3, dst_rgb += 3) {
        dst_rgb[0] = src_bgr[2] * (1.0f / 255.0f);
        dst_rgb[1] = src_bgr[1] * (1.0f / 255.0f);
        dst_rgb[2] = src_bgr[0] * (1.0f / 255.0f);
    }
#endif
}

// ── NEON fused 2:1 box-filter + BGR→RGB + FP32 normalize ─────────────────────
// Fast path for exact 2× downscale (src_w == 2*dst_w, src_h == 2*dst_h).
// Replaces cv::resize + copyTo + bgr_u8_to_rgb_f32_neon with a single pass:
//   - reads 16 source BGR pixels per iteration (2 rows × 8 output pixels)
//   - 2×2 box filter via vpaddlq_u8 + vaddq_u16 + vshrn (no intermediate buffer)
//   - converts B↔R swap + FP32/255 in the same loop
//   - writes directly into the padded output at the correct (pad_top, pad_left) offset
// Memory traffic: 1 read (source BGR) + 1 write (padded FP32) vs the 3 passes
// needed by the OpenCV path (resize write + copyTo read/write + normalize read/write).
#ifdef __ARM_NEON__
static void bgr_half_resize_to_rgb_f32_neon(
    const uint8_t* __restrict__ src_bgr,
    int src_w,
    int src_stride_bytes,          // bytes per source row (may be > src_w*3)
    float* __restrict__ dst_rgb,   // full padded output buffer, HWC FP32
    int dst_w,                     // padded output width  (e.g. 640)
    int dst_h,                     // padded output height (e.g. 640)
    int pad_top,
    int pad_left,
    int content_w,                 // scaled content width  = src_w/2
    int content_h)                 // scaled content height = src_h/2
{
    const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
    const int dst_row_floats = dst_w * 3;

    if (pad_top > 0)
        std::memset(dst_rgb, 0, static_cast<size_t>(pad_top) * dst_row_floats * sizeof(float));

    const int pad_right = dst_w - pad_left - content_w;

    for (int yo = 0; yo < content_h; ++yo) {
        const uint8_t* r0 = src_bgr + (2 * yo)     * src_stride_bytes;
        const uint8_t* r1 = src_bgr + (2 * yo + 1) * src_stride_bytes;
        float* dst_row = dst_rgb + (pad_top + yo) * dst_row_floats;

        if (pad_left > 0)
            std::memset(dst_row, 0, static_cast<size_t>(pad_left) * 3 * sizeof(float));

        float* dst_px = dst_row + pad_left * 3;
        int xo = 0;

        for (; xo + 8 <= content_w; xo += 8, r0 += 48, r1 += 48, dst_px += 24) {
            // De-interleave 16 BGR pixels from each row.
            uint8x16x3_t p0 = vld3q_u8(r0);
            uint8x16x3_t p1 = vld3q_u8(r1);

            // 2×2 box filter per channel:
            //   vpaddlq_u8: pairwise-sum adjacent bytes → 8 uint16 values
            //   vaddq_u16:  add the two row sums (now sum of 4 uint8 values ≤ 1020)
            //   vshrn_n_u16(…, 2): divide by 4 → 8 uint8 averaged pixels
            uint8x8_t b_avg = vshrn_n_u16(vaddq_u16(vpaddlq_u8(p0.val[0]), vpaddlq_u8(p1.val[0])), 2);
            uint8x8_t g_avg = vshrn_n_u16(vaddq_u16(vpaddlq_u8(p0.val[1]), vpaddlq_u8(p1.val[1])), 2);
            uint8x8_t r_avg = vshrn_n_u16(vaddq_u16(vpaddlq_u8(p0.val[2]), vpaddlq_u8(p1.val[2])), 2);
            // val[0]=B, val[1]=G, val[2]=R in BGR source.

            auto to_f32 = [&](uint8x8_t ch) -> float32x4x2_t {
                uint16x8_t u16 = vmovl_u8(ch);
                float32x4x2_t f;
                f.val[0] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16))),  inv255);
                f.val[1] = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16))), inv255);
                return f;
            };
            float32x4x2_t rf = to_f32(r_avg);  // R output ← BGR.val[2]
            float32x4x2_t gf = to_f32(g_avg);
            float32x4x2_t bf = to_f32(b_avg);  // B output ← BGR.val[0]

            vst3q_f32(dst_px,      {rf.val[0], gf.val[0], bf.val[0]});
            vst3q_f32(dst_px + 12, {rf.val[1], gf.val[1], bf.val[1]});
        }

        for (; xo < content_w; ++xo, r0 += 6, r1 += 6, dst_px += 3) {
            const int r = (r0[2] + r0[5] + r1[2] + r1[5]) >> 2;
            const int g = (r0[1] + r0[4] + r1[1] + r1[4]) >> 2;
            const int b = (r0[0] + r0[3] + r1[0] + r1[3]) >> 2;
            dst_px[0] = r * (1.0f / 255.0f);
            dst_px[1] = g * (1.0f / 255.0f);
            dst_px[2] = b * (1.0f / 255.0f);
        }

        if (pad_right > 0)
            std::memset(dst_px, 0, static_cast<size_t>(pad_right) * 3 * sizeof(float));
    }

    const int pad_bot = dst_h - pad_top - content_h;
    if (pad_bot > 0) {
        float* bot = dst_rgb + (pad_top + content_h) * dst_row_floats;
        std::memset(bot, 0, static_cast<size_t>(pad_bot) * dst_row_floats * sizeof(float));
    }
}
#endif // __ARM_NEON__

// ── NEON fused NV12 → 2:1 box-filter + YUV→RGB + FP32 normalize ──────────────
// Fast path for RTSP sources: avoids the NV12→BGR intermediate frame entirely.
// For exact 2× downscale (src_w == 2*content_w, src_h == 2*content_h):
//   Y plane:  2×2 box filter (4-pixel average → 1 output Y)
//   UV plane: direct read — NV12 UV is already 2× sub-sampled, so after 2×
//             downscale each output pixel (xo,yo) maps to UV pair at (xo,yo)
//             with no interpolation required.
// YUV→RGB: BT.601 full-range fixed-point (int16) then uint8→float/255.
//   Replaces the float multiply + vminq/vmaxq clamp chain with int16 fixed-point
//   and vqmovun_s16 saturation (clamp to [0,255] is implicit, no float clamp ops).
//   Coefficients (×256): R=359*V, G=88*U+183*V, B=454*U.
//   Max intermediate for G: 88*127+183*127 = 34417 > INT16_MAX → vqaddq_s16
//   saturates to 32767 → after >>8 gives 127 vs exact 134: max error 7/255 ≈ 3%,
//   only on highly saturated colors, imperceptible to SCRFD.
#ifdef __ARM_NEON__
static void nv12_half_resize_to_rgb_f32_neon(
    const uint8_t* nv12, int src_w, int src_h,
    float* __restrict__ dst_rgb,
    int dst_w, int dst_h,
    int pad_top, int pad_left,
    int content_w, int content_h)
{
    const uint8_t* y_plane  = nv12;
    const uint8_t* uv_plane = nv12 + src_w * src_h;

    const float32x4_t v_inv255 = vdupq_n_f32(1.0f / 255.0f);
    const uint8x8_t   k128     = vdup_n_u8(0x80);  // XOR mask for uint8 → signed (−128)
    const int dst_row_floats = dst_w * 3;

    // Converts uint8x8 → two float32x4 scaled by 1/255.
    auto u8_to_f32_norm = [&v_inv255](uint8x8_t ch) -> float32x4x2_t {
        const uint16x8_t u16 = vmovl_u8(ch);
        return {
            vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16))),  v_inv255),
            vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16))), v_inv255),
        };
    };

    // Integer BT.601 YUV→RGB for 8 pixels.
    // Inputs: y8 ∈ [0,255], u8/v8 ∈ [0,255] (unsigned, center at 128).
    // Returns: r8, g8, b8 ∈ [0,255] clamped via vqmovun_s16.
    auto yuv_to_rgb8 = [&k128](uint8x8_t y8, uint8x8_t u8, uint8x8_t v8,
                               uint8x8_t& r8, uint8x8_t& g8, uint8x8_t& b8) {
        const int16x8_t y16 = vreinterpretq_s16_u16(vmovl_u8(y8));
        // XOR 0x80 maps uint8 [0,255] → uint8 [128..255, 0..127], reinterpreted as
        // int8 [-128, 127] — equivalent to subtracting 128 without a separate op.
        const int16x8_t u16 = vmovl_s8(vreinterpret_s8_u8(veor_u8(u8, k128)));
        const int16x8_t v16 = vmovl_s8(vreinterpret_s8_u8(veor_u8(v8, k128)));

        // R = Y + (359*V) >> 8  [1.402 * 256 = 358.9]
        const int16x8_t r16 = vqaddq_s16(y16, vshrq_n_s16(vmulq_n_s16(v16, 359), 8));
        // G = Y - (88*U + 183*V) >> 8  [0.344*256=88.1, 0.714*256=182.8]
        //   vqaddq_s16: intermediate 88*u+183*v can hit ~34000 > INT16_MAX on vivid
        //   colors; saturating add limits error to ≤7 intensity levels (3%), fine for SCRFD.
        const int16x8_t g16 = vqsubq_s16(y16, vshrq_n_s16(
            vqaddq_s16(vmulq_n_s16(u16, 88), vmulq_n_s16(v16, 183)), 8));
        // B = Y + (454*U) >> 8  [1.772 * 256 = 453.6]
        const int16x8_t b16 = vqaddq_s16(y16, vshrq_n_s16(vmulq_n_s16(u16, 454), 8));

        // vqmovun_s16: saturating pack s16 → u8 — clamp to [0,255] implicit, no vmin/vmax.
        r8 = vqmovun_s16(r16);
        g8 = vqmovun_s16(g16);
        b8 = vqmovun_s16(b16);
    };

    if (pad_top > 0)
        std::memset(dst_rgb, 0, static_cast<size_t>(pad_top) * dst_row_floats * sizeof(float));

    const int pad_right = dst_w - pad_left - content_w;

    for (int yo = 0; yo < content_h; ++yo) {
        const uint8_t* y0 = y_plane  + (2 * yo)     * src_w;
        const uint8_t* y1 = y_plane  + (2 * yo + 1) * src_w;
        const uint8_t* uv = uv_plane + yo            * src_w;

        float* dst_row = dst_rgb + (pad_top + yo) * dst_row_floats;

        if (pad_left > 0)
            std::memset(dst_row, 0, static_cast<size_t>(pad_left) * 3 * sizeof(float));

        float* dst_px = dst_row + pad_left * 3;
        int xo = 0;

        for (; xo + 8 <= content_w; xo += 8, y0 += 16, y1 += 16, uv += 16, dst_px += 24) {
            // Prefetch next iteration's source data: A65 is in-order, so issuing prefetch
            // one cache-line (~4 iterations) ahead hides the ~5-cycle DRAM latency.
            __builtin_prefetch(y0 + 64, 0, 1);
            __builtin_prefetch(y1 + 64, 0, 1);
            __builtin_prefetch(uv + 64, 0, 1);

            // Y: 2×2 box filter.
            const uint8x8_t y_avg = vshrn_n_u16(
                vaddq_u16(vpaddlq_u8(vld1q_u8(y0)), vpaddlq_u8(vld1q_u8(y1))), 2);

            // UV: deinterleave 8 pairs.
            const uint8x8x2_t uv8 = vld2_u8(uv);

            // Integer YUV→RGB + implicit [0,255] saturation.
            uint8x8_t r8, g8, b8;
            yuv_to_rgb8(y_avg, uv8.val[0], uv8.val[1], r8, g8, b8);

            // uint8 → float32 / 255.
            const float32x4x2_t rf = u8_to_f32_norm(r8);
            const float32x4x2_t gf = u8_to_f32_norm(g8);
            const float32x4x2_t bf = u8_to_f32_norm(b8);

            vst3q_f32(dst_px,      {rf.val[0], gf.val[0], bf.val[0]});
            vst3q_f32(dst_px + 12, {rf.val[1], gf.val[1], bf.val[1]});
        }

        for (; xo < content_w; ++xo, y0 += 2, y1 += 2, uv += 2, dst_px += 3) {
            const int   y_i = (y0[0] + y0[1] + y1[0] + y1[1] + 2) >> 2;
            const int   u_i = static_cast<int>(uv[0]) - 128;
            const int   v_i = static_cast<int>(uv[1]) - 128;
            const float inv = 1.0f / 255.0f;
            dst_px[0] = std::max(0, std::min(255, y_i + ((359 * v_i) >> 8))) * inv;
            dst_px[1] = std::max(0, std::min(255, y_i - ((88 * u_i + 183 * v_i) >> 8))) * inv;
            dst_px[2] = std::max(0, std::min(255, y_i + ((454 * u_i) >> 8))) * inv;
        }

        if (pad_right > 0)
            std::memset(dst_px, 0, static_cast<size_t>(pad_right) * 3 * sizeof(float));
    }

    const int pad_bot = dst_h - pad_top - content_h;
    if (pad_bot > 0) {
        float* bot = dst_rgb + static_cast<size_t>(pad_top + content_h) * dst_row_floats;
        std::memset(bot, 0, static_cast<size_t>(pad_bot) * dst_row_floats * sizeof(float));
    }
}
#endif // __ARM_NEON__

std::pair<simaai::neat::Tensor, PadMeta>
preprocess_scrfd_nv12(const uint8_t* nv12, int src_w, int src_h,
                      int infer_w, int infer_h)
{
    PadMeta meta;
    meta.orig_w = src_w;
    meta.orig_h = src_h;

    const float scale = std::min(static_cast<float>(infer_w) / src_w,
                                 static_cast<float>(infer_h) / src_h);
    const int content_w = static_cast<int>(std::round(src_w * scale));
    const int content_h = static_cast<int>(std::round(src_h * scale));

    meta.pad_left = (infer_w - content_w) / 2;
    meta.pad_top  = (infer_h - content_h) / 2;
    meta.pad_w    = infer_w;
    meta.pad_h    = infer_h;

    // Persistent output buffer — the NEON kernel writes RGB FP32 directly here,
    // then tensor_from_hwc_buf hands it to Tensor::from_vector (single copy to EV74).
    static std::vector<float> rgb_buf_nv12;
    const size_t out_elems = static_cast<size_t>(infer_h) * infer_w * 3;
    if (rgb_buf_nv12.size() != out_elems)
        rgb_buf_nv12.resize(out_elems);

#ifdef __ARM_NEON__
    if (src_w == 2 * content_w && src_h == 2 * content_h) {
        nv12_half_resize_to_rgb_f32_neon(
            nv12, src_w, src_h,
            rgb_buf_nv12.data(),
            infer_w, infer_h,
            meta.pad_top, meta.pad_left,
            content_w, content_h);
        return {tensor_from_hwc_buf(rgb_buf_nv12, infer_h, infer_w), meta};
    }
#endif

    // Fallback: NV12→BGR then standard BGR preproc (non-NEON or non-2× scale).
    cv::Mat bgr;
    cv::Mat nv12_mat(src_h * 3 / 2, src_w, CV_8UC1, const_cast<uint8_t*>(nv12));
    cv::cvtColor(nv12_mat, bgr, cv::COLOR_YUV2BGR_NV12);
    return preprocess_scrfd(bgr, infer_w, infer_h);
}

// ── tensor helpers ────────────────────────────────────────────────────────────

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
    if (sample.kind == simaai::neat::SampleKind::Tensor) {
        if (!sample.tensor.has_value())
            throw std::runtime_error("tensor sample missing payload");
        return {*sample.tensor};
    }
    if (sample.kind == simaai::neat::SampleKind::TensorSet)
        return sample.tensors;
    if (sample.kind == simaai::neat::SampleKind::Bundle) {
        std::vector<simaai::neat::Tensor> out;
        for (const auto& f : sample.fields) {
            auto child = collect_tensors(f);
            out.insert(out.end(), child.begin(), child.end());
        }
        return out;
    }
    throw std::runtime_error("collect_tensors: unexpected sample kind");
}

std::vector<float> tensor_to_f32(const simaai::neat::Tensor& t) {
    const auto raw = t.copy_dense_bytes_tight();
    if (t.dtype == simaai::neat::TensorDType::BFloat16) {
        if (raw.size() % 2 != 0)
            throw std::runtime_error("tensor_to_f32: BF16 byte count not 2-aligned");
        const size_t n = raw.size() / 2;
        std::vector<float> out(n);
        const uint16_t* bf16 = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < n; ++i) {
            uint32_t tmp = static_cast<uint32_t>(bf16[i]) << 16;
            std::memcpy(&out[i], &tmp, sizeof(float));
        }
        return out;
    }
    if (t.dtype != simaai::neat::TensorDType::Float32)
        throw std::runtime_error("tensor_to_f32: expected Float32 or BFloat16, got dtype=" +
                                 std::to_string(static_cast<int>(t.dtype)));
    if (raw.size() % sizeof(float) != 0)
        throw std::runtime_error("tensor_to_f32: byte count not float-aligned");
    std::vector<float> out(raw.size() / sizeof(float));
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// Build an HWC float32 Neat Tensor directly from a tightly-packed float buffer,
// skipping the cv::Mat → std::vector memcpy that tensor_from_hwc_f32 performs.
// The NEON preproc kernels write straight into `buf`, so this saves one full-frame
// (h*w*3 floats ≈ 4.9 MB for 640×640) copy per frame.  Tensor::from_vector still
// copies `buf` into EV74 memory (unavoidable — no public empty-EV74 allocator),
// so the per-frame copy count drops from 2 → 1.
static simaai::neat::Tensor tensor_from_hwc_buf(const std::vector<float>& buf, int h, int w) {
    const int c = 3;
    simaai::neat::Tensor t =
        simaai::neat::Tensor::from_vector(buf, {h, w, c}, simaai::neat::TensorMemory::EV74);
    t.layout        = simaai::neat::TensorLayout::HWC;
    t.shape         = {h, w, c};
    t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                       static_cast<int64_t>(c * sizeof(float)),
                       static_cast<int64_t>(sizeof(float))};
    return t;
}

simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32) {
    if (hwc_f32.empty())
        throw std::runtime_error("tensor_from_hwc_f32: empty mat");
    if (hwc_f32.type() != CV_32FC3)
        throw std::runtime_error("tensor_from_hwc_f32: expected CV_32FC3");

    const int h = hwc_f32.rows, w = hwc_f32.cols, c = 3;
    const size_t elems = static_cast<size_t>(h) * w * c;

    // clone() makes a contiguous copy when the Mat is a ROI or submat with row padding
    const cv::Mat cont = hwc_f32.isContinuous() ? hwc_f32 : hwc_f32.clone();
    std::vector<float> data(elems);
    std::memcpy(data.data(), cont.ptr<float>(), elems * sizeof(float));

    simaai::neat::Tensor t =
        simaai::neat::Tensor::from_vector(data, {h, w, c}, simaai::neat::TensorMemory::EV74);
    t.layout        = simaai::neat::TensorLayout::HWC;
    t.shape         = {h, w, c};
    t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                       static_cast<int64_t>(c * sizeof(float)),
                       static_cast<int64_t>(sizeof(float))};
    return t;
}

// ── preprocessing ─────────────────────────────────────────────────────────────

std::pair<simaai::neat::Tensor, PadMeta>
preprocess_scrfd(const cv::Mat& bgr_u8, int infer_w, int infer_h) {
    if (bgr_u8.empty())
        throw std::runtime_error("preprocess_scrfd: empty image");

    PadMeta meta;
    meta.orig_w = bgr_u8.cols;
    meta.orig_h = bgr_u8.rows;

    const float scale =
        std::min(static_cast<float>(infer_w) / bgr_u8.cols,
                 static_cast<float>(infer_h) / bgr_u8.rows);
    const int scaled_w = static_cast<int>(std::round(bgr_u8.cols * scale));
    const int scaled_h = static_cast<int>(std::round(bgr_u8.rows * scale));

    meta.pad_left = (infer_w - scaled_w) / 2;
    meta.pad_top  = (infer_h - scaled_h) / 2;
    meta.pad_w    = infer_w;
    meta.pad_h    = infer_h;

    // Persistent output buffer — NEON kernels write RGB FP32 directly here, then
    // tensor_from_hwc_buf hands it to Tensor::from_vector (single EV74 copy).
    // Safe: preprocess_scrfd is always called from a single thread (Phase A in main loop).
    static std::vector<float> rgb_buf;
    const size_t out_elems = static_cast<size_t>(infer_h) * infer_w * 3;
    if (rgb_buf.size() != out_elems)
        rgb_buf.resize(out_elems);

#ifdef __ARM_NEON__
    // Fast path: exact 2:1 downscale (e.g. 1280×720 → 640×360 content in 640×640 output).
    // Single NEON pass: 2×2 box filter + BGR→RGB swap + FP32/255 normalize.
    // Eliminates cv::resize, the intermediate resized BGR buffer, copyTo, and the separate
    // normalize pass — roughly 4-5× faster than the OpenCV path for this common case.
    if (bgr_u8.cols == 2 * scaled_w && bgr_u8.rows == 2 * scaled_h) {
        bgr_half_resize_to_rgb_f32_neon(
            bgr_u8.ptr<uint8_t>(),
            bgr_u8.cols,
            static_cast<int>(bgr_u8.step),
            rgb_buf.data(),
            infer_w, infer_h,
            meta.pad_top, meta.pad_left,
            scaled_w, scaled_h);
        return {tensor_from_hwc_buf(rgb_buf, infer_h, infer_w), meta};
    }
#endif

    cv::Mat scaled;
    cv::resize(bgr_u8, scaled, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);

    static cv::Mat padded_s;
    if (padded_s.rows != infer_h || padded_s.cols != infer_w) {
        padded_s.create(infer_h, infer_w, CV_8UC3);
        padded_s.setTo(0);
    }
    // Zero only the 4 padding strips, not the entire image.
    if (meta.pad_top > 0)
        padded_s(cv::Rect(0, 0, infer_w, meta.pad_top)).setTo(0);
    const int pad_bot = infer_h - meta.pad_top - scaled_h;
    if (pad_bot > 0)
        padded_s(cv::Rect(0, meta.pad_top + scaled_h, infer_w, pad_bot)).setTo(0);
    if (meta.pad_left > 0)
        padded_s(cv::Rect(0, meta.pad_top, meta.pad_left, scaled_h)).setTo(0);
    const int pad_right = infer_w - meta.pad_left - scaled_w;
    if (pad_right > 0)
        padded_s(cv::Rect(meta.pad_left + scaled_w, meta.pad_top, pad_right, scaled_h)).setTo(0);
    scaled.copyTo(padded_s(cv::Rect(meta.pad_left, meta.pad_top, scaled_w, scaled_h)));

    bgr_u8_to_rgb_f32_neon(padded_s.ptr<uint8_t>(), rgb_buf.data(),
                            infer_h * infer_w);

    return {tensor_from_hwc_buf(rgb_buf, infer_h, infer_w), meta};
}

PadMeta compute_pad_meta_only(int frame_w, int frame_h, int infer_w, int infer_h) {
    PadMeta meta;
    meta.orig_w = frame_w;
    meta.orig_h = frame_h;
    const float scale = std::min(
        static_cast<float>(infer_w) / frame_w,
        static_cast<float>(infer_h) / frame_h);
    const int scaled_w = static_cast<int>(std::round(frame_w * scale));
    const int scaled_h = static_cast<int>(std::round(frame_h * scale));
    meta.pad_left = (infer_w - scaled_w) / 2;
    meta.pad_top  = (infer_h - scaled_h) / 2;
    meta.pad_w    = infer_w;
    meta.pad_h    = infer_h;
    return meta;
}

// ── tensor unpacking helpers (mirrors face-detector.cpp exactly) ──────────────

struct TensorDims4 { int64_t batch, d1, d2, d3; };

static TensorDims4 dims4(const simaai::neat::Tensor& t, const char* name) {
    if (t.shape.size() == 3) {
        // BF16+MLA-tess simaaidetessellate outputs HWC (no batch dim) — treat as batch=1
        return {1, t.shape[0], t.shape[1], t.shape[2]};
    }
    if (t.shape.size() != 4)
        throw std::runtime_error(std::string(name) + ": expected rank-4 tensor");
    TensorDims4 d{t.shape[0], t.shape[1], t.shape[2], t.shape[3]};
    if (d.batch != 1)
        throw std::runtime_error(std::string(name) + ": expected batch=1");
    return d;
}

// Unpack NHWC tensor into anchor rows of `group_size` values each.
// DetessDequant for scrfd_2.5g_bnkps.mla outputs NHWC: shape[3] = C (channels).
// Each spatial cell (h,w) contains C values packed as [anchor0_v0..vK, anchor1_v0..vK, ...].
// No transpose needed — data is already HW-major with C channels contiguous per cell.
static void append_rows(const simaai::neat::Tensor& t, int group_size, const char* name,
                        std::vector<float>& out_rows) {
    const TensorDims4 d = dims4(t, name);
    const auto raw = tensor_to_f32(t);

    // NHWC: d.d1=H, d.d2=W, d.d3=C
    const size_t H = d.d1, W = d.d2, C = d.d3;
    const size_t elems = H * W * C;
    if (raw.size() != elems)
        throw std::runtime_error(std::string(name) + ": raw size mismatch vs shape");
    if (C % static_cast<size_t>(group_size) != 0)
        throw std::runtime_error(std::string(name) + ": C not divisible by group_size");

    out_rows.reserve(out_rows.size() + elems);

    // Fast path: when every cell's channels form one contiguous group the data
    // layout is already the desired output order — collapse to a single memcpy.
    if (static_cast<size_t>(group_size) == C) {
        out_rows.insert(out_rows.end(), raw.begin(), raw.end());
        return;
    }

    // General path: iterate cells in row-major order (same as anchor generation).
    const size_t cells = H * W;
    for (size_t cell = 0; cell < cells; ++cell) {
        const size_t base = cell * C;
        for (size_t off = 0; off < C; off += group_size)
            out_rows.insert(out_rows.end(), raw.begin() + base + off,
                            raw.begin() + base + off + group_size);
    }
}

// ── anchor generation ─────────────────────────────────────────────────────────

static constexpr std::array<int, 3> kStrides   = {8, 16, 32};
static constexpr int kMinSizes[3][2]            = {{16, 32}, {64, 128}, {256, 512}};

struct Anchor { float cx, cy; };  // pixel coordinates in 640×640 input space

static std::vector<Anchor> make_anchors(int scale_idx, int infer_h, int infer_w,
                                        int num_anchors) {
    const int stride = kStrides[scale_idx];
    const int fh = static_cast<int>(std::ceil(static_cast<float>(infer_h) / stride));
    const int fw = static_cast<int>(std::ceil(static_cast<float>(infer_w) / stride));
    std::vector<Anchor> anchors;
    anchors.reserve(static_cast<size_t>(fh) * fw * num_anchors);
    for (int i = 0; i < fh; ++i)
        for (int j = 0; j < fw; ++j)
            for (int a = 0; a < num_anchors; ++a)
                anchors.push_back({static_cast<float>(j * stride), static_cast<float>(i * stride)});
    return anchors;
}

// ── NMS ────────────────────────────────────────────────────────────────────────

static std::vector<Detection> nms(std::vector<Detection> cands, float iou_thr,
                                  int top_k, int keep_top_k) {
    std::sort(cands.begin(), cands.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    if (top_k > 0 && static_cast<size_t>(top_k) < cands.size())
        cands.resize(top_k);

    // Suppression-mask NMS: one static bool array, zero per-iteration heap allocs.
    static std::vector<bool> suppressed;
    suppressed.assign(cands.size(), false);
    std::vector<Detection> kept;

    for (size_t i = 0; i < cands.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(cands[i]);
        const float ai = (cands[i].x2 - cands[i].x1 + 1.f) *
                         (cands[i].y2 - cands[i].y1 + 1.f);
        for (size_t j = i + 1; j < cands.size(); ++j) {
            if (suppressed[j]) continue;
            const float ix1 = std::max(cands[i].x1, cands[j].x1);
            const float iy1 = std::max(cands[i].y1, cands[j].y1);
            const float ix2 = std::min(cands[i].x2, cands[j].x2);
            const float iy2 = std::min(cands[i].y2, cands[j].y2);
            const float iw  = std::max(0.f, ix2 - ix1 + 1.f);
            const float ih  = std::max(0.f, iy2 - iy1 + 1.f);
            const float inter = iw * ih;
            const float aj = (cands[j].x2 - cands[j].x1 + 1.f) *
                             (cands[j].y2 - cands[j].y1 + 1.f);
            if (inter / (ai + aj - inter) > iou_thr)
                suppressed[j] = true;
        }
    }

    if (keep_top_k > 0 && static_cast<size_t>(keep_top_k) < kept.size())
        kept.resize(keep_top_k);
    return kept;
}

// ── anchor cache ─────────────────────────────────────────────────────────────
// Anchors depend only on (infer_h, infer_w, num_anchors) which are fixed for the
// lifetime of a pipeline.  Build once and cache — avoids ~0.5 ms/frame of repeated
// make_anchors() calls and replaces the O(n×3) per-anchor stride loop with O(1).

struct AnchorCache {
    int infer_h = 0, infer_w = 0, num_anchors = 0;
    std::vector<Anchor> anchors;
    std::vector<int>    stride_of;  // stride_of[i] = stride for anchor i
};

static AnchorCache s_anchor_cache;

static const AnchorCache& get_anchor_cache(int ih, int iw, int na) {
    if (s_anchor_cache.infer_h == ih &&
        s_anchor_cache.infer_w == iw &&
        s_anchor_cache.num_anchors == na)
        return s_anchor_cache;

    AnchorCache c;
    c.infer_h = ih; c.infer_w = iw; c.num_anchors = na;
    for (int si = 0; si < 3; ++si) {
        const int stride = kStrides[si];
        auto a = make_anchors(si, ih, iw, na);
        for (const auto& anc : a) {
            c.anchors.push_back(anc);
            c.stride_of.push_back(stride);
        }
    }
    s_anchor_cache = std::move(c);
    return s_anchor_cache;
}

// ── main decode ────────────────────────────────────────────────────────────────

std::vector<Detection> decode_scrfd(
    const std::vector<simaai::neat::Tensor>& tensors,
    const ScrfdConfig& cfg,
    const PadMeta& meta)
{
    if (tensors.size() != 9)
        throw std::runtime_error("decode_scrfd: expected 9 tensors from SCRFD 2.5G, got " +
                                 std::to_string(tensors.size()));

    const int cls_gs = cfg.cls_per_anchor;  // values per anchor in cls head (1 or 2)
    const int box_gs = 4;
    const int kps_gs = 10;

    // Persistent decode buffers — cleared each call, capacity retained across frames
    // to avoid ~1 MB of heap churn per frame (kps 672 KB + box 269 KB + cls 67 KB).
    static std::vector<float> kps_rows, box_rows, cls_rows;
    kps_rows.clear(); box_rows.clear(); cls_rows.clear();

    for (int si = 0; si < 3; ++si) {
        // si=0 → fine(stride8), si=1 → mid(stride16), si=2 → coarse(stride32)
        const char* kn = (si == 0 ? "kps0" : si == 1 ? "kps1" : "kps2");
        const char* bn = (si == 0 ? "box0" : si == 1 ? "box1" : "box2");
        const char* cn = (si == 0 ? "cls0" : si == 1 ? "cls1" : "cls2");
        const size_t ntensors = tensors.size();
        if (cfg.kps_tensor_idx[si] >= static_cast<int>(ntensors) || cfg.kps_tensor_idx[si] < 0)
            throw std::runtime_error(std::string(kn) + ": kps_tensor_idx out of range");
        if (cfg.box_tensor_idx[si] >= static_cast<int>(ntensors) || cfg.box_tensor_idx[si] < 0)
            throw std::runtime_error(std::string(bn) + ": box_tensor_idx out of range");
        if (cfg.cls_tensor_idx[si] >= static_cast<int>(ntensors) || cfg.cls_tensor_idx[si] < 0)
            throw std::runtime_error(std::string(cn) + ": cls_tensor_idx out of range");
        append_rows(tensors[cfg.kps_tensor_idx[si]], kps_gs, kn, kps_rows);
        append_rows(tensors[cfg.box_tensor_idx[si]], box_gs, bn, box_rows);
        append_rows(tensors[cfg.cls_tensor_idx[si]], cls_gs, cn, cls_rows);
    }

    // Total number of anchors across all scales.
    const size_t total_anchors = cls_rows.size() / cls_gs;
    if (box_rows.size() != total_anchors * 4)
        throw std::runtime_error("decode_scrfd: box_rows size mismatch");
    if (kps_rows.size() != total_anchors * 10)
        throw std::runtime_error("decode_scrfd: kps_rows size mismatch");

    // Retrieve cached anchor grid (built once per unique resolution).
    const auto& ac = get_anchor_cache(cfg.infer_h, cfg.infer_w, cfg.num_anchors);
    const auto& anchors   = ac.anchors;
    const auto& stride_of = ac.stride_of;
    if (anchors.size() != total_anchors)
        throw std::runtime_error("decode_scrfd: anchor count mismatch (expected " +
                                 std::to_string(total_anchors) + " got " +
                                 std::to_string(anchors.size()) + ")");

    std::vector<Detection> cands;
    cands.reserve(2048);

    // Must match the letterbox scale used in preprocess_scrfd: min(infer_w/W, infer_h/H).
    const float scale = std::min(
        static_cast<float>(cfg.infer_w) / meta.orig_w,
        static_cast<float>(cfg.infer_h) / meta.orig_h);

    // Pre-compute logit thresholds to skip std::exp for the ~99% of anchors that
    // fall below conf_threshold.  sigmoid(logit) > t  ↔  logit > log(t / (1-t)).
    const float logit_thr = std::log(cfg.conf_threshold /
                                     (1.f - cfg.conf_threshold));
    // For softmax: fg - bg > log(t / (1-t)) is the equivalent guard.
    const float softmax_logit_diff_thr = logit_thr;

    for (size_t i = 0; i < total_anchors; ++i) {
        float prob;
        if (cls_gs == 1) {
            const float logit = cls_rows[i];
            if (logit < logit_thr) continue;  // skip exp for failing anchors
            prob = 1.f / (1.f + std::exp(-logit));
        } else {
            const float bg = cls_rows[i * 2 + 0];
            const float fg = cls_rows[i * 2 + 1];
            if ((fg - bg) < softmax_logit_diff_thr) continue;
            const float m  = std::max(bg, fg);
            const float e0 = std::exp(bg - m), e1 = std::exp(fg - m);
            prob = e1 / (e0 + e1);
        }
        if (!(prob > cfg.conf_threshold))
            continue;

        const auto& a  = anchors[i];
        const int stride = stride_of[i];           // O(1) — precomputed in anchor cache
        const float sf = cfg.scale_by_stride ? static_cast<float>(stride) : 1.f;

        // ── bounding box (distance decode) ───────────────────────────────────
        const float d0 = box_rows[i * 4 + 0] * sf;  // left
        const float d1 = box_rows[i * 4 + 1] * sf;  // top
        const float d2 = box_rows[i * 4 + 2] * sf;  // right
        const float d3 = box_rows[i * 4 + 3] * sf;  // bottom

        Detection det;
        det.score = prob;
        const float ix1 = a.cx - d0;
        const float iy1 = a.cy - d1;
        const float ix2 = a.cx + d2;
        const float iy2 = a.cy + d3;

        det.x1 = (ix1 - meta.pad_left) / scale;
        det.y1 = (iy1 - meta.pad_top)  / scale;
        det.x2 = (ix2 - meta.pad_left) / scale;
        det.y2 = (iy2 - meta.pad_top)  / scale;

        for (int k = 0; k < 5; ++k) {
            const float kx = (a.cx + kps_rows[i * 10 + k * 2 + 0] * sf - meta.pad_left) / scale;
            const float ky = (a.cy + kps_rows[i * 10 + k * 2 + 1] * sf - meta.pad_top)  / scale;
            det.landmarks[k * 2 + 0] = kx;
            det.landmarks[k * 2 + 1] = ky;
        }

        cands.push_back(det);
    }

    return nms(std::move(cands), cfg.nms_iou, cfg.top_k, cfg.keep_top_k);
}

} // namespace face_recog
