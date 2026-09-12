#include "align.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace face_recog {

// ArcFace 112×112 canonical landmark positions from InsightFace.
const float kArcFaceTemplate[5][2] = {
    {38.2946f, 51.6963f},  // left  eye
    {73.5318f, 51.5014f},  // right eye
    {56.0252f, 71.7366f},  // nose  tip
    {41.5493f, 92.3655f},  // left  mouth corner
    {70.7299f, 92.2041f},  // right mouth corner
};

// Closed-form least-squares similarity transform (4-DOF: scale, rotation, tx, ty).
// Model: xi' = a*xi - b*yi + c,  yi' = b*xi + a*yi + d  (a=s·cosθ, b=s·sinθ)
//
// Normal equations for the 4-parameter system with n point pairs reduce to:
//   [ss  0   sx  -sy] [a]   [sxdx + sydy]
//   [0   ss  sy   sx] [b] = [sxdy - sydx]
//   [sx  sy  n    0 ] [c]   [sdx        ]
//   [-sy sx  0    n ] [d]   [sdy        ]
// where ss=Σ(xi²+yi²), sx=Σxi, sy=Σyi, sdx=Σdxi, sdy=Σdyi,
//       sxdx=Σ(xi·dxi+yi·dyi), sxdy=Σ(xi·dyi-yi·dxi).
// This factors via Schur complement into two independent 2×2 systems — zero
// heap allocations, ~50 multiply-adds, no matrix library calls.
template <typename Pts>
static cv::Mat similarity_transform_lsq(const Pts& src, const Pts& dst) {
    const int n = static_cast<int>(src.size());
    double ss = 0, sx = 0, sy = 0, sdx = 0, sdy = 0, sxdx = 0, sxdy = 0;
    for (int i = 0; i < n; ++i) {
        const double x  = src[i].x, y  = src[i].y;
        const double dx = dst[i].x, dy = dst[i].y;
        ss   += x*x + y*y;
        sx   += x;   sy   += y;
        sdx  += dx;  sdy  += dy;
        sxdx += x*dx + y*dy;
        sxdy += x*dy - y*dx;
    }
    // Schur complement: eliminate (c,d) to get a 2×2 system for (a,b).
    // S = ss - (sx²+sy²)/n
    const double inv_n = 1.0 / n;
    const double S  = ss - (sx*sx + sy*sy) * inv_n;
    const double Ra = sxdx - (sx*sdx + sy*sdy) * inv_n;
    const double Rb = sxdy - (sx*sdy - sy*sdx) * inv_n;
    const double a  = Ra / S,  b = Rb / S;
    const double c  = (sdx - a*sx + b*sy) * inv_n;
    const double d  = (sdy - b*sx - a*sy) * inv_n;

    cv::Mat M(2, 3, CV_64F);
    M.at<double>(0,0) =  a;  M.at<double>(0,1) = -b;  M.at<double>(0,2) = c;
    M.at<double>(1,0) =  b;  M.at<double>(1,1) =  a;  M.at<double>(1,2) = d;
    return M;
}

cv::Mat align_face(const cv::Mat& bgr_frame, const Landmarks& lm) {
    if (bgr_frame.empty())
        throw std::runtime_error("align_face: empty frame");

    // std::array avoids two 5-element heap allocations per call.
    // dst is constant — declared static so it is built once.
    static const std::array<cv::Point2f, 5> dst = {{
        {kArcFaceTemplate[0][0], kArcFaceTemplate[0][1]},
        {kArcFaceTemplate[1][0], kArcFaceTemplate[1][1]},
        {kArcFaceTemplate[2][0], kArcFaceTemplate[2][1]},
        {kArcFaceTemplate[3][0], kArcFaceTemplate[3][1]},
        {kArcFaceTemplate[4][0], kArcFaceTemplate[4][1]},
    }};
    std::array<cv::Point2f, 5> src;
    for (int i = 0; i < 5; ++i)
        src[i] = {lm[i * 2], lm[i * 2 + 1]};

    cv::Mat transform = similarity_transform_lsq(src, dst);

    // Pre-allocated static crop Mat: warpAffine reuses the buffer when size/type match.
    static cv::Mat s_crop(kArcFaceH, kArcFaceW, CV_8UC3);
    cv::warpAffine(bgr_frame, s_crop, transform, cv::Size(kArcFaceW, kArcFaceH),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return s_crop;
}

cv::Mat align_face_nv12(const uint8_t* nv12, int W, int H, const Landmarks& lm) {
    // Compute a tight bounding box around the 5 landmarks + 30% margin,
    // then convert only that small region to BGR instead of the full frame.
    float min_x = lm[0], max_x = lm[0], min_y = lm[1], max_y = lm[1];
    for (int i = 1; i < 5; ++i) {
        min_x = std::min(min_x, lm[i * 2]);
        max_x = std::max(max_x, lm[i * 2]);
        min_y = std::min(min_y, lm[i * 2 + 1]);
        max_y = std::max(max_y, lm[i * 2 + 1]);
    }
    // 60% margin (was 35%) keeps more context around the face so the similarity
    // warp uses real source pixels rather than reflected border pixels for most
    // head poses, improving parity with the full-frame BGR alignment path.
    const float margin = std::max(max_x - min_x, max_y - min_y) * 0.60f;

    // Round ROI to even boundaries (NV12 chroma is 2×2 subsampled).
    int rx1 = std::max(0,   static_cast<int>(min_x - margin) & ~1);
    int ry1 = std::max(0,   static_cast<int>(min_y - margin) & ~1);
    int rx2 = std::min(W,  (static_cast<int>(max_x + margin) + 2) & ~1);
    int ry2 = std::min(H,  (static_cast<int>(max_y + margin) + 2) & ~1);
    const int rW = rx2 - rx1;
    const int rH = ry2 - ry1;
    if (rW <= 0 || rH <= 0) throw std::runtime_error("align_face_nv12: degenerate ROI");

    // Extract the NV12 sub-region into a contiguous buffer.
    std::vector<uint8_t> roi_buf(static_cast<size_t>(rW) * rH * 3 / 2);
    // Y plane: copy row-by-row from the frame
    for (int y = 0; y < rH; ++y)
        std::memcpy(roi_buf.data() + y * rW, nv12 + (ry1 + y) * W + rx1, rW);
    // UV plane: interleaved U,V at half resolution
    const uint8_t* src_uv = nv12 + W * H;
    uint8_t*       dst_uv = roi_buf.data() + rW * rH;
    for (int uv_y = 0; uv_y < rH / 2; ++uv_y)
        std::memcpy(dst_uv + uv_y * rW,
                    src_uv + (ry1 / 2 + uv_y) * W + rx1, rW);

    // Convert the small ROI from NV12 to BGR.
    cv::Mat roi_nv12_mat(rH * 3 / 2, rW, CV_8UC1, roi_buf.data());
    cv::Mat roi_bgr;
    cv::cvtColor(roi_nv12_mat, roi_bgr, cv::COLOR_YUV2BGR_NV12);

    // Adjust landmarks to ROI-relative coordinates for the similarity transform.
    std::array<cv::Point2f, 5> src_pts;
    for (int i = 0; i < 5; ++i)
        src_pts[i] = {lm[i * 2] - rx1, lm[i * 2 + 1] - ry1};

    static const std::array<cv::Point2f, 5> dst_pts = {{
        {kArcFaceTemplate[0][0], kArcFaceTemplate[0][1]},
        {kArcFaceTemplate[1][0], kArcFaceTemplate[1][1]},
        {kArcFaceTemplate[2][0], kArcFaceTemplate[2][1]},
        {kArcFaceTemplate[3][0], kArcFaceTemplate[3][1]},
        {kArcFaceTemplate[4][0], kArcFaceTemplate[4][1]},
    }};

    cv::Mat transform = similarity_transform_lsq(src_pts, dst_pts);
    static cv::Mat s_crop(kArcFaceH, kArcFaceW, CV_8UC3);
    cv::warpAffine(roi_bgr, s_crop, transform, cv::Size(kArcFaceW, kArcFaceH),
                   cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return s_crop;
}

cv::Mat preprocess_arcface_crop(const cv::Mat& crop_bgr) {
    if (crop_bgr.empty() || crop_bgr.cols != kArcFaceW || crop_bgr.rows != kArcFaceH)
        throw std::runtime_error("preprocess_arcface_crop: expected 112×112 BGR");

    // Fused BGR→RGB + uint8→float32 + scale to [-1,1] in a single pass,
    // avoiding the intermediate RGB Mat that the two-call path allocates.
    static cv::Mat s_f32(kArcFaceH, kArcFaceW, CV_32FC3);
    const int pixels = kArcFaceH * kArcFaceW;
    const uint8_t* src = crop_bgr.ptr<uint8_t>();
    float* dst_ptr = s_f32.ptr<float>();
    constexpr float inv127_5 = 1.f / 127.5f;
    for (int p = 0; p < pixels; ++p) {
        dst_ptr[p * 3 + 0] = src[p * 3 + 2] * inv127_5 - 1.f;  // R (was B)
        dst_ptr[p * 3 + 1] = src[p * 3 + 1] * inv127_5 - 1.f;  // G
        dst_ptr[p * 3 + 2] = src[p * 3 + 0] * inv127_5 - 1.f;  // B (was R)
    }
    return s_f32;
}

} // namespace face_recog
