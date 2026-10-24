#include "overlay.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string>

namespace face_recog {

static cv::Scalar identity_color(int gallery_index) {
    static const std::array<cv::Scalar, 8> kColors = {
        cv::Scalar(0, 255, 0),   cv::Scalar(255, 128, 0), cv::Scalar(0, 128, 255),
        cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 0),
        cv::Scalar(128, 0, 255), cv::Scalar(255, 0, 128),
    };
    if (gallery_index < 0) return cv::Scalar(120, 120, 120);
    return kColors[gallery_index % kColors.size()];
}

static cv::Scalar landmark_color(int kp_index) {
    static const std::array<cv::Scalar, 5> kLMColors = {
        cv::Scalar(0, 0, 255),   // left  eye
        cv::Scalar(0, 255, 0),   // right eye
        cv::Scalar(255, 165, 0), // nose
        cv::Scalar(255, 0, 255), // left  mouth
        cv::Scalar(0, 255, 255), // right mouth
    };
    return kLMColors[kp_index % 5];
}

void draw_overlay(
    cv::Mat&                        bgr,
    const std::vector<Detection>&   detections,
    const std::vector<MatchResult>& matches,
    const OverlayConfig&            cfg)
{
    const double font_scale = cfg.font_scale_x10 / 10.0;

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        const auto& m = (i < matches.size()) ? matches[i]
                                              : MatchResult{"?", -2.f, -1};

        const cv::Scalar color = identity_color(m.index);

        const int x1 = std::max(0, static_cast<int>(std::lround(d.x1)));
        const int y1 = std::max(0, static_cast<int>(std::lround(d.y1)));
        const int x2 = std::min(bgr.cols - 1, static_cast<int>(std::lround(d.x2)));
        const int y2 = std::min(bgr.rows - 1, static_cast<int>(std::lround(d.y2)));

        if (x2 > x1 && y2 > y1)
            cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color,
                          cfg.bbox_thickness);

        std::string label = m.name;
        if (cfg.draw_score && m.score >= -1.f)
            label += " " + cv::format("%.2f", m.score);

        int baseline = 0;
        const cv::Size tsz =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
        const int ty = std::max(tsz.height + 4, y1);
        cv::rectangle(bgr, cv::Point(x1, ty - tsz.height - 4),
                      cv::Point(x1 + tsz.width, ty), color, cv::FILLED);
        cv::putText(bgr, label, cv::Point(x1, ty - 2), cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        if (cfg.draw_landmarks) {
            for (int k = 0; k < 5; ++k) {
                const int kx = static_cast<int>(std::lround(d.landmarks[k * 2]));
                const int ky = static_cast<int>(std::lround(d.landmarks[k * 2 + 1]));
                cv::circle(bgr, cv::Point(kx, ky), cfg.landmark_radius,
                           landmark_color(k), -1);
            }
        }
    }
}

// ── NV12 drawing helpers ──────────────────────────────────────────────────────

namespace {

// BT.601 studio-swing BGR→YUV (matches OpenCV COLOR_YUV2BGR_NV12 inverse).
struct YUV { uint8_t Y, U, V; };
static YUV bgr_to_yuv(uint8_t B, uint8_t G, uint8_t R) {
    return {
        static_cast<uint8_t>(std::clamp((( 66*R + 129*G +  25*B + 128) >> 8) + 16,  16, 235)),
        static_cast<uint8_t>(std::clamp(((-38*R -  74*G + 112*B + 128) >> 8) + 128, 16, 240)),
        static_cast<uint8_t>(std::clamp(((112*R -  94*G -  18*B + 128) >> 8) + 128, 16, 240)),
    };
}

// Copy an even-aligned ROI out of an NV12 frame into a contiguous NV12 buffer.
// `rx`/`ry` must be even and `rw`/`rh` even so the ROI's 2×2 chroma blocks line up
// with the frame's chroma grid.
static void nv12_roi_extract(const uint8_t* nv12, int W, int H,
                             int rx, int ry, int rw, int rh,
                             std::vector<uint8_t>& out)
{
    (void)H;
    out.resize(static_cast<size_t>(rw) * rh * 3 / 2);
    const uint8_t* Y   = nv12;
    const uint8_t* UV  = nv12 + static_cast<size_t>(W) * H;
    uint8_t*       oY  = out.data();
    uint8_t*       oUV = out.data() + static_cast<size_t>(rw) * rh;
    for (int y = 0; y < rh; ++y)
        std::memcpy(oY + static_cast<size_t>(y) * rw,
                    Y + static_cast<size_t>(ry + y) * W + rx, rw);
    // Chroma rows hold W bytes (W/2 interleaved UV pairs); for even `rx` the byte
    // offset of pixel column rx is exactly rx.
    for (int y = 0; y < rh / 2; ++y)
        std::memcpy(oUV + static_cast<size_t>(y) * rw,
                    UV + static_cast<size_t>(ry / 2 + y) * W + rx, rw);
}

// Write drawn overlay pixels back into the NV12 frame.
//
// Both Y and UV are written ONLY for pixels/blocks that actually changed after
// draw_overlay ran on the ROI.  Unchanged pixels are left completely untouched,
// preserving the original decoder luma and chroma without any round-trip error.
//
// Why this matters:
//   OpenCV COLOR_YUV2BGR_NV12 bilinear-upsamples UV, but converting BGR back with
//   box-averaged UV is not an exact inverse.  Writing back even "unchanged" pixels
//   through bgr_to_yuv() introduces ±1 Y and 2-4 UV-unit errors that show up as a
//   faint brightness rectangle at the ROI boundary — the encoder then treats this
//   artificial edge as real content and wastes bits on it every frame.
//
//   Y: skipped per-pixel when bgr == bgr_orig.
//   UV: skipped per 2×2 block when all 12 bytes match.
static void nv12_roi_write_bgr_selective(uint8_t* nv12, int W, int H,
                                         int rx, int ry,
                                         const cv::Mat& bgr,
                                         const cv::Mat& bgr_orig)
{
    (void)H;
    uint8_t*  Y  = nv12;
    uint8_t*  UV = nv12 + static_cast<size_t>(W) * H;
    const int rw = bgr.cols, rh = bgr.rows;

    // Y plane: write only pixels that changed.
    for (int y = 0; y < rh; ++y) {
        const uint8_t* src  = bgr.ptr<uint8_t>(y);
        const uint8_t* orig = bgr_orig.ptr<uint8_t>(y);
        uint8_t*       yp   = Y + static_cast<size_t>(ry + y) * W + rx;
        for (int x = 0; x < rw; ++x, src += 3, orig += 3) {
            if (src[0] != orig[0] || src[1] != orig[1] || src[2] != orig[2])
                yp[x] = bgr_to_yuv(src[0], src[1], src[2]).Y;
        }
    }

    // UV plane: only update 2×2 blocks where at least one pixel changed.
    for (int by = 0; by < rh / 2; ++by) {
        uint8_t*       uvp = UV + static_cast<size_t>(ry / 2 + by) * W + rx;
        const uint8_t* r0  = bgr.ptr<uint8_t>(by * 2);
        const uint8_t* r1  = bgr.ptr<uint8_t>(by * 2 + 1);
        const uint8_t* o0  = bgr_orig.ptr<uint8_t>(by * 2);
        const uint8_t* o1  = bgr_orig.ptr<uint8_t>(by * 2 + 1);
        for (int bx = 0; bx < rw / 2; ++bx) {
            const uint8_t* a  = r0  + bx * 2 * 3;
            const uint8_t* b  = r1  + bx * 2 * 3;
            const uint8_t* oa = o0  + bx * 2 * 3;
            const uint8_t* ob = o1  + bx * 2 * 3;
            if (a[0]==oa[0] && a[1]==oa[1] && a[2]==oa[2] &&
                a[3]==oa[3] && a[4]==oa[4] && a[5]==oa[5] &&
                b[0]==ob[0] && b[1]==ob[1] && b[2]==ob[2] &&
                b[3]==ob[3] && b[4]==ob[4] && b[5]==ob[5])
                continue;
            const int sb = a[0] + a[3] + b[0] + b[3];
            const int sg = a[1] + a[4] + b[1] + b[4];
            const int sr = a[2] + a[5] + b[2] + b[5];
            const auto c = bgr_to_yuv(static_cast<uint8_t>(sb / 4),
                                      static_cast<uint8_t>(sg / 4),
                                      static_cast<uint8_t>(sr / 4));
            uvp[bx * 2 + 0] = c.U;
            uvp[bx * 2 + 1] = c.V;
        }
    }
}

} // anonymous namespace

// Draws by converting only a small ROI around each face to BGR, running the SAME
// OpenCV drawing code as draw_overlay (so antialiased text and circular landmarks
// are identical), then converting that ROI back to NV12.
//
// This replaces the earlier hand-rolled NV12 primitives, which wrote chroma only at
// even-x/even-y and so left stale background chroma bleeding through labels drawn at
// odd offsets. It also avoids converting the whole 1280×720 frame: a face ROI is
// ~200×250 px, roughly 18x less work, and it keeps the encoder fed with NV12 so the
// GStreamer chain needs no software BGR→NV12 pass.
void draw_overlay_nv12(
    uint8_t*                        nv12, int W, int H,
    const std::vector<Detection>&   detections,
    const std::vector<MatchResult>& matches,
    const OverlayConfig&            cfg)
{
    const double font_scale = cfg.font_scale_x10 / 10.0;
    const int    t = cfg.bbox_thickness;
    const int    r = cfg.landmark_radius;

    std::vector<uint8_t> roi_buf;

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        const auto& m = (i < matches.size()) ? matches[i] : MatchResult{"?", -2.f, -1};

        // Mirror draw_overlay's label metrics so the ROI is guaranteed to contain it.
        std::string label = m.name;
        if (cfg.draw_score && m.score >= -1.f)
            label += " " + cv::format("%.2f", m.score);
        int baseline = 0;
        const cv::Size tsz =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);

        const int bx1 = static_cast<int>(std::lround(d.x1));
        const int by1 = static_cast<int>(std::lround(d.y1));
        const int bx2 = static_cast<int>(std::lround(d.x2));
        const int by2 = static_cast<int>(std::lround(d.y2));

        // Union of everything drawn: box outline, label above the box, landmarks.
        int ux1 = bx1 - t - 1;
        int uy1 = std::min(by1 - tsz.height - baseline - 8, by1) - t - 1;
        int ux2 = std::max(bx2, bx1 + tsz.width) + t + 2;
        int uy2 = by2 + t + 2;
        if (cfg.draw_landmarks) {
            for (int k = 0; k < 5; ++k) {
                const int kx = static_cast<int>(std::lround(d.landmarks[k * 2]));
                const int ky = static_cast<int>(std::lround(d.landmarks[k * 2 + 1]));
                ux1 = std::min(ux1, kx - r - 1); uy1 = std::min(uy1, ky - r - 1);
                ux2 = std::max(ux2, kx + r + 2); uy2 = std::max(uy2, ky + r + 2);
            }
        }

        // Clamp to the frame, then snap origin down and size down to even so the
        // ROI's chroma blocks align with the frame's.
        ux1 = std::max(0, ux1) & ~1;
        uy1 = std::max(0, uy1) & ~1;
        ux2 = std::min(W, ux2);
        uy2 = std::min(H, uy2);
        const int rw = (ux2 - ux1) & ~1;
        const int rh = (uy2 - uy1) & ~1;
        if (rw <= 1 || rh <= 1) continue;

        nv12_roi_extract(nv12, W, H, ux1, uy1, rw, rh, roi_buf);
        cv::Mat roi_nv12(rh * 3 / 2, rw, CV_8UC1, roi_buf.data());
        cv::Mat roi_bgr;
        cv::cvtColor(roi_nv12, roi_bgr, cv::COLOR_YUV2BGR_NV12);
        const cv::Mat roi_bgr_orig = roi_bgr.clone();

        // Shift this detection into ROI-local coordinates and reuse draw_overlay.
        Detection ds = d;
        ds.x1 -= ux1; ds.x2 -= ux1;
        ds.y1 -= uy1; ds.y2 -= uy1;
        for (int k = 0; k < 5; ++k) {
            ds.landmarks[k * 2]     -= ux1;
            ds.landmarks[k * 2 + 1] -= uy1;
        }
        draw_overlay(roi_bgr, {ds}, {m}, cfg);

        nv12_roi_write_bgr_selective(nv12, W, H, ux1, uy1, roi_bgr, roi_bgr_orig);
    }
}

} // namespace face_recog
