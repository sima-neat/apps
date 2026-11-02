#include "overlay.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
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

} // namespace face_recog
