#pragma once

#include "patchcore_memory_bank.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace patchcore {

inline const cv::Size kModelInputSize(224, 224);
constexpr double kOverlayLutLow = 191.0;

inline cv::Rect letterbox_rect(const cv::Size& size) {
  const double scale = std::min(static_cast<double>(kModelInputSize.width) / size.width,
                                static_cast<double>(kModelInputSize.height) / size.height);
  const int w = static_cast<int>(std::lround(size.width * scale));
  const int h = static_cast<int>(std::lround(size.height * scale));
  return {(kModelInputSize.width - w) / 2, (kModelInputSize.height - h) / 2, w, h};
}

struct OverlayMaps {
  cv::Mat levels;
  cv::Mat mask;
};

inline OverlayMaps overlay_maps(const AnomalyResult& result, int map_h, int map_w, cv::Size size,
                                double sigma, double lower, double upper) {
  cv::Mat scores(map_h, map_w, CV_32F, const_cast<float*>(result.score_map.data()));
  cv::Mat canvas, heat;
  cv::resize(scores, canvas, kModelInputSize, 0, 0, cv::INTER_LINEAR);
  const auto region = letterbox_rect(size);
  heat = canvas(region);
  if (sigma > 0) {
    // Preserve the configured blur width in output pixels on the small model canvas.
    cv::GaussianBlur(heat, heat, cv::Size(), sigma * region.width / size.width,
                     sigma * region.height / size.height,
                     cv::BORDER_REFLECT_101 | cv::BORDER_ISOLATED);
  }
  cv::resize(heat, heat, size, 0, 0, cv::INTER_LINEAR);
  OverlayMaps maps;
  maps.mask = heat > upper;
  if (cv::countNonZero(maps.mask) == 0)
    return {};
  const double scale = (255 - kOverlayLutLow) / std::max(upper - lower, 1e-6);
  cv::convertScaleAbs(heat, maps.levels, scale, kOverlayLutLow - upper * scale);
  return maps;
}

inline cv::Mat draw_overlay(const cv::Mat& bgr, const AnomalyResult& result, int map_h, int map_w,
                            double sigma, double alpha, double patch_scale_min,
                            double patch_scale_max) {
  const auto maps =
      overlay_maps(result, map_h, map_w, bgr.size(), sigma, patch_scale_min, patch_scale_max);
  cv::Mat out = bgr.clone();
  if (!maps.mask.empty()) {
    cv::Mat color, blended;
    cv::applyColorMap(maps.levels, color, cv::COLORMAP_JET);
    cv::addWeighted(bgr, 1 - alpha, color, alpha, 0, blended);
    blended.copyTo(out, maps.mask);
  }
  return out;
}

inline const cv::Mat& nv12_colormap() {
  static const auto table = [] {
    cv::Mat levels(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i)
      levels.at<uint8_t>(0, i) = i;
    cv::Mat colors, swatches, i420;
    cv::applyColorMap(levels, colors, cv::COLORMAP_JET);
    // Each color fills a complete 2x2 chroma block.
    cv::resize(colors, swatches, cv::Size(512, 2), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(swatches, i420, cv::COLOR_BGR2YUV_I420);
    const auto* y_plane = i420.ptr<uint8_t>();
    const auto* u_plane = y_plane + swatches.total();
    const auto* v_plane = u_plane + levels.total();
    cv::Mat result(256, 1, CV_8UC3);
    for (int i = 0; i < 256; ++i)
      result.at<cv::Vec3b>(i) = {y_plane[2 * i], u_plane[i], v_plane[i]};
    return result;
  }();
  return table;
}

inline void draw_nv12_overlay(cv::Mat& nv12, const AnomalyResult& result, int map_h, int map_w,
                              double sigma, double alpha, double patch_scale_min,
                              double patch_scale_max) {
  const int height = nv12.rows * 2 / 3, width = nv12.cols;
  const auto maps =
      overlay_maps(result, map_h, map_w, {width, height}, sigma, patch_scale_min, patch_scale_max);
  if (maps.mask.empty())
    return;
  const auto& table = nv12_colormap();
  cv::Mat y_colors, y_table, blended;
  cv::extractChannel(table, y_table, 0);
  cv::LUT(maps.levels, y_table, y_colors);
  cv::Mat y = nv12.rowRange(0, height);
  cv::addWeighted(y, 1 - alpha, y_colors, alpha, 0, blended);
  blended.copyTo(y, maps.mask);
  // One color sample per 2x2 NV12 chroma block.
  cv::Mat uv(height / 2, width / 2, CV_8UC2, nv12.ptr(height), nv12.step);
  cv::Mat uv_indices, uv_mask, color;
  cv::resize(maps.levels, uv_indices, uv.size(), 0, 0, cv::INTER_NEAREST);
  cv::applyColorMap(uv_indices, color, table);
  cv::Mat uv_colors(uv.size(), CV_8UC2);
  const int channels[] = {1, 0, 2, 1};
  cv::mixChannels(&color, 1, &uv_colors, 1, channels, 2);
  cv::resize(maps.mask, uv_mask, uv.size(), 0, 0, cv::INTER_NEAREST);
  cv::addWeighted(uv, 1 - alpha, uv_colors, alpha, 0, blended);
  blended.copyTo(uv, uv_mask);
}

} // namespace patchcore
