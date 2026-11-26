// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "efficientsam3.h"

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace efficientsam3 {

namespace {

// Python's round(): halves go to the even neighbour.
int round_even(double value) {
  return static_cast<int>(std::nearbyint(value));
}

} // namespace

void preprocess(const cv::Mat& frame, cv::Mat& out) {
  const int height = frame.rows * 2 / 3;
  const int width = frame.cols;
  const bool shrink = width > kModelSize || height > kModelSize;
  const int interpolation = shrink ? cv::INTER_AREA : cv::INTER_LINEAR;
  cv::Mat y, uv, rgb;
  cv::resize(frame.rowRange(0, height), y, {kModelSize, kModelSize}, 0, 0, interpolation);
  const cv::Mat uv_plane(height / 2, width / 2, CV_8UC2, const_cast<uchar*>(frame.ptr(height)));
  cv::resize(uv_plane, uv, {kModelSize / 2, kModelSize / 2}, 0, 0, interpolation);
  cv::cvtColorTwoPlane(y, uv, rgb, cv::COLOR_YUV2RGB_NV12);
  rgb.convertTo(out, CV_32FC3, 1.0 / 127.5, -1.0);
}

std::vector<cv::Point> mask_outline(const cv::Mat& mask_logits, const Box& box, int width,
                                    int height, float logit_threshold) {
  const int mask_h = mask_logits.rows;
  const int mask_w = mask_logits.cols;
  const double cell_w = static_cast<double>(width) / mask_w;
  const double cell_h = static_cast<double>(height) / mask_h;
  const auto [x0, y0, x1, y1] = box;
  const int mx0 = std::max(0, static_cast<int>(std::floor(x0 / cell_w)) - 1);
  const int my0 = std::max(0, static_cast<int>(std::floor(y0 / cell_h)) - 1);
  const int mx1 = std::min(mask_w, static_cast<int>(std::ceil(x1 / cell_w)) + 1);
  const int my1 = std::min(mask_h, static_cast<int>(std::ceil(y1 / cell_h)) + 1);
  const int out_x0 = round_even(mx0 * cell_w);
  const int out_y0 = round_even(my0 * cell_h);
  const cv::Size out_size(round_even(mx1 * cell_w) - out_x0, round_even(my1 * cell_h) - out_y0);
  // Upscale the mask cells under the box before thresholding, as SAM 3 does, so the outline
  // follows the interpolated mask.
  cv::Mat upscaled;
  cv::resize(mask_logits(cv::Range(my0, my1), cv::Range(mx0, mx1)), upscaled, out_size, 0, 0,
             cv::INTER_LINEAR);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(upscaled > logit_threshold, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const auto& largest =
      *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  std::vector<cv::Point> polygon;
  cv::approxPolyDP(largest, polygon, 0.004 * cv::arcLength(largest, true), true);
  if (polygon.size() < 3) {
    return {};
  }
  for (cv::Point& point : polygon) {
    point += cv::Point(out_x0, out_y0);
  }
  return polygon;
}

double box_iou(const Box& a, const Box& b) {
  const double inter =
      static_cast<double>(std::max(0, std::min(a[2], b[2]) - std::max(a[0], b[0]))) *
      std::max(0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const double area_a = static_cast<double>(a[2] - a[0]) * (a[3] - a[1]);
  const double area_b = static_cast<double>(b[2] - b[0]) * (b[3] - b[1]);
  return inter / (area_a + area_b - inter);
}

std::vector<Segment> segments_of(const cv::Mat& detections, const cv::Mat& masks,
                                 const SegmentOptions& options) {
  const auto logit_threshold =
      static_cast<float>(std::log(options.mask_threshold / (1.0 - options.mask_threshold)));
  std::vector<int> best(detections.rows);
  std::iota(best.begin(), best.end(), 0);
  std::stable_sort(best.begin(), best.end(), [&](int a, int b) {
    return detections.at<float>(a, 4) > detections.at<float>(b, 4);
  });
  const double sx = static_cast<double>(options.width) / kModelSize;
  const double sy = static_cast<double>(options.height) / kModelSize;
  std::vector<Segment> segments;
  int n = 0;
  for (const int query : best) {
    // Duplicates and empty masks are skipped, so the limit counts the instances sent.
    if (segments.size() == static_cast<std::size_t>(options.max_detections)) {
      break;
    }
    const float* row = detections.ptr<float>(query);
    if (!(row[4] > static_cast<float>(options.min_score))) {
      continue;
    }
    ++n;
    const int x0 = std::clamp(static_cast<int>(std::floor(row[0] * sx)), 0, options.width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(row[1] * sy)), 0, options.height - 1);
    const Box box{
        x0, y0, std::max(x0 + 1, std::min(options.width, static_cast<int>(std::ceil(row[2] * sx)))),
        std::max(y0 + 1, std::min(options.height, static_cast<int>(std::ceil(row[3] * sy))))};
    if (std::any_of(segments.begin(), segments.end(),
                    [&](const Segment& kept) { return box_iou(box, kept.box) > kDuplicateIou; })) {
      continue;
    }
    cv::Mat mask_logits;
    cv::extractChannel(masks, mask_logits, query);
    std::vector<cv::Point> outline =
        mask_outline(mask_logits, box, options.width, options.height, logit_threshold);
    if (!outline.empty()) {
      segments.push_back({"seg_" + std::to_string(n), options.label,
                          std::nearbyint(row[4] * 1e4) / 1e4, box, std::move(outline)});
    }
  }
  return segments;
}

std::string segments_json(const std::vector<Segment>& segments) {
  nlohmann::ordered_json items = nlohmann::ordered_json::array();
  std::size_t size = sizeof(R"({"segments":[]})") - 1U;
  for (const Segment& segment : segments) {
    nlohmann::ordered_json mask = nlohmann::ordered_json::array();
    for (const cv::Point& point : segment.mask) {
      mask.push_back({point.x, point.y});
    }
    const auto& [x0, y0, x1, y1] = segment.box;
    nlohmann::ordered_json item = {{"id", segment.id},
                                   {"label", segment.label},
                                   {"confidence", segment.confidence},
                                   {"bbox", {x0, y0, x1 - x0, y1 - y0}},
                                   {"mask_format", "polygon"},
                                   {"mask", mask}};
    size += item.dump().size() + 1U;
    if (size > kMetadataByteBudget) {
      break;
    }
    items.push_back(std::move(item));
  }
  return nlohmann::ordered_json{{"segments", items}}.dump();
}

void OverlayClock::add_frame(int64_t pts_ms, const std::string& frame_id) {
  waiting_.emplace_back(pts_ms, frame_id);
}

std::vector<OverlayClock::Message> OverlayClock::add_result(int64_t pts_ms,
                                                            const std::string& data) {
  std::vector<Message> messages;
  while (!waiting_.empty() && waiting_.front().first <= pts_ms) {
    const auto [frame_pts, frame_id] = waiting_.front();
    waiting_.pop_front();
    const bool closer_to_previous = previous_ && frame_pts - previous_->first < pts_ms - frame_pts;
    messages.push_back({closer_to_previous ? previous_->second : data, frame_pts, frame_id});
  }
  previous_ = {pts_ms, data};
  return messages;
}

} // namespace efficientsam3
