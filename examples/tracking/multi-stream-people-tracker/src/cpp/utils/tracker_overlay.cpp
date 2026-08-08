#include "examples/tracking/multi-stream-people-tracker/src/cpp/utils/tracker_overlay_api.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace multi_stream_people_tracker {
namespace {

std::array<std::uint8_t, 3> hsv_to_bgr(float hue) {
  constexpr float saturation = 0.82f;
  constexpr float value = 1.0f;
  const float chroma = value * saturation;
  const float sector = hue / 60.0f;
  const float x = chroma * (1.0f - std::abs(std::fmod(sector, 2.0f) - 1.0f));
  float red = 0.0f;
  float green = 0.0f;
  float blue = 0.0f;
  if (sector < 1.0f) {
    red = chroma;
    green = x;
  } else if (sector < 2.0f) {
    red = x;
    green = chroma;
  } else if (sector < 3.0f) {
    green = chroma;
    blue = x;
  } else if (sector < 4.0f) {
    green = x;
    blue = chroma;
  } else if (sector < 5.0f) {
    red = x;
    blue = chroma;
  } else {
    red = chroma;
    blue = x;
  }
  const float offset = value - chroma;
  const auto channel = [offset](float component) {
    return static_cast<std::uint8_t>(std::lround((component + offset) * 255.0f));
  };
  return {channel(blue), channel(green), channel(red)};
}

void fill_bgr(std::uint8_t* destination, int pixels, TrackColor color) {
#if defined(__aarch64__)
  const uint8x16x3_t value = {vdupq_n_u8(color.blue), vdupq_n_u8(color.green),
                              vdupq_n_u8(color.red)};
  while (pixels >= 16) {
    vst3q_u8(destination, value);
    destination += 48;
    pixels -= 16;
  }
#endif
  while (pixels-- > 0) {
    destination[0] = color.blue;
    destination[1] = color.green;
    destination[2] = color.red;
    destination += 3;
  }
}

void fill_rect(cv::Mat& frame, int x1, int y1, int x2, int y2, TrackColor color) {
  x1 = std::clamp(x1, 0, frame.cols);
  x2 = std::clamp(x2, 0, frame.cols);
  y1 = std::clamp(y1, 0, frame.rows);
  y2 = std::clamp(y2, 0, frame.rows);
  if (x1 >= x2 || y1 >= y2) {
    return;
  }
  for (int y = y1; y < y2; ++y) {
    fill_bgr(frame.ptr<std::uint8_t>(y) + 3 * x1, x2 - x1, color);
  }
}

void draw_box(cv::Mat& frame, int x1, int y1, int x2, int y2, TrackColor color, bool predicted) {
  x1 = std::clamp(x1, 0, frame.cols - 1);
  x2 = std::clamp(x2, 0, frame.cols - 1);
  y1 = std::clamp(y1, 0, frame.rows - 1);
  y2 = std::clamp(y2, 0, frame.rows - 1);
  if (x1 >= x2 || y1 >= y2) {
    return;
  }
  if (!predicted) {
    fill_rect(frame, x1, y1, x2 + 1, std::min(y1 + 2, y2 + 1), color);
    fill_rect(frame, x1, std::max(y1, y2 - 1), x2 + 1, y2 + 1, color);
    fill_rect(frame, x1, y1, std::min(x1 + 2, x2 + 1), y2 + 1, color);
    fill_rect(frame, std::max(x1, x2 - 1), y1, x2 + 1, y2 + 1, color);
    return;
  }
  for (int x = x1; x <= x2; x += 6) {
    fill_rect(frame, x, y1, std::min(x + 3, x2 + 1), y1 + 1, color);
    fill_rect(frame, x, y2, std::min(x + 3, x2 + 1), y2 + 1, color);
  }
  for (int y = y1; y <= y2; y += 6) {
    fill_rect(frame, x1, y, x1 + 1, std::min(y + 3, y2 + 1), color);
    fill_rect(frame, x2, y, x2 + 1, std::min(y + 3, y2 + 1), color);
  }
}

constexpr std::array<std::uint16_t, 11> kGlyphs{
    0b010111010010111, // 0
    0b010110010010111, // 1
    0b110001010100111, // 2
    0b110001010001110, // 3
    0b101101111001001, // 4
    0b111100110001110, // 5
    0b011100111101111, // 6
    0b111001010010010, // 7
    0b111101111101111, // 8
    0b111101111001110, // 9
    0b101111101111101, // #
};

void draw_compact_id(cv::Mat& frame, int x, int y, int track_id, TrackColor background) {
  const std::string label = "#" + std::to_string(track_id);
  const int label_width = 2 + static_cast<int>(label.size()) * 4;
  x = std::clamp(x, 0, std::max(0, frame.cols - label_width));
  y = std::clamp(y, 0, std::max(0, frame.rows - 7));
  fill_rect(frame, x, y, x + label_width, y + 7, background);
  const TrackColor foreground{255, 255, 255};
  for (std::size_t character = 0; character < label.size(); ++character) {
    const int glyph_index = label[character] == '#' ? 10 : label[character] - '0';
    const std::uint16_t glyph = kGlyphs[static_cast<std::size_t>(glyph_index)];
    for (int row = 0; row < 5; ++row) {
      for (int column = 0; column < 3; ++column) {
        const int bit = 14 - (row * 3 + column);
        if ((glyph & (1U << bit)) != 0) {
          fill_rect(frame, x + 1 + static_cast<int>(character) * 4 + column, y + 1 + row,
                    x + 2 + static_cast<int>(character) * 4 + column, y + 2 + row, foreground);
        }
      }
    }
  }
}

} // namespace

TrackColor track_color(int track_id) {
  constexpr float kGoldenAngle = 137.50776405f;
  float hue = std::fmod(static_cast<float>(std::max(0, track_id)) * kGoldenAngle, 360.0f);
  if (hue < 0.0f) {
    hue += 360.0f;
  }
  const auto bgr = hsv_to_bgr(hue);
  return TrackColor{bgr[0], bgr[1], bgr[2]};
}

void draw_tracks_bgr(cv::Mat& frame, const std::vector<TrackedDetection>& tracks, float min_score) {
  if (frame.empty()) {
    return;
  }
  if (frame.type() != CV_8UC3) {
    throw std::invalid_argument("draw_tracks_bgr expects a CV_8UC3 frame");
  }
  for (const auto& track : tracks) {
    if (track.score < min_score) {
      continue;
    }
    const int x1 = static_cast<int>(std::lround(track.x1));
    const int y1 = static_cast<int>(std::lround(track.y1));
    const int x2 = static_cast<int>(std::lround(track.x2));
    const int y2 = static_cast<int>(std::lround(track.y2));
    const TrackColor color = track_color(track.track_id);
    draw_box(frame, x1, y1, x2, y2, color, track.predicted);
    draw_compact_id(frame, x1, y1 >= 8 ? y1 - 7 : y1, track.track_id, color);
  }
}

} // namespace multi_stream_people_tracker
