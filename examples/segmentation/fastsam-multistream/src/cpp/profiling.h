#pragma once

#include "support/runtime/example_utils.h"

#include <cstdio>

namespace app {

class ProfileWindow {
 public:
  ProfileWindow(bool enabled, int interval, int index)
      : enabled_(enabled), interval_(interval), index_(index) {}

  void note_drop(int dropped) { drops_ += dropped; }

  void add(double frame_ms) {
    if (!enabled_) {
      return;
    }
    if (window_start_ms_ == 0.0) {
      window_start_ms_ = sima_examples::time_ms();
    }
    ++frames_;
    frame_ms_ += frame_ms;
    if (frames_ >= interval_) {
      flush();
    }
  }

  void flush() {
    if (!enabled_ || frames_ == 0) {
      return;
    }
    const double n = frames_;
    const double now = sima_examples::time_ms();
    const double elapsed_s = (now - window_start_ms_) / 1000.0;
    const double fps = elapsed_s > 0.0 ? n / elapsed_s : 0.0;
    std::printf("[profile s%d] frames=%d fps=%.1f avg_frame_ms=%.1f drops=%d\n",
                index_, frames_, fps, frame_ms_ / n, drops_);
    frames_ = 0;
    drops_ = 0;
    frame_ms_ = 0.0;
    window_start_ms_ = now;
  }

 private:
  bool enabled_;
  int interval_;
  int index_;
  int frames_ = 0;
  int drops_ = 0;
  double frame_ms_ = 0.0;
  double window_start_ms_ = 0.0;
};

}  // namespace app
