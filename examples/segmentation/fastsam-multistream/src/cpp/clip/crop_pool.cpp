#include "clip/crop_pool.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace app::clip {

void crop_into(cv::Mat& dst, const cv::Mat& window_rgb, const cv::Mat& submask, int px, float bg) {
  const int h = window_rgb.rows;
  const int w = window_rgb.cols;
  const double scale = static_cast<double>(px) / std::min(h, w);
  const int nw = std::max(px, static_cast<int>(std::lround(w * scale)));
  const int nh = std::max(px, static_cast<int>(std::lround(h * scale)));
  const int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;

  cv::Mat resized;
  cv::resize(window_rgb, resized, cv::Size(nw, nh), 0, 0, interp);
  cv::Mat keep_full;
  cv::resize(submask, keep_full, cv::Size(nw, nh), 0, 0, cv::INTER_NEAREST);

  const int y0 = (nh - px) / 2;
  const int x0 = (nw - px) / 2;
  const cv::Rect roi(x0, y0, px, px);

  resized(roi).convertTo(dst, CV_32FC3, 1.0 / 255.0);
  dst.setTo(cv::Scalar(bg, bg, bg), keep_full(roi) == 0);
}

constexpr int kDefaultCropWorkers = 8;  // used when config sets crop_workers <= 0

CropPool::CropPool(int workers) : workers_(workers > 0 ? workers : kDefaultCropWorkers) {
  threads_.reserve(workers_);
  for (int i = 0; i < workers_; ++i) {
    threads_.emplace_back([this] { worker_loop(); });
  }
}

CropPool::~CropPool() { close(); }

void CropPool::worker_loop() {
  std::uint64_t seen = 0;  // last job generation this worker has run
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      work_cv_.wait(lock, [this, seen] { return stop_ || generation_ != seen; });
      if (stop_) {
        return;
      }
      seen = generation_;
    }
    // Steal indices until the batch is drained; body_/count_ stay fixed for this generation.
    for (int i = next_.fetch_add(1); i < count_; i = next_.fetch_add(1)) {
      (*body_)(i);
    }
    if (remaining_.fetch_sub(1) == 1) {
      std::lock_guard<std::mutex> lock(mu_);
      done_cv_.notify_one();
    }
  }
}

void CropPool::parallel_for(int count, const std::function<void(int)>& body) {
  if (count <= 0) {
    return;
  }
  if (count == 1) {
    body(0);  // skip the pool handshake for the common single-crop batch
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    body_ = &body;
    count_ = count;
    next_.store(0);
    remaining_.store(workers_);
    ++generation_;
  }
  work_cv_.notify_all();

  std::unique_lock<std::mutex> lock(mu_);
  done_cv_.wait(lock, [this] { return remaining_.load() == 0; });
  body_ = nullptr;
}

void CropPool::close() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (stop_) {
      return;
    }
    stop_ = true;
  }
  work_cv_.notify_all();
  for (auto& t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  threads_.clear();
}

}  // namespace app::clip
