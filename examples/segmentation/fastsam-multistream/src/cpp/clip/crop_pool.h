#pragma once

#include "fastsam.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace app::clip {

constexpr int kClipImagePx = 256;  // MobileCLIP-S2 image encoder input side

// Resize + centre-crop window_rgb to px*px, mask the background out to bg, write float32 into dst.
void crop_into(cv::Mat& dst, const cv::Mat& window_rgb, const cv::Mat& submask, int px = kClipImagePx,
               float bg = 1.0f);

// Fixed thread pool that runs one parallel_for at a time (the encoder serialises its calls).
class CropPool {
 public:
  explicit CropPool(int workers = 0);  // workers <= 0 -> kDefaultCropWorkers
  ~CropPool();

  CropPool(const CropPool&) = delete;
  CropPool& operator=(const CropPool&) = delete;

  int workers() const { return workers_; }

  // Run body(i) for every i in [0, count); returns once all have finished.
  void parallel_for(int count, const std::function<void(int)>& body);

  void close();

 private:
  void worker_loop();

  int workers_;
  std::vector<std::thread> threads_;

  std::mutex mu_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  bool stop_ = false;

  const std::function<void(int)>* body_ = nullptr;  // valid only while a job is in flight
  int count_ = 0;
  std::uint64_t generation_ = 0;
  std::atomic<int> next_{0};
  std::atomic<int> remaining_{0};
};

}  // namespace app::clip
