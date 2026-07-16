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

// A fixed pool of worker threads that runs one parallel-for at a time. The CLIP encoder
// serialises its calls (it holds its own mutex), so there is only ever a single job in
// flight and parallel_for blocks until every index has been processed.
class CropPool {
 public:
  // workers <= 0 -> kDefaultCropWorkers; kept modest so crops don't starve the GStreamer threads.
  explicit CropPool(int workers = 0);
  ~CropPool();

  CropPool(const CropPool&) = delete;
  CropPool& operator=(const CropPool&) = delete;

  int workers() const { return workers_; }

  // Run body(i) for every i in [0, count) across the pool; returns once all have finished.
  void parallel_for(int count, const std::function<void(int)>& body);

  void close();

 private:
  void worker_loop();

  int workers_;
  std::vector<std::thread> threads_;

  std::mutex mu_;
  std::condition_variable work_cv_;  // wakes workers when a job is posted
  std::condition_variable done_cv_;  // wakes parallel_for when the job finishes
  bool stop_ = false;

  // Current job, valid between parallel_for posting it and the last worker finishing it.
  const std::function<void(int)>* body_ = nullptr;
  int count_ = 0;
  std::uint64_t generation_ = 0;   // bumped once per job so workers detect new work
  std::atomic<int> next_{0};       // next index to claim (lock-free work stealing)
  std::atomic<int> remaining_{0};  // workers still running the current job
};

}  // namespace app::clip
