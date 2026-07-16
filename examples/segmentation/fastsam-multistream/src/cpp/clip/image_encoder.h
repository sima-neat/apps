#pragma once

#include "clip/crop_pool.h"
#include "fastsam.h"
#include "neat.h"
#include "neat/models.h"

#include <opencv2/core.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace app::clip {

// The CLIP image tower is compiled for a fixed batch of 16 crops per run().
constexpr int kClipBatch = 16;

// One scored crop: its original candidate index and softmax score (high = better match).
struct Ranked {
  int index = -1;
  double score = 0.0;
};

// MobileCLIP image encoder, shared across streams
class ImageEncoder {
 public:
  ImageEncoder(const std::string& model_path, const simaai::neat::RunOptions& run_opt,
               int crop_workers);

  int crop_workers() const { return crop_pool_.workers(); }

  // Encode crops to [512] feature vectors, one per crop.
  std::vector<std::vector<float>> encode(const std::vector<Fastsam::Crop>& crops, int timeout_ms);

  // Encode the candidates, score them against the prompt features, and return the best index (empty
  // if none clears min_score) plus the full ranking, high -> low.
  std::pair<std::vector<int>, std::vector<Ranked>>
  best_match(const std::vector<std::pair<int, Fastsam::Crop>>& candidates,
             const std::vector<std::vector<float>>& text_features, double min_score, int timeout_ms);

  void close();

 private:
  std::vector<float> stack_;   // kClipBatch*256*256*3 input buffer, reused across encode() calls
  std::vector<cv::Mat> rows_;  // 256x256 CV_32FC3 views into stack_
  std::mutex mu_;
  std::unique_ptr<simaai::neat::Model> model_;
  simaai::neat::Model::Runner runner_;
  CropPool crop_pool_;
};

}  // namespace app::clip
