#pragma once

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
constexpr int kClipImagePx = 256;  // MobileCLIP-S2 image encoder input side

// MobileCLIP image encoder
class ImageEncoder {
 public:
  ImageEncoder(const std::string& model_path, const simaai::neat::RunOptions& run_opt);

  // Encode crops to [512] feature vectors, one per crop.
  std::vector<std::vector<float>> encode(const std::vector<Fastsam::Crop>& crops, int timeout_ms);

  std::vector<int> best_match(const std::vector<std::pair<int, Fastsam::Crop>>& candidates,
             const std::vector<std::vector<float>>& text_features, double min_score, int timeout_ms);

  void close();

 private:
  std::vector<float> stack_;   // kClipBatch*256*256*3 input buffer, reused across encode() calls
  std::vector<cv::Mat> rows_;  // 256x256 CV_32FC3 views into stack_
  std::mutex mu_;
  std::unique_ptr<simaai::neat::Model> model_;
  simaai::neat::Model::Runner runner_;
};

}  // namespace app::clip
