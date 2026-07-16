#include "image_encoder.h"

#include "support/runtime/example_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neat = simaai::neat;

namespace app::clip {
namespace {

neat::Tensor make_input_tensor(const std::vector<float>& stack) {
  return neat::Tensor::from_vector(stack, {kClipBatch, kClipImagePx, kClipImagePx, 3},
                                   neat::TensorMemory::EV74);
}

// Resize + centre-crop window_rgb to px*px, mask the background out to bg, write float32 into dst.
void crop_into(cv::Mat& dst, const cv::Mat& window_rgb, const cv::Mat& submask,
               int px = kClipImagePx, float bg = 1.0f) {
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

// Softmax over 100 * cosine(image, text[0]) -- the single-query CLIP retrieval score.
std::vector<double> retrieve_scores(const std::vector<std::vector<float>>& image_features,
                                    const std::vector<std::vector<float>>& text_features) {
  const std::size_t n = image_features.size();
  std::vector<double> probs(n, 0.0);
  if (n == 0 || text_features.empty()) {
    return probs;
  }

  const auto& query = text_features.front();
  double qnorm = 0.0;
  for (const float v : query) {
    qnorm += static_cast<double>(v) * v;
  }
  qnorm = std::sqrt(qnorm);
  if (qnorm == 0.0) {
    return probs;
  }

  std::vector<double> logits(n, 0.0);
  double max_logit = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const auto& im = image_features[i];
    const std::size_t dim = std::min(im.size(), query.size());
    double dot = 0.0;
    double inorm = 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
      dot += static_cast<double>(im[k]) * query[k];
      inorm += static_cast<double>(im[k]) * im[k];
    }
    inorm = std::sqrt(inorm);
    logits[i] = inorm > 0.0 ? 100.0 * dot / (inorm * qnorm) : 0.0;
    max_logit = std::max(max_logit, logits[i]);
  }

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum > 0.0) {
    for (std::size_t i = 0; i < n; ++i) {
      probs[i] /= sum;
    }
  }
  return probs;
}

}  // namespace

ImageEncoder::ImageEncoder(const std::string& model_path, const neat::RunOptions& run_opt)
    : stack_(static_cast<std::size_t>(kClipBatch) * kClipImagePx * kClipImagePx * 3, 0.0f),
      model_(std::make_unique<neat::Model>(model_path)),
      runner_(model_->build(neat::TensorList{make_input_tensor(stack_)},
                            neat::Model::RouteOptions{}, run_opt)) {
  const std::size_t img = static_cast<std::size_t>(kClipImagePx) * kClipImagePx * 3;
  rows_.reserve(kClipBatch);
  for (int i = 0; i < kClipBatch; ++i) {
    rows_.emplace_back(kClipImagePx, kClipImagePx, CV_32FC3, stack_.data() + i * img);
  }
}

std::vector<std::vector<float>> ImageEncoder::encode(const std::vector<Fastsam::Crop>& crops,
                                                     int timeout_ms) {
  // Serialize the whole call: every stream shares this one input buffer and model runner.
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::vector<float>> feats;
  feats.reserve(crops.size());
  const std::size_t img = static_cast<std::size_t>(kClipImagePx) * kClipImagePx * 3;

  for (std::size_t start = 0; start < crops.size(); start += kClipBatch) {
    const std::size_t end = std::min(crops.size(), start + static_cast<std::size_t>(kClipBatch));
    const int n = static_cast<int>(end - start);
    if (n < kClipBatch) {
      std::fill(stack_.begin() + static_cast<std::size_t>(n) * img, stack_.end(), 0.0f);
    }
    // Crops within a batch are prepped in parallel over OpenCV's thread pool.
    cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& range) {
      for (int i = range.start; i < range.end; ++i) {
        const Fastsam::Crop& crop = crops[start + static_cast<std::size_t>(i)];
        crop_into(rows_[i], crop.window, crop.submask);
      }
    });

    const neat::Tensor inp = make_input_tensor(stack_);
    const auto out = runner_.run(neat::TensorList{inp}, timeout_ms);
    if (out.empty()) {
      throw std::runtime_error("image encoder returned no output");
    }
    const auto vals = sima_examples::tensor_to_floats(out.front());
    const int dim = static_cast<int>(vals.size() / static_cast<std::size_t>(kClipBatch));
    for (int i = 0; i < n; ++i) {
      feats.emplace_back(vals.begin() + static_cast<std::size_t>(i) * dim,
                         vals.begin() + static_cast<std::size_t>(i + 1) * dim);
    }
  }
  return feats;
}

void ImageEncoder::close() {
  std::lock_guard<std::mutex> lock(mu_);
  runner_.close();
}

std::vector<int>
ImageEncoder::best_match(const std::vector<std::pair<int, Fastsam::Crop>>& candidates,
                         const std::vector<std::vector<float>>& text_features, double min_score,
                         int timeout_ms) {
  if (candidates.empty()) {
    return {};
  }
  std::vector<int> idxs;
  std::vector<Fastsam::Crop> crops;
  idxs.reserve(candidates.size());
  crops.reserve(candidates.size());
  for (const auto& c : candidates) {
    idxs.push_back(c.first);
    crops.push_back(c.second);
  }

  const auto image_features = encode(crops, timeout_ms);
  const auto scores = retrieve_scores(image_features, text_features);

  int best_idx = -1;
  double best_score = 0.0;
  for (std::size_t i = 0; i < idxs.size(); ++i) {
    const double score = i < scores.size() ? scores[i] : 0.0;
    if (best_idx < 0 || score > best_score) {
      best_idx = idxs[i];
      best_score = score;
    }
  }

  if (best_idx >= 0 && best_score >= min_score) {
    return {best_idx};
  }
  return {};
}

}  // namespace app::clip
