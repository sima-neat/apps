#pragma once

#include "config.h"
#include "fastsam.h"
#include "image_encoder.h"
#include "neat.h"
#include "profiling.h"

#include <nodes/io/MetadataSender.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace app {

struct StreamInfo {
  int width = 0;
  int height = 0;
  int fps = 0;
};

struct Frame {
  cv::Mat rgb;
  std::int64_t pts_ns = -1;
  std::int64_t frame_id = -1;
};

struct Segment {
  float confidence = 0.0f;
  std::vector<cv::Point> polygon;  // frame pixels
};

simaai::neat::RunOptions make_run_options(int queue_depth, int build_timeout_ms = 30000);

StreamInfo probe_rtsp(const std::string& url);

struct StreamRuntime {
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  int frame_w = 0;
  int frame_h = 0;
  int out_fps = 0;
  int video_port = 0;
};

class LatestFrame {
 public:
  void store(Frame frame);
  std::optional<Frame> take(int& dropped);
  void close();
  bool done() const;

 private:
  mutable std::mutex mu_;
  std::optional<Frame> frame_;
  int overwrites_ = 0;
  bool closed_ = false;
};

class Segmenter {
 public:
  Segmenter(const AppConfig& cfg, const StreamInfo& info);

  // Per-frame pipeline: segment the frame with FastSAM, decode boxes+masks, pick the
  // prompt-matching anchor via CLIP, then track it by IoU and return polygons.
  std::vector<Segment> run(const AppConfig& cfg, const cv::Mat& rgb, Fastsam& fastsam,
                           clip::ImageEncoder& image_encoder,
                           const std::vector<std::vector<float>>& text_features);

  void close();

 private:
  // Crop each mask's object, CLIP-encode it, and return the mask whose crop best matches the
  // prompt (empty if none clears min_score). Runs on the background thread.
  cv::Mat clip_select(const AppConfig& cfg, clip::ImageEncoder& image_encoder,
                      const std::vector<std::vector<float>>& text_features, const cv::Mat& rgb,
                      const std::vector<cv::Mat>& masks) const;
                      
  // Return the mask index that best overlaps anchor_ by IoU (and advance anchor_ to it),
  // or empty if none clears track_iou_ or there is no anchor yet.
  std::vector<int> track(const Fastsam::Segmentation& seg, int n);

  Fastsam::Geometry geom_;
  int clip_interval_;
  double track_iou_;
  int since_clip_;
  cv::Mat anchor_;
  std::future<cv::Mat> clip_future_;
  bool clip_pending_ = false;
};

// Per-stream orchestration and I/O
class Stream {
 public:
  Stream(const AppConfig& cfg, int index, std::string url, const StreamInfo& info);
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  void start(const AppConfig& cfg);

  enum class Pull { Frame, Idle, Closed };
  Pull pull_frame();
  void close_source();

  bool process(const AppConfig& cfg, Fastsam& fastsam, clip::ImageEncoder& image_encoder,
               const std::vector<std::vector<float>>& text_features, const std::string& label);

  bool frame_limit_reached(const AppConfig& cfg) const;
  bool done() const;
  void close();

  int index() const { return index_; }
  long long processed() const { return processed_; }
  void flush_profile() { profile_.flush(); }

 private:
  int index_;
  std::string url_;
  StreamInfo info_;
  LatestFrame latest_frame_;
  Segmenter segmenter_;

  std::optional<StreamRuntime> graph_;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender_;
  ProfileWindow profile_;
  long long processed_ = 0;
};

}  // namespace app
