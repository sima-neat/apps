#include "pipeline.h"

#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/example_utils.h"

#include <graphs/Fragments.h>
#include <nodes/groups/VideoSender.h>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace neat = simaai::neat;

namespace app {

namespace {

constexpr int kDefaultFps = 30;
constexpr int kPullTimeoutMs = 100;
inline constexpr const char* kChannelInfer = "infer";
inline constexpr const char* kChannelVideo = "video";

std::int64_t wall_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

int target_fps(const AppConfig& cfg, int src_fps) {
  return (cfg.max_fps > 0) ? std::min(src_fps, cfg.max_fps) : src_fps;
}

neat::nodes::groups::RtspDecodedInputOptions make_source_options(const AppConfig& cfg,
                                                                 const std::string& url, int fps,
                                                                 int width, int height) {
  neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.fallback_h264_width = width;
  opt.fallback_h264_height = height;
  opt.fallback_h264_fps = fps;
  opt.output_caps.enable = true;
  opt.output_caps.format = "NV12";
  opt.output_caps.width = width;
  opt.output_caps.height = height;
  opt.output_caps.fps = fps;
  opt.output_caps.memory = neat::CapsMemory::Any;
  return opt;
}

StreamRuntime build_stream(const AppConfig& cfg, int stream_index, const std::string& url, int frame_w,
                           int frame_h, int fps) {
  const int out_fps = target_fps(cfg, fps);

  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;

  std::vector<std::string> outputs = {kChannelInfer};
  if (cfg.video_enabled) {
    outputs.push_back(kChannelVideo);
  }
  auto branch = neat::graphs::Branch("source", outputs);

  neat::Graph graph;
  if (out_fps < fps) {
    neat::Graph source("rate_source");
    source.add(neat::nodes::groups::RtspDecodedInput(make_source_options(cfg, url, fps, frame_w, frame_h)));
    source.add(neat::nodes::VideoRate());
    source.add(neat::nodes::CapsRaw("NV12", frame_w, frame_h, out_fps));
    graph.connect(source, branch, link);
  } else {
    auto source = neat::nodes::groups::RtspDecodedInput(make_source_options(cfg, url, fps, frame_w, frame_h));
    graph.connect(source, branch, link);
  }

  neat::Graph infer_graph(kChannelInfer);
  infer_graph.add(neat::nodes::Output(kChannelInfer, neat::OutputOptions::Latest()));
  graph.connect(branch, infer_graph, link);

  int video_port = 0;
  if (cfg.video_enabled) {
    auto sender_opt = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(frame_w, frame_h,
                                                                                std::max(1, out_fps));
    sender_opt.host = cfg.insight_host;
    sender_opt.channel = stream_index;
    sender_opt.video_port_base = cfg.video_port_base;
    sender_opt.encoder.bitrate_kbps = cfg.bitrate_kbps;
    video_port = sender_opt.video_port();

    neat::Graph video_graph(kChannelVideo);
    video_graph.connect(neat::nodes::Input(kChannelVideo),
                        neat::nodes::groups::VideoSender(sender_opt));
    graph.connect(branch, video_graph, link);
  }

  neat::RunOptions run_opt;
  run_opt.preset = neat::RunPreset::Realtime;
  run_opt.queue_depth = 3;
  run_opt.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_opt.output_memory = neat::OutputMemory::Owned;
  neat::Run run = graph.build(run_opt);
  return StreamRuntime{std::move(graph), std::move(run), frame_w, frame_h, out_fps, video_port};
}

cv::Mat tensor_rgb_from_decoded(const neat::Tensor& tensor, int width, int height) {
  const auto payload = tensor.copy_payload_bytes();
  const std::size_t need = static_cast<std::size_t>(width) * height * 3 / 2;
  if (payload.size() < need) {
    throw std::runtime_error("decoded frame payload smaller than expected NV12 size");
  }
  const cv::Mat nv12(height * 3 / 2, width, CV_8UC1, const_cast<std::uint8_t*>(payload.data()));
  cv::Mat rgb;
  cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
  return rgb;
}

std::unique_ptr<neat::MetadataSender> build_metadata_sender(const AppConfig& cfg, int stream_index) {
  neat::MetadataSenderOptions opt;
  opt.host = cfg.insight_host;
  opt.channel = stream_index;
  opt.metadata_port_base = cfg.metadata_port_base;
  std::string err;
  auto sender = std::make_unique<neat::MetadataSender>(opt, &err);
  sima_examples::require(sender->ok(), "metadata sender init failed: " + err);
  return sender;
}

std::string metadata_segments_json(const std::vector<Segment>& segments, const std::string& label) {
  nlohmann::json arr = nlohmann::json::array();
  int n = 1;
  for (const auto& seg : segments) {
    if (seg.polygon.empty()) {
      continue;
    }
    nlohmann::json poly = nlohmann::json::array();
    for (const auto& p : seg.polygon) {
      poly.push_back(nlohmann::json::array({p.x, p.y}));
    }
    nlohmann::json entry;
    entry["id"] = "seg_" + std::to_string(n++);
    entry["label"] = label;
    entry["confidence"] = static_cast<double>(seg.confidence);
    entry["mask_format"] = "polygon";
    entry["mask"] = std::move(poly);
    arr.push_back(std::move(entry));
  }
  nlohmann::json root;
  root["segments"] = std::move(arr);
  return root.dump();
}

void send_metadata(neat::MetadataSender& sender, const std::string& json, long long frame_id,
                   std::int64_t timestamp_ms) {
  std::string err;
  if (!sender.send_metadata("segmentation", json, timestamp_ms, std::to_string(frame_id), &err)) {
    std::cerr << "[warn] metadata send failed: " << err << "\n";
  }
}

}  // namespace

neat::RunOptions make_run_options(int queue_depth, int build_timeout_ms) {
  neat::RunOptions opt;
  opt.queue_depth = queue_depth;
  opt.overflow_policy = neat::OverflowPolicy::Block;
  opt.preset = neat::RunPreset::Balanced;
  opt.input_timeout_ms = build_timeout_ms;
  opt.startup_preflight = true;
  return opt;
}

StreamInfo probe_rtsp(const std::string& url) {
  sima_examples::RtspStreamInfo probed;
  sima_examples::require(
      sima_examples::probe_rtsp_stream_info(url, sima_examples::RtspProbeOptions{}, probed),
      "failed to probe RTSP stream: " + url);
  StreamInfo info;
  info.width = probed.width;
  info.height = probed.height;
  info.fps = probed.fps > 0 ? probed.fps : kDefaultFps;
  return info;
}

void LatestFrame::store(Frame frame) {
  std::lock_guard<std::mutex> lock(mu_);
  if (closed_) {
    return;
  }
  if (frame_.has_value()) {
    ++overwrites_;
  }
  frame_ = std::move(frame);
}

std::optional<Frame> LatestFrame::take(int& dropped) {
  std::lock_guard<std::mutex> lock(mu_);
  dropped = overwrites_;
  overwrites_ = 0;
  if (!frame_.has_value()) {
    return std::nullopt;
  }
  std::optional<Frame> out = std::move(frame_);
  frame_.reset();
  return out;
}

void LatestFrame::close() {
  std::lock_guard<std::mutex> lock(mu_);
  closed_ = true;
}

bool LatestFrame::done() const {
  std::lock_guard<std::mutex> lock(mu_);
  return closed_ && !frame_.has_value();
}

Segmenter::Segmenter(const AppConfig& cfg, const StreamInfo& info)
    : geom_(Fastsam::get_letterbox_geometry(info.width, info.height, cfg.infer_size)),
      clip_interval_(std::max(1, cfg.clip_interval)),
      track_iou_(cfg.track_iou),
      since_clip_(std::max(1, cfg.clip_interval)) {}

Stream::Stream(const AppConfig& cfg, int index, std::string url, const StreamInfo& info)
    : index_(index),
      url_(std::move(url)),
      info_(info),
      segmenter_(cfg, info),
      profile_(cfg.profile, cfg.profile_interval, index) {}

void Stream::start(const AppConfig& cfg) {
  graph_.emplace(build_stream(cfg, index_, url_, info_.width, info_.height, info_.fps));
  metadata_sender_ = build_metadata_sender(cfg, index_);
  std::cout << "[stream " << index_ << "] rtsp=" << url_ << " " << info_.width << "x" << info_.height
            << "@"
            << graph_->out_fps << " video="
            << (cfg.video_enabled ? std::to_string(graph_->video_port) : std::string("disabled"))
            << " metadata=" << metadata_sender_->metadata_port() << "\n";
}

Stream::Pull Stream::pull_frame() {
  neat::Sample sample;
  neat::PullError perr;
  const auto status = graph_->run.pull(kChannelInfer, kPullTimeoutMs, sample, &perr);
  if (status == neat::PullStatus::Timeout) {
    return Pull::Idle;
  }
  if (status == neat::PullStatus::Closed) {
    return Pull::Closed;
  }
  if (status != neat::PullStatus::Ok) {
    throw std::runtime_error("pull failed: " + perr.message);
  }
  const auto tensors = neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    return Pull::Idle;
  }
  Frame frame;
  frame.rgb = tensor_rgb_from_decoded(tensors.front(), info_.width, info_.height);
  frame.pts_ns = sample.pts_ns;
  frame.frame_id = sample.frame_id;
  latest_frame_.store(std::move(frame));
  return Pull::Frame;
}

void Stream::close_source() {
  latest_frame_.close();
}

bool Stream::process(const AppConfig& cfg, Fastsam& fastsam, clip::ImageEncoder& image_encoder,
                     const std::vector<std::vector<float>>& text_features, const std::string& label) {
  int dropped = 0;
  auto frame = latest_frame_.take(dropped);
  profile_.note_drop(dropped);
  if (!frame.has_value()) {
    return false;
  }

  const double frame_start = sima_examples::time_ms();
  const auto segments = segmenter_.run(cfg, frame->rgb, fastsam, image_encoder, text_features);

  ++processed_;
  if (processed_ <= cfg.warmup_frames) {
    return true;
  }
  const std::int64_t ts_ms = frame->pts_ns >= 0 ? frame->pts_ns / 1'000'000 : wall_ms();
  const long long frame_id = frame->frame_id >= 0 ? frame->frame_id : processed_;
  send_metadata(*metadata_sender_, metadata_segments_json(segments, label), frame_id, ts_ms);
  profile_.add(sima_examples::time_ms() - frame_start);
  return true;
}

std::vector<Segment> Segmenter::run(const AppConfig& cfg, const cv::Mat& rgb, Fastsam& fastsam,
                                    clip::ImageEncoder& image_encoder,
                                    const std::vector<std::vector<float>>& text_features) {
  const auto out = fastsam.run(rgb, cfg.timeout_ms);

  const int top_k = cfg.clip_max_crops > 0 ? cfg.clip_max_crops : 0;
  const Fastsam::Segmentation seg = Fastsam::decode(out, top_k);
  const int n = static_cast<int>(seg.boxes.size());

  // Get a finished background CLIP pick as the tracking anchor, then relaunch on interval.
  if (clip_pending_ &&
      clip_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    anchor_ = clip_future_.get();
    clip_pending_ = false;
  }
  ++since_clip_;
  if (!clip_pending_ && (since_clip_ >= clip_interval_ || anchor_.empty())) {
    std::vector<cv::Mat> snapshot;
    snapshot.reserve(n);
    for (int i = 0; i < n; ++i) {
      snapshot.push_back(seg.masks[i].clone());
    }
    clip_future_ = std::async(std::launch::async,
                              [this, &cfg, &image_encoder, &text_features, frame = rgb.clone(),
                               masks = std::move(snapshot)]() {
                                return clip_select(cfg, image_encoder, text_features, frame, masks);
                              });
    clip_pending_ = true;
    since_clip_ = 0;
  }

  // Follow the anchor across this frame and emit the matched mask as a frame-space polygon.
  std::vector<Segment> segments;
  for (const int i : track(seg, n)) {
    auto poly = Fastsam::mask_polygon(seg.masks[i], geom_);
    if (!poly.empty()) {
      segments.push_back(Segment{seg.boxes[i][4], std::move(poly)});
    }
  }
  return segments;
}

cv::Mat Segmenter::clip_select(const AppConfig& cfg, clip::ImageEncoder& image_encoder,
                            const std::vector<std::vector<float>>& text_features, const cv::Mat& rgb,
                            const std::vector<cv::Mat>& masks) const {
  std::vector<std::pair<int, Fastsam::Crop>> candidates;
  for (int i = 0; i < static_cast<int>(masks.size()); ++i) {
    auto crop = Fastsam::object_crop(masks[i], rgb, geom_, static_cast<double>(cfg.clip_min_area),
                                     cfg.clip_max_frac, cfg.clip_max_box_frac);
    if (crop) {
      candidates.emplace_back(i, std::move(*crop));
    }
  }
  const auto keep = image_encoder.best_match(candidates, text_features, cfg.clip_min_score,
                                             cfg.timeout_ms);
  if (keep.empty()) {
    return cv::Mat();
  }
  return masks[keep.front()].clone();
}

std::vector<int> Segmenter::track(const Fastsam::Segmentation& seg, int n) {
  if (anchor_.empty()) {
    return {};
  }
  const cv::Mat a = anchor_ > 0;
  const double a_sum = cv::countNonZero(a);
  int best_i = -1;
  double best_iou = 0.0;
  for (int i = 0; i < n; ++i) {
    const cv::Mat m = seg.masks[i] > 0;
    cv::Mat inter;
    cv::bitwise_and(a, m, inter);
    const double inter_sum = cv::countNonZero(inter);
    if (inter_sum == 0.0) {
      continue;
    }
    const double uni = a_sum + cv::countNonZero(m) - inter_sum;
    const double iou = uni > 0.0 ? inter_sum / uni : 0.0;
    if (iou > best_iou) {
      best_iou = iou;
      best_i = i;
    }
  }
  if (best_i >= 0 && best_iou >= track_iou_) {
    anchor_ = seg.masks[best_i].clone();
    return {best_i};
  }
  return {};
}

bool Stream::frame_limit_reached(const AppConfig& cfg) const {
  return cfg.frames > 0 && processed_ >= cfg.frames;
}

bool Stream::done() const {
  return latest_frame_.done();
}

void Segmenter::close() {
  if (clip_pending_ && clip_future_.valid()) {
    clip_future_.wait();
    clip_pending_ = false;
  }
}

void Stream::close() {
  segmenter_.close();
  if (graph_) {
    graph_->run.close();
  }
}

}  // namespace app
