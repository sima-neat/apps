// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @example efficientsam3-promptable-segmenter.cpp
 * Single-stream RTSP promptable segmentation with EfficientSAM3 and Insight output.
 */
#include "clip_tokenizer.h"
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kPullTimeoutMs = 20000;
constexpr int kModelSize = 1008;
constexpr int kTextTokens = 16;
constexpr int kTextDim = 256;
constexpr int kVocabSize = 49408;
// Three frames on the model keep the MLA busy while the host prepares and reads frames (13 FPS; 11
// FPS with two).
constexpr int kPipelineDepth = 3;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct Config {
  std::string model_path;
  std::string text_encoder;
  std::string prompt;
  std::string rtsp_url;
  bool tcp = true;
  int latency_ms = 100;
  int frames = 0;
  double min_score = 0.3;
  int max_detections = 20;
  double mask_threshold = 0.5;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host;
  int video_port = 9000;
  int metadata_port = 9100;
  std::string save_dir;
  int save_every = 0;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.text_encoder = raw.string_or("model.text_encoder", "");
  cfg.prompt = raw.string_or("prompt.text", "");
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.3);
  cfg.max_detections = raw.int_or("inference.max_detections", 20);
  cfg.mask_threshold = raw.double_or("inference.mask_threshold", 0.5);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  return cfg;
}

// -------------------------------------------------------------------------------- pipeline

neat::Tensor ev74_tensor(const std::vector<float>& values, std::vector<int64_t> shape) {
  neat::Tensor tensor =
      neat::Tensor::from_vector(values, std::move(shape), neat::TensorMemory::EV74);
  tensor.layout = neat::TensorLayout::HWC;
  return tensor;
}

std::pair<std::vector<float>, int> encode_prompt(const Config& cfg) {
  const fs::path vocab =
      sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() /
      "bpe_simple_vocab_16e6.txt.gz";
  const std::vector<int64_t> tokens = ClipTokenizer(vocab).encode(cfg.prompt, kTextTokens);
  // The MLA cannot look up token ids, so the text encoder takes them one-hot.
  std::vector<float> one_hot(static_cast<std::size_t>(kTextTokens) * kVocabSize, 0.0F);
  for (int i = 0; i < kTextTokens; ++i) {
    one_hot[static_cast<std::size_t>(i) * kVocabSize + tokens[i]] = 1.0F;
  }
  const neat::Tensor input = ev74_tensor(one_hot, {1, kTextTokens, kVocabSize});
  neat::Model model(cfg.text_encoder);
  neat::Model::Runner encoder = model.build(neat::TensorList{input});
  encoder.push(neat::TensorList{input});
  const std::vector<float> features =
      sima_examples::tensor_to_floats(encoder.pull(kPullTimeoutMs).tensors.at(0));
  encoder.close();
  std::vector<float> text(static_cast<std::size_t>(kTextTokens + 1) * kTextDim, 0.0F);
  std::copy_n(features.begin(), kTextTokens * kTextDim, text.begin());
  int count = 0;
  for (int i = 0; i < kTextTokens; ++i) {
    text[kTextTokens * kTextDim + i] = tokens[i] != 0 ? 1.0F : 0.0F;
    count += tokens[i] != 0 ? 1 : 0;
  }
  return {text, count};
}

sima_examples::RtspStreamInfo probe_stream(const Config& cfg) {
  sima_examples::RtspProbeOptions options;
  options.latency_ms = cfg.latency_ms;
  options.rtsp_tcp = cfg.tcp;
  sima_examples::RtspStreamInfo stream;
  sima_examples::require(sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, options, stream) &&
                             stream.width > 0 && stream.height > 0 && stream.fps > 0,
                         "could not read the frame size and rate of " + cfg.rtsp_url);
  return stream;
}

struct Source {
  neat::Graph graph; // the run refers to the graph, so it lives here until the run closes
  neat::Run run;
};

Source build_source(const Config& cfg, int width, int height, int fps) {
  neat::nodes::groups::RtspDecodedInputOptions source;
  source.url = cfg.rtsp_url;
  source.tcp = cfg.tcp;
  source.latency_ms = cfg.latency_ms;
  source.fallback_h264_width = width;
  source.fallback_h264_height = height;
  source.fallback_h264_fps = fps;

  auto video = neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(width, height, fps);
  video.host = cfg.insight_host;
  video.video_port_base = cfg.video_port;

  neat::OutputOptions frames = neat::OutputOptions::EveryFrame(8);
  frames.drop = true;

  Source built;
  const neat::Graph decoded = neat::nodes::groups::RtspDecodedInput(source);
  built.graph.connect(decoded, neat::nodes::groups::VideoSender(video));
  built.graph.connect(decoded, neat::nodes::Output("frame", frames));
  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  built.run = built.graph.build(run_options);
  return built;
}

cv::Mat nv12(const neat::Tensor& tensor) {
  cv::Mat frame(tensor.height() * 3 / 2, tensor.width(), CV_8UC1);
  // contiguous() drops the row padding the decoder adds to some frames.
  sima_examples::require(tensor.contiguous().copy_payload_bytes_to(frame.data, frame.total()),
                         "decoded frame is not NV12");
  return frame;
}

void preprocess(const cv::Mat& frame, cv::Mat& out) {
  const int height = frame.rows * 2 / 3;
  const int width = frame.cols;
  const bool shrink = width > kModelSize || height > kModelSize;
  const int interpolation = shrink ? cv::INTER_AREA : cv::INTER_LINEAR;
  cv::Mat y, uv, rgb;
  cv::resize(frame.rowRange(0, height), y, {kModelSize, kModelSize}, 0, 0, interpolation);
  const cv::Mat uv_plane(height / 2, width / 2, CV_8UC2, const_cast<uchar*>(frame.ptr(height)));
  cv::resize(uv_plane, uv, {kModelSize / 2, kModelSize / 2}, 0, 0, interpolation);
  cv::cvtColorTwoPlane(y, uv, rgb, cv::COLOR_YUV2RGB_NV12);
  rgb.convertTo(out, CV_32FC3, 1.0 / 127.5, -1.0);
}

// ----------------------------------------------------------------------------- postprocess

using Box = std::array<int, 4>; // x0, y0, x1, y1 in frame pixels

// Python's round(): halves go to the even neighbour.
int round_even(double value) {
  return static_cast<int>(std::nearbyint(value));
}

std::vector<cv::Point> mask_outline(const cv::Mat& mask_logits, const Box& box, int width,
                                    int height, float logit_threshold) {
  const int mask_h = mask_logits.rows;
  const int mask_w = mask_logits.cols;
  const double cell_w = static_cast<double>(width) / mask_w;
  const double cell_h = static_cast<double>(height) / mask_h;
  const auto [x0, y0, x1, y1] = box;
  const int mx0 = std::max(0, static_cast<int>(std::floor(x0 / cell_w)) - 1);
  const int my0 = std::max(0, static_cast<int>(std::floor(y0 / cell_h)) - 1);
  const int mx1 = std::min(mask_w, static_cast<int>(std::ceil(x1 / cell_w)) + 1);
  const int my1 = std::min(mask_h, static_cast<int>(std::ceil(y1 / cell_h)) + 1);
  const int out_x0 = round_even(mx0 * cell_w);
  const int out_y0 = round_even(my0 * cell_h);
  const cv::Size out_size(round_even(mx1 * cell_w) - out_x0, round_even(my1 * cell_h) - out_y0);
  // Upscale the mask cells under the box before thresholding, as SAM 3 does, so the outline follows
  // the interpolated mask.
  cv::Mat upscaled;
  cv::resize(mask_logits(cv::Range(my0, my1), cv::Range(mx0, mx1)), upscaled, out_size, 0, 0,
             cv::INTER_LINEAR);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(upscaled > logit_threshold, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const auto& largest =
      *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  std::vector<cv::Point> polygon;
  cv::approxPolyDP(largest, polygon, 0.004 * cv::arcLength(largest, true), true);
  if (polygon.size() < 3) {
    return {};
  }
  for (cv::Point& point : polygon) {
    point += cv::Point(out_x0, out_y0);
  }
  return polygon;
}

struct Segment {
  std::string id;
  std::string label;
  double confidence = 0.0;
  Box box{};
  std::vector<cv::Point> mask;
};

// detections: queries x 6 (x1, y1, x2, y2, score, -) in model pixels; masks: mask_h x mask_w x
// queries logits.
std::vector<Segment> segments_of(const cv::Mat& detections, const cv::Mat& masks, const Config& cfg,
                                 int width, int height) {
  const auto logit_threshold =
      static_cast<float>(std::log(cfg.mask_threshold / (1.0 - cfg.mask_threshold)));
  std::vector<int> best(detections.rows);
  std::iota(best.begin(), best.end(), 0);
  std::stable_sort(best.begin(), best.end(), [&](int a, int b) {
    return detections.at<float>(a, 4) > detections.at<float>(b, 4);
  });
  best.resize(std::min<std::size_t>(best.size(), cfg.max_detections));
  const double sx = static_cast<double>(width) / kModelSize;
  const double sy = static_cast<double>(height) / kModelSize;
  std::vector<Segment> segments;
  int n = 0;
  for (const int query : best) {
    const float* row = detections.ptr<float>(query);
    if (!(row[4] > static_cast<float>(cfg.min_score))) {
      continue;
    }
    ++n;
    const int x0 = std::clamp(static_cast<int>(std::floor(row[0] * sx)), 0, width - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(row[1] * sy)), 0, height - 1);
    const Box box{x0, y0,
                  std::max(x0 + 1, std::min(width, static_cast<int>(std::ceil(row[2] * sx)))),
                  std::max(y0 + 1, std::min(height, static_cast<int>(std::ceil(row[3] * sy))))};
    cv::Mat mask_logits;
    cv::extractChannel(masks, mask_logits, query);
    std::vector<cv::Point> outline = mask_outline(mask_logits, box, width, height, logit_threshold);
    if (!outline.empty()) {
      segments.push_back({"seg_" + std::to_string(n), cfg.prompt,
                          std::nearbyint(row[4] * 1e4) / 1e4, box, std::move(outline)});
    }
  }
  return segments;
}

std::string segments_json(const std::vector<Segment>& segments) {
  nlohmann::ordered_json items = nlohmann::ordered_json::array();
  for (const Segment& segment : segments) {
    nlohmann::ordered_json mask = nlohmann::ordered_json::array();
    for (const cv::Point& point : segment.mask) {
      mask.push_back({point.x, point.y});
    }
    const auto& [x0, y0, x1, y1] = segment.box;
    items.push_back({{"id", segment.id},
                     {"label", segment.label},
                     {"confidence", segment.confidence},
                     {"bbox", {x0, y0, x1 - x0, y1 - y0}},
                     {"mask_format", "polygon"},
                     {"mask", mask}});
  }
  return nlohmann::ordered_json{{"segments", items}}.dump();
}

// Insight draws metadata only on the frame with its timestamp, and the model segments every second
// or third frame, so every frame is sent the result of the segmented frame nearest to it in time.
class OverlayClock {
public:
  struct Message {
    std::string data;
    int64_t timestamp_ms;
    std::string frame_id;
  };

  void add_frame(const neat::Sample& sample) {
    waiting_.emplace_back(sample.pts_ns / 1'000'000, std::to_string(sample.frame_id));
  }

  std::vector<Message> add_result(int64_t pts_ms, const std::string& data) {
    std::vector<Message> messages;
    while (!waiting_.empty() && waiting_.front().first <= pts_ms) {
      const auto [frame_pts, frame_id] = waiting_.front();
      waiting_.pop_front();
      const bool closer_to_previous =
          previous_ && frame_pts - previous_->first < pts_ms - frame_pts;
      messages.push_back({closer_to_previous ? previous_->second : data, frame_pts, frame_id});
    }
    previous_ = {pts_ms, data};
    return messages;
  }

private:
  std::deque<std::pair<int64_t, std::string>> waiting_;
  std::optional<std::pair<int64_t, std::string>> previous_;
};

std::optional<neat::Sample> newest_frame(neat::Run& run, OverlayClock& overlay, int timeout_ms) {
  std::optional<neat::Sample> newest;
  while (std::optional<neat::Sample> sample = run.pull("frame", timeout_ms)) {
    overlay.add_frame(*sample);
    newest = std::move(sample);
    timeout_ms = 0;
  }
  return newest;
}

void save_frame(const fs::path& path, const cv::Mat& frame, const std::vector<Segment>& segments) {
  const cv::Scalar color(255, 191, 0);
  cv::Mat bgr;
  cv::cvtColor(frame, bgr, cv::COLOR_YUV2BGR_NV12);
  std::vector<std::vector<cv::Point>> outlines;
  for (const Segment& segment : segments) {
    outlines.push_back(segment.mask);
  }
  cv::Mat filled = bgr.clone();
  cv::fillPoly(filled, outlines, color);
  cv::Mat annotated;
  cv::addWeighted(filled, 0.5, bgr, 0.5, 0.0, annotated);
  cv::polylines(annotated, outlines, true, color, 2);
  for (const Segment& segment : segments) {
    const auto [x0, y0, x1, y1] = segment.box;
    cv::rectangle(annotated, cv::Point(x0, y0), cv::Point(x1, y1), color, 2);
    cv::putText(annotated, segment.label + cv::format(" %.2f", segment.confidence),
                cv::Point(x0, std::max(12, y0 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);
  }
  cv::imwrite(path.string(), annotated);
}

// ------------------------------------------------------------------------------------- run

void run(const Config& cfg) {
  const auto [text, tokens] = encode_prompt(cfg);
  const neat::Tensor text_tensor = ev74_tensor(text, {1, kTextTokens + 1, kTextDim});
  neat::RunOptions model_options;
  model_options.queue_depth = kPipelineDepth;
  std::vector<neat::Tensor> images;
  for (int i = 0; i <= kPipelineDepth; ++i) {
    images.push_back(ev74_tensor(std::vector<float>(kModelSize * kModelSize * 3, 0.0F),
                                 {kModelSize, kModelSize, 3}));
  }
  neat::Model model(cfg.model_path);
  neat::Model::Runner runner = model.build(neat::TensorList{images[0], text_tensor},
                                           neat::Model::RouteOptions{}, model_options);
  const sima_examples::RtspStreamInfo stream = probe_stream(cfg);
  const int width = stream.width;
  const int height = stream.height;
  Source source = build_source(cfg, width, height, stream.fps);
  neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.metadata_port_base = cfg.metadata_port;
  const neat::MetadataSender metadata(metadata_options);
  std::cout << "rtsp=" << cfg.rtsp_url << " stream=" << width << "x" << height << "@" << stream.fps
            << " prompt='" << cfg.prompt << "' tokens=" << tokens << " insight=" << cfg.insight_host
            << " video=" << cfg.video_port << " metadata=" << cfg.metadata_port << " channel=0"
            << std::endl;

  OverlayClock overlay;
  std::deque<std::pair<int64_t, std::optional<cv::Mat>>> in_flight;
  int pushed = 0;
  int processed = 0;
  int window_instances = 0;
  double window_start_ms = sima_examples::time_ms();
  g_stop_requested = 0;
  const auto previous_sigint = std::signal(SIGINT, request_stop);
  const auto finish = [&] {
    std::signal(SIGINT, previous_sigint);
    source.run.close();
    runner.close();
    std::cout << "processed=" << processed << " video_sender=" << cfg.insight_host << ":"
              << cfg.video_port << std::endl;
  };
  try {
    while (g_stop_requested == 0 && (cfg.frames <= 0 || processed < cfg.frames)) {
      if (static_cast<int>(in_flight.size()) < kPipelineDepth) {
        std::optional<neat::Sample> sample =
            newest_frame(source.run, overlay, in_flight.empty() ? kPullTimeoutMs : 0);
        sima_examples::require(sample || !in_flight.empty(), "timed out waiting for a frame");
        if (sample) {
          const cv::Mat frame = nv12(sample->tensors.at(0));
          const neat::Tensor& image = images[pushed % images.size()];
          {
            neat::Mapping mapping = image.map_write();
            cv::Mat out(kModelSize, kModelSize, CV_32FC3, mapping.data);
            preprocess(frame, out);
          }
          runner.push(neat::TensorList{image, text_tensor});
          ++pushed;
          const bool save =
              !cfg.save_dir.empty() && cfg.save_every > 0 && pushed % cfg.save_every == 0;
          in_flight.emplace_back(sample->pts_ns / 1'000'000,
                                 save ? std::optional<cv::Mat>(frame) : std::nullopt);
          continue;
        }
      }

      const neat::Sample result = runner.pull(kPullTimeoutMs);
      sima_examples::require(result.tensors.size() == 2, "timed out waiting for the model");
      const auto [pts_ms, saved_frame] = in_flight.front();
      in_flight.pop_front();
      const neat::Tensor& detections = result.tensors[0];
      const neat::Tensor& masks = result.tensors[1];
      const neat::Mapping detections_view = detections.view_read();
      const neat::Mapping masks_view = masks.view_read();
      const auto queries = static_cast<int>(masks.shape.at(2));
      const cv::Mat detection_rows(queries, static_cast<int>(detections.shape.at(2)), CV_32F,
                                   detections_view.data);
      const cv::Mat mask_planes(static_cast<int>(masks.shape.at(0)),
                                static_cast<int>(masks.shape.at(1)), CV_32FC(queries),
                                masks_view.data);
      const std::vector<Segment> segments =
          segments_of(detection_rows, mask_planes, cfg, width, height);
      for (const auto& message : overlay.add_result(pts_ms, segments_json(segments))) {
        std::string error;
        if (!metadata.send_metadata("segmentation", message.data, message.timestamp_ms,
                                    message.frame_id, &error) &&
            !error.empty()) {
          throw std::runtime_error("Insight metadata send failed: " + error);
        }
      }

      ++processed;
      window_instances += static_cast<int>(segments.size());
      if (saved_frame) {
        save_frame(fs::path(cfg.save_dir) / ("frame_" + std::to_string(processed) + ".jpg"),
                   *saved_frame, segments);
      }
      if (cfg.profile && processed % cfg.profile_interval == 0) {
        const double elapsed_s = (sima_examples::time_ms() - window_start_ms) / 1000.0;
        std::cout << cv::format("[profile] frames=%d segmentation_fps=%.1f avg_instances=%.2f",
                                cfg.profile_interval, cfg.profile_interval / elapsed_s,
                                static_cast<double>(window_instances) / cfg.profile_interval)
                  << std::endl;
        window_start_ms = sima_examples::time_ms();
        window_instances = 0;
      }
    }
  } catch (...) {
    finish();
    throw;
  }
  finish();
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config") {
        sima_examples::require(i + 1 < argc, "--config requires a path");
        config_path = argv[++i];
      } else if (arg == "--help" || arg == "-h") {
        std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }
    const Config cfg = load_config(config_path);
    if (!cfg.save_dir.empty()) {
      fs::create_directories(cfg.save_dir);
    }
    run(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
