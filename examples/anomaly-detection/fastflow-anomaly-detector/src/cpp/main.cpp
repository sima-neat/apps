// Copyright 2026 SiMa Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @example fastflow-anomaly-detector.cpp
 * FastFlow anomaly detection on one RTSP stream, heatmap video to Insight.
 *
 *   RTSP decode (NV12) --> frame --> BGR --> model (CVU preprocess + MLA FastFlow)
 *                                |                             |
 *                                +--> heatmap, boxes and verdict map --> video_sender
 *                                     drawn on the frame        (H264 RTP/UDP -> Insight)
 *
 * FastFlow is trained on good parts only and returns a 256x256 map with the probability
 * that each pixel does not belong to a good part. Connected regions at or above
 * inference.threshold that cover at least inference.min_region_px map pixels are the
 * defects; smaller ones are the speckle a good part produces. The application blends a
 * heatmap over those regions, boxes them, adds the verdict banner, and sends the result
 * to Insight as H.264 video. output.save_dir writes the same frames as JPEGs.
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <csignal>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kPullTimeoutMs = 20000;
// On-screen frame rate window, in frames.
constexpr std::size_t kFpsWindow = 30;

// Set by Ctrl-C, so the loop ends and the summary still prints.
volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

// ---------------------------------------------------------------------------- config

struct Config {
  std::string model_path;
  std::array<float, 3> mean{}; // input normalisation the package expects, RGB, on 0..1 pixels
  std::array<float, 3> stddev{};
  std::string rtsp_url;
  bool tcp = true;
  int latency_ms = 100;
  int frames = 0;          // 0 runs until stopped
  double threshold = 0.5;  // map probability that counts as anomalous
  int min_region_px = 300; // smallest connected region, in map pixels, that counts as a defect
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host;
  int video_port = 9000;
  double heat_max = 0.7; // map probability drawn at full heatmap intensity
  double alpha = 0.55;   // heatmap opacity over the frame
  std::string save_dir;
  int save_every = 0;
};

/// One normalisation value per channel, in R, G, B order, from a list such as `[0.4, 0.4, 0.4]`.
std::array<float, 3> rgb_stats(const sima_examples::ScalarConfig& raw, const std::string& key,
                               std::array<float, 3> fallback) {
  const auto text = raw.string_value(key);
  if (!text) {
    return fallback;
  }
  std::string list = *text;
  std::replace_if(
      list.begin(), list.end(), [](char c) { return c == '[' || c == ']' || c == ','; }, ' ');
  std::istringstream items(list);
  std::array<float, 3> parsed{};
  std::string extra;
  if (!(items >> parsed[0] >> parsed[1] >> parsed[2]) || items >> extra) {
    throw std::runtime_error(key + " must be three numbers, one per RGB channel");
  }
  return parsed;
}

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.mean = rgb_stats(raw, "model.normalize.mean", {0.3893437F, 0.35421189F, 0.36142577F});
  cfg.stddev = rgb_stats(raw, "model.normalize.stddev", {1.0F, 1.0F, 1.0F});
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.threshold = raw.double_or("inference.threshold", 0.5);
  cfg.min_region_px = raw.int_or("inference.min_region_px", 300);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.heat_max = raw.double_or("output.heat_max", 0.7);
  cfg.alpha = raw.double_or("output.alpha", 0.55);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);

  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.rtsp_url.empty(), "source.rtsp_url must be set");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(std::find(cfg.stddev.begin(), cfg.stddev.end(), 0.0F) == cfg.stddev.end(),
                         "model.normalize.stddev must not contain zero");
  sima_examples::require(cfg.threshold >= 0.0 && cfg.threshold <= 1.0,
                         "inference.threshold must be between 0 and 1");
  sima_examples::require(cfg.min_region_px > 0, "inference.min_region_px must be > 0");
  sima_examples::require(cfg.alpha >= 0.0 && cfg.alpha <= 1.0,
                         "output.alpha must be between 0 and 1");
  sima_examples::require(cfg.heat_max > cfg.threshold,
                         "output.heat_max must be greater than inference.threshold");
  return cfg;
}

// -------------------------------------------------------------------------- pipeline

struct Source {
  simaai::neat::Graph graph; // the run refers to the graph, so it lives here until the
  simaai::neat::Run run;     // run closes
};

struct Detector {
  std::unique_ptr<simaai::neat::Model> model;
  int map_side = 0;
};

/// Frame size and rate, which the graph needs before it starts.
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

/// Graph: RTSP decode only, one "frame" output.
///
/// The model is not attached to the decoder: the CVU resize would pull the decoder's row
/// padding into the frame border and flag it as a defect.
Source build_source(const Config& cfg, int width, int height, int fps) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions source;
  source.url = cfg.rtsp_url;
  source.tcp = cfg.tcp;
  source.latency_ms = cfg.latency_ms;
  source.payload_type = 96;
  source.insert_queue = true;
  source.decoder_name = "decoder";
  source.decoder_raw_output = true;
  source.auto_caps_from_stream = true;
  source.fallback_h264_width = width;
  source.fallback_h264_height = height;
  source.fallback_h264_fps = fps;
  source.output_caps.enable = true;
  source.output_caps.format = "NV12";
  source.output_caps.width = width;
  source.output_caps.height = height;
  source.output_caps.fps = fps;
  source.output_caps.memory = simaai::neat::CapsMemory::Any;

  Source built;
  built.graph.connect(simaai::neat::nodes::groups::RtspDecodedInput(source),
                      simaai::neat::nodes::Output("frame",
                                                  simaai::neat::OutputOptions::EveryFrame(4)));
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  built.run = built.graph.build(run_options);
  return built;
}

/// A BGR frame in, the anomaly map out. Also carries the side of the square map.
Detector build_model(const Config& cfg, int width, int height) {
  simaai::neat::Model::Options options;
  options.preprocess.kind = simaai::neat::InputKind::Image;
  options.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  options.preprocess.normalize.mean = cfg.mean;
  options.preprocess.normalize.stddev = cfg.stddev;
  options.preprocess.normalize.has_explicit_stats = true;
  options.preprocess.input_max_width = width;
  options.preprocess.input_max_height = height;
  options.preprocess.input_max_depth = 3;
  Detector detector;
  detector.model = std::make_unique<simaai::neat::Model>(cfg.model_path, options);
  detector.map_side = static_cast<int>(detector.model->output_specs().at(0).shape.at(1));
  return detector;
}

/// A push graph: annotated frames in, H.264 RTP/UDP out to the Insight viewer.
struct InsightVideo {
  simaai::neat::Graph graph{"insight"};
  simaai::neat::Run run;
  int port = 0;

  InsightVideo(const Config& cfg, int width, int height, int fps) {
    simaai::neat::InputOptions source;
    source.payload_type = simaai::neat::PayloadType::Image;
    source.format = simaai::neat::FormatTag::RGB;
    source.width = width;
    source.height = height;
    source.depth = 3;
    source.fps_n = fps;
    source.fps_d = 1;
    source.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
    auto options =
        simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(width, height, fps);
    options.host = cfg.insight_host;
    options.channel = 0;
    options.video_port_base = cfg.video_port;
    options.encoder.bitrate_kbps = 4000;
    port = options.video_port();
    graph.add(simaai::neat::nodes::Input(source));
    graph.add(simaai::neat::nodes::groups::VideoSender(options));
    // The graph negotiates its caps from a first sample.
    const cv::Mat first(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    run = graph.build(simaai::neat::TensorList{tensor_of(first)});
  }

  /// An RGB frame as a tensor the encoder can read.
  static simaai::neat::Tensor tensor_of(const cv::Mat& rgb) {
    return simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
                                             simaai::neat::TensorMemory::EV74);
  }

  void send(const cv::Mat& frame) {
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    sima_examples::require(run.push(simaai::neat::TensorList{tensor_of(rgb)}),
                           "Insight video push failed");
  }

  void close() {
    run.close();
  }
};

// ----------------------------------------------------------------------- postprocess

/// One connected region of the anomaly map, in map pixels.
struct Region {
  cv::Rect bbox;
  double score = 0.0; // highest probability inside the region
};

/// Regions above the threshold that cover at least min_region_px, and their mask.
std::vector<Region> regions_from_map(const cv::Mat& anomaly_map, double threshold,
                                     int min_region_px, cv::Mat& region_mask) {
  const cv::Mat above = anomaly_map >= threshold;
  cv::Mat labels, stats, centroids;
  const int count = cv::connectedComponentsWithStats(above, labels, stats, centroids, 8);
  std::vector<Region> regions;
  region_mask = cv::Mat::zeros(anomaly_map.size(), CV_8UC1);
  for (int index = 1; index < count; ++index) {
    const int* stat = stats.ptr<int>(index);
    if (stat[cv::CC_STAT_AREA] < min_region_px) {
      continue;
    }
    const cv::Mat member = labels == index;
    Region region;
    region.bbox = cv::Rect(stat[cv::CC_STAT_LEFT], stat[cv::CC_STAT_TOP], stat[cv::CC_STAT_WIDTH],
                           stat[cv::CC_STAT_HEIGHT]);
    cv::minMaxLoc(anomaly_map, nullptr, &region.score, nullptr, nullptr, member);
    region_mask.setTo(1, member);
    regions.push_back(region);
  }
  return regions;
}

/// The frame tensor of a pulled sample, bare or in a labelled field.
simaai::neat::Tensor frame_tensor(const simaai::neat::Sample& sample) {
  for (const simaai::neat::Sample& field : sample.fields) {
    if (field.stream_label == "frame") {
      return simaai::neat::tensors_from_sample(field).front();
    }
  }
  return simaai::neat::tensors_from_sample(sample).front();
}

/// The model output as a map_side x map_side float matrix.
cv::Mat map_of(const simaai::neat::Tensor& tensor, int map_side) {
  cv::Mat map(map_side, map_side, CV_32FC1);
  const std::size_t bytes = map.total() * sizeof(float);
  sima_examples::require(tensor.dtype == simaai::neat::TensorDType::Float32 &&
                             tensor.dense_bytes_tight() == bytes &&
                             tensor.copy_dense_bytes_tight_to(map.data, bytes),
                         "model must return a Float32 map_side x map_side anomaly map");
  return map;
}

/// Blend the heatmap over the defect regions, box them and add the verdict banner.
void render(cv::Mat& frame, const cv::Mat& anomaly_map, const std::vector<Region>& regions,
            const cv::Mat& region_mask, const Config& cfg, double fps) {
  const int width = frame.cols;
  const int height = frame.rows;
  if (!regions.empty()) {
    // Fixed display span, so the overlay brightness does not flicker between frames.
    cv::Mat intensity;
    cv::min(cv::max((anomaly_map - cfg.threshold) / (cfg.heat_max - cfg.threshold), 0.0), 1.0,
            intensity);
    cv::Mat scaled;
    intensity.convertTo(scaled, CV_8UC1, 255.0);
    cv::resize(scaled, scaled, cv::Size(width, height));
    cv::Mat heat;
    cv::applyColorMap(scaled, heat, cv::COLORMAP_JET);
    cv::Mat blended;
    cv::addWeighted(frame, 1.0 - cfg.alpha, heat, cfg.alpha, 0.0, blended);
    cv::Mat mask;
    cv::resize(region_mask, mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::copyTo(blended, frame, mask);
    const double sx = static_cast<double>(width) / anomaly_map.cols;
    const double sy = static_cast<double>(height) / anomaly_map.rows;
    for (const Region& region : regions) {
      const cv::Rect& box = region.bbox;
      cv::rectangle(frame, cv::Point(static_cast<int>(box.x * sx), static_cast<int>(box.y * sy)),
                    cv::Point(static_cast<int>((box.x + box.width) * sx),
                              static_cast<int>((box.y + box.height) * sy)),
                    cv::Scalar(0, 0, 255), 2);
    }
  }
  const bool anomaly = !regions.empty();
  const cv::Scalar color = anomaly ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0);
  double top_score = 0.0;
  cv::minMaxLoc(anomaly_map, nullptr, &top_score);
  cv::rectangle(frame, cv::Point(0, 0), cv::Point(width, 34), cv::Scalar(0, 0, 0), -1);
  cv::putText(frame,
              cv::format("%s  score=%.3f  regions=%d", anomaly ? "ANOMALY" : "OK", top_score,
                         static_cast<int>(regions.size())),
              cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
  cv::putText(frame, cv::format("%5.1f fps", fps), cv::Point(width - 130, 24),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

// --------------------------------------------------------------------------------- run

void run(const Config& cfg) {
  const sima_examples::RtspStreamInfo stream = probe_stream(cfg);
  if (stream.width != stream.height) {
    std::cerr << "[warn] stream is " << stream.width << "x" << stream.height
              << "; the model letterboxes non-square frames, which flattens the map\n";
  }
  Detector detector = build_model(cfg, stream.width, stream.height);
  InsightVideo video(cfg, stream.width, stream.height, stream.fps);
  // Last: the source starts reading at once, and an RTSP server drops a reader that stalls.
  Source source = build_source(cfg, stream.width, stream.height, stream.fps);
  std::cout << "rtsp=" << cfg.rtsp_url << " stream=" << stream.width << "x" << stream.height << "@"
            << stream.fps << " map=" << detector.map_side << "x" << detector.map_side
            << " threshold=" << cfg.threshold << " min_region_px=" << cfg.min_region_px
            << " insight=" << cfg.insight_host << " video=" << video.port << " channel=0"
            << std::endl;

  int processed = 0;
  int flagged = 0;
  int window_regions = 0;
  double window_start_ms = sima_examples::time_ms();
  std::deque<double> stamps; // on-screen frame rate window
  g_stop_requested = 0;
  const auto previous_sigint = std::signal(SIGINT, request_stop);
  const auto finish = [&] {
    std::signal(SIGINT, previous_sigint);
    source.run.close();
    video.close();
    std::cout << "processed=" << processed << " flagged=" << flagged
              << " video_sender=" << cfg.insight_host << ":" << video.port << std::endl;
  };
  try {
    while (g_stop_requested == 0 && (cfg.frames <= 0 || processed < cfg.frames)) {
      std::optional<simaai::neat::Sample> sample = source.run.pull("frame", kPullTimeoutMs);
      if (!sample) {
        std::cerr << "[warn] timed out waiting for a frame\n";
        continue;
      }
      // Release the sample right away: the decoder's small buffer pool stalls otherwise.
      cv::Mat frame;
      std::string err;
      sima_examples::require(sima_examples::nv12_to_bgr(frame_tensor(*sample), frame, err), err);
      sample.reset();
      const cv::Mat anomaly_map =
          map_of(detector.model->run(std::vector<cv::Mat>{frame}, kPullTimeoutMs).front(),
                 detector.map_side);
      cv::Mat region_mask;
      const std::vector<Region> regions =
          regions_from_map(anomaly_map, cfg.threshold, cfg.min_region_px, region_mask);

      stamps.push_back(sima_examples::time_ms());
      if (stamps.size() > kFpsWindow) {
        stamps.pop_front();
      }
      const double span_ms = stamps.back() - stamps.front();
      const double live_fps =
          stamps.size() > 1 && span_ms > 0.0 ? (stamps.size() - 1) * 1000.0 / span_ms : 0.0;
      render(frame, anomaly_map, regions, region_mask, cfg, live_fps);
      video.send(frame);

      ++processed;
      flagged += regions.empty() ? 0 : 1;
      window_regions += static_cast<int>(regions.size());
      if (!cfg.save_dir.empty() && cfg.save_every != 0 && processed % cfg.save_every == 0) {
        cv::imwrite(
            (fs::path(cfg.save_dir) / ("frame_" + std::to_string(processed) + ".jpg")).string(),
            frame);
      }
      if (cfg.profile && cfg.profile_interval > 0 && processed % cfg.profile_interval == 0) {
        const double elapsed_s = (sima_examples::time_ms() - window_start_ms) / 1000.0;
        std::cout << cv::format("[profile] frames=%d output_fps=%.1f avg_regions=%.2f",
                                cfg.profile_interval, cfg.profile_interval / elapsed_s,
                                static_cast<double>(window_regions) / cfg.profile_interval)
                  << std::endl;
        window_start_ms = sima_examples::time_ms();
        window_regions = 0;
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
    bool validate_config_only = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config") {
        sima_examples::require(i + 1 < argc, "--config requires a path");
        config_path = argv[++i];
      } else if (arg == "--validate-config-only") {
        validate_config_only = true;
      } else if (arg == "--help" || arg == "-h") {
        std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }
    const Config cfg = load_config(config_path);
    if (validate_config_only) {
      std::cout << "Config validated: " << config_path.string() << "\n";
      return 0;
    }
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
