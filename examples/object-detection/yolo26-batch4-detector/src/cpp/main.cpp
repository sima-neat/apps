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
//
// Batch-4 YOLO26 detector: four Insight RTSP streams, one MLA dispatch, CPU decode.
//
// Up to four RTSP streams are decoded, one frame is taken from each, and all
// four are submitted to the MLA as a single [4, 640, 640, 3] batch. The model
// returns the six YOLO26 heads, which are decoded on the CPU (A65) per batch
// lane, so every stream gets its own detections. The source H.264 is forwarded
// to Insight and detections are timestamped to the exact analysed frame.
//
// Why the batch is assembled here
//   Model-managed preprocessing is a single-image convenience path; it rejects
//   a multi-image submission outright. A batched pack therefore has to be fed
//   one already-assembled [N, 640, 640, 3] tensor, which is what the prefetcher
//   below builds.
//
// Why the decode runs on the CPU
//   The model-managed box decode (decode_type=YoloV26) is not batch-aware. It
//   collapses the six heads into one BBOX payload and decodes lane 0 only,
//   silently discarding lanes 1-3 — no error, no warning. Using it here would
//   report one stream's detections on all four channels. Decoding the raw heads
//   per lane is what keeps detections attributed to the frame they came from.
//
// Heads are matched by shape, not by output order
//   A bbox head carries 4 channels and a class head 80, and the grid size (80,
//   40, 20) orders the three levels. Nothing depends on the order in which the
//   compiler happens to emit the outputs.

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/object_detection/obj_detection_utils.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>
#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

// The YOLO26 head set: three detection levels, each a bbox head and a class
// head. Channel count tells the two apart, grid size orders the levels.
constexpr int kBboxChannels = 4;
constexpr int kClassChannels = 80;
constexpr int kLevels = 3;
constexpr int kExpectedOutputs = 2 * kLevels;
constexpr int kMaxStreams = 4;
constexpr int kPadValue = 114;

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  std::vector<std::string> rtsp_urls;
  bool tcp = true;
  int latency_ms = 100;
  int frames = 0;
  double score_threshold = 0.35;
  int max_detections = 100;
  std::string insight_host;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  fs::path debug_dir;
  int save_every = 0;
  int timeout_ms = 20000;
  int warmup_frames = 10;
  bool profile = false;
  int profile_interval = 50;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

// One dispatch carries one frame per stream, so each stream sees the dispatch
// rate and the aggregate frame rate is that times the stream count.
struct ProfileWindow {
  bool enabled = false;
  int streams = 1;
  int interval = 50;
  int dispatches = 0;
  int detections = 0;
  double start_ms = 0.0;
  double batch_ms = 0.0;
  double infer_ms = 0.0;
  double decode_ms = 0.0;
  double publish_ms = 0.0;

  void add(double batch, double infer, double decode, double publish, int found) {
    if (!enabled)
      return;
    if (dispatches == 0)
      start_ms = sima_examples::time_ms();
    ++dispatches;
    batch_ms += batch;
    infer_ms += infer;
    decode_ms += decode;
    publish_ms += publish;
    detections += found;
    if (dispatches >= interval)
      flush();
  }

  void flush() {
    if (!enabled || dispatches == 0)
      return;
    const double elapsed = std::max(sima_examples::time_ms() - start_ms, 1e-6);
    const double dispatch_fps = static_cast<double>(dispatches) * 1000.0 / elapsed;
    const auto avg = [this](double value) { return value / static_cast<double>(dispatches); };
    std::cout << "[profile] dispatch_fps=" << dispatch_fps << " per_stream_fps=" << dispatch_fps
              << " aggregate_fps=" << dispatch_fps * static_cast<double>(streams)
              << " | avg_batch_ms=" << avg(batch_ms) << " avg_infer_ms=" << avg(infer_ms)
              << " avg_decode_ms=" << avg(decode_ms) << " avg_publish_ms=" << avg(publish_ms)
              << " | avg_detections="
              << static_cast<double>(detections) / static_cast<double>(dispatches) << "\n";
    dispatches = 0;
    detections = 0;
    start_ms = 0.0;
    batch_ms = 0.0;
    infer_ms = 0.0;
    decode_ms = 0.0;
    publish_ms = 0.0;
  }
};

struct StreamRuntime {
  int index = 0;
  std::string url;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  int frame_w = 0;
  int frame_h = 0;
  int fps = 0;
  int video_port = 0;
  int processed = 0;
};

// Letterbox geometry needed to map boxes back onto the source frame.
struct Geometry {
  double scale = 1.0;
  int dx = 0;
  int dy = 0;
};

// One reusable batch: the MLA input plus what the lanes came from. Two of these
// alternate so the next batch can be pulled and letterboxed while the current
// one is still being inferred and decoded.
struct BatchSlot {
  std::vector<float> data;
  std::vector<cv::Mat> frames;
  std::vector<Geometry> geometry;
  std::vector<simaai::neat::Sample> samples;
  bool ready = false;
};

// --------------------------------------------------------------------------
// configuration
// --------------------------------------------------------------------------
CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

std::string strip_inline_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  std::string out;
  out.reserve(line.size());
  for (char c : line) {
    if (c == '\'' && !in_double) {
      in_single = !in_single;
    } else if (c == '"' && !in_single) {
      in_double = !in_double;
    } else if (c == '#' && !in_single && !in_double) {
      break;
    }
    out.push_back(c);
  }
  return out;
}

std::string unquote(std::string value) {
  value = sima_examples::trim_copy(value);
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

// ScalarConfig handles dotted scalar keys; the stream list needs its own pass.
std::vector<std::string> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<std::string> streams;
  bool in_streams = false;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(without_comment).empty()) {
      continue;
    }

    int indent = 0;
    while (indent < static_cast<int>(without_comment.size()) &&
           (without_comment[static_cast<std::size_t>(indent)] == ' ' ||
            without_comment[static_cast<std::size_t>(indent)] == '\t')) {
      ++indent;
    }
    const std::string line = sima_examples::trim_copy(without_comment);

    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams && line == "streams:") {
      in_streams = true;
      streams_indent = indent;
      continue;
    }
    if (in_streams && line.rfind("- ", 0) == 0) {
      const std::string value = unquote(line.substr(2));
      if (value.empty()) {
        throw std::runtime_error("streams entries must be non-empty strings");
      }
      streams.push_back(value);
    }
  }
  if (streams.empty()) {
    throw std::runtime_error("streams must list at least one RTSP URL");
  }
  return streams;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.rtsp_urls.empty(), "streams must list at least one RTSP URL");
  sima_examples::require(cfg.rtsp_urls.size() <= kMaxStreams,
                         "this example supports up to 4 streams");
  for (const auto& url : cfg.rtsp_urls) {
    sima_examples::require(!url.empty() && url.front() != '<',
                           "streams still contains a placeholder URL");
  }
  sima_examples::require(!cfg.insight_host.empty() && cfg.insight_host.front() != '<',
                         "output.insight.host must be set");
  sima_examples::require(cfg.score_threshold > 0.0 && cfg.score_threshold < 1.0,
                         "inference.score_threshold must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.timeout_ms > 0, "runtime.timeout_ms must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.profile_interval > 0, "runtime.profile_interval must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.score_threshold = raw.double_or("inference.score_threshold", 0.35);
  cfg.max_detections = raw.int_or("inference.max_detections", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.debug_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 20000);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 10);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 50);
  validate_config(cfg);
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

// --------------------------------------------------------------------------
// preprocessing
// --------------------------------------------------------------------------
// Aspect-preserving resize into net x net, pad 114, centered. The scaled pixels
// are written straight into the batch lane as normalised RGB float32, so the
// batch buffer is filled in place instead of being stacked from four freshly
// allocated frames.
Geometry letterbox_into(const cv::Mat& bgr, int net, float* dst) {
  const double scale =
      std::min(static_cast<double>(net) / bgr.cols, static_cast<double>(net) / bgr.rows);
  const int new_w = static_cast<int>(std::lround(bgr.cols * scale));
  const int new_h = static_cast<int>(std::lround(bgr.rows * scale));
  const int dx = (net - new_w) / 2;
  const int dy = (net - new_h) / 2;

  cv::Mat lane(net, net, CV_32FC3, dst);
  lane.setTo(cv::Scalar::all(static_cast<double>(kPadValue) / 255.0));

  // Resize and colour-convert at the scaled size, not at the full canvas size.
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
  cv::Mat roi = lane(cv::Rect(dx, dy, new_w, new_h));
  resized.convertTo(roi, CV_32FC3, 1.0 / 255.0);

  return Geometry{scale, dx, dy};
}

// --------------------------------------------------------------------------
// six model outputs -> heads -> detections (all on the CPU)
// --------------------------------------------------------------------------
// A single head plane for one lane: a dense [H, W, C] view over the output.
struct HeadPlane {
  const float* data = nullptr;
  int grid = 0;
  int channels = 0;

  const float* cell(int row, int col) const {
    return data + (static_cast<std::size_t>(row) * static_cast<std::size_t>(grid) +
                   static_cast<std::size_t>(col)) *
                      static_cast<std::size_t>(channels);
  }
};

struct LaneHeads {
  std::array<HeadPlane, kLevels> bbox;
  std::array<HeadPlane, kLevels> cls;
};

// One model output, copied to the host. The MLA writes its heads into device
// memory that `Tensor::data_ptr` cannot address, so each dispatch materialises
// them once here and all lanes then read the same buffers.
struct HeadTensor {
  std::vector<float> data;
  std::vector<int64_t> shape;
};

// Six [N,H,W,C] outputs -> the head planes of one lane, sorted by descending
// grid so level 0 is the 80x80 map, matching the strides decode_lane derives.
LaneHeads heads_from_outputs(const std::vector<HeadTensor>& outputs, int lane) {
  std::vector<HeadPlane> bbox;
  std::vector<HeadPlane> cls;
  for (const auto& tensor : outputs) {
    if (tensor.shape.size() != 4) {
      throw std::runtime_error("expected [N,H,W,C] outputs, got rank " +
                               std::to_string(tensor.shape.size()));
    }
    const int batch = static_cast<int>(tensor.shape[0]);
    if (lane >= batch) {
      throw std::runtime_error("lane " + std::to_string(lane) + " out of range for batch " +
                               std::to_string(batch));
    }
    const int grid = static_cast<int>(tensor.shape[1]);
    const int channels = static_cast<int>(tensor.shape[3]);
    const std::size_t lane_stride =
        static_cast<std::size_t>(grid) * static_cast<std::size_t>(tensor.shape[2]) *
        static_cast<std::size_t>(channels);
    if (tensor.data.size() < (static_cast<std::size_t>(lane) + 1) * lane_stride) {
      throw std::runtime_error("output smaller than its declared shape");
    }

    HeadPlane plane;
    plane.data = tensor.data.data() + static_cast<std::size_t>(lane) * lane_stride;
    plane.grid = grid;
    plane.channels = channels;

    if (channels == kBboxChannels) {
      bbox.push_back(plane);
    } else if (channels == kClassChannels) {
      cls.push_back(plane);
    } else {
      throw std::runtime_error("output with " + std::to_string(channels) +
                               " channels is neither a bbox head nor a class head");
    }
  }
  if (bbox.size() != kLevels || cls.size() != kLevels) {
    throw std::runtime_error("expected 3 bbox and 3 class heads, got " +
                             std::to_string(bbox.size()) + " and " + std::to_string(cls.size()));
  }

  const auto by_descending_grid = [](const HeadPlane& a, const HeadPlane& b) {
    return a.grid > b.grid;
  };
  std::sort(bbox.begin(), bbox.end(), by_descending_grid);
  std::sort(cls.begin(), cls.end(), by_descending_grid);

  LaneHeads heads;
  for (int level = 0; level < kLevels; ++level) {
    if (bbox[static_cast<std::size_t>(level)].grid != cls[static_cast<std::size_t>(level)].grid) {
      throw std::runtime_error("bbox and class grids disagree at level " + std::to_string(level));
    }
    heads.bbox[static_cast<std::size_t>(level)] = bbox[static_cast<std::size_t>(level)];
    heads.cls[static_cast<std::size_t>(level)] = cls[static_cast<std::size_t>(level)];
  }
  return heads;
}

// YOLO26 reg_max=1: raw l/t/r/b distances plus class logits. No NMS — the
// one-to-one head is already deduplicated, so detections are only ranked and
// capped.
//
// The class heads hold 672 000 logits per frame, so the decode is dominated by
// whatever runs across all of them. Sigmoid is monotonic, which means a cell
// clears a probability threshold exactly when its logit clears the matching
// logit threshold. Thresholding the raw logits keeps the exponential and the
// per-cell argmax on the handful of surviving cells instead of the whole tensor.
std::vector<objdet::Box> decode_lane(const LaneHeads& heads, int net, double score_threshold,
                                     int max_detections) {
  const float logit_threshold =
      static_cast<float>(std::log(score_threshold / (1.0 - score_threshold)));
  std::vector<objdet::Box> detections;

  for (int level = 0; level < kLevels; ++level) {
    const HeadPlane& box = heads.bbox[static_cast<std::size_t>(level)];
    const HeadPlane& cls = heads.cls[static_cast<std::size_t>(level)];
    const float stride = static_cast<float>(net) / static_cast<float>(box.grid);

    for (int row = 0; row < cls.grid; ++row) {
      for (int col = 0; col < cls.grid; ++col) {
        const float* logits = cls.cell(row, col);
        int best_class = 0;
        float best_logit = logits[0];
        for (int c = 1; c < cls.channels; ++c) {
          if (logits[c] > best_logit) {
            best_logit = logits[c];
            best_class = c;
          }
        }
        if (best_logit < logit_threshold) {
          continue;
        }

        const float* ltrb = box.cell(row, col);
        const float anchor_x = static_cast<float>(col) + 0.5f;
        const float anchor_y = static_cast<float>(row) + 0.5f;

        objdet::Box det;
        det.score = 1.0f / (1.0f + std::exp(-std::clamp(best_logit, -30.0f, 30.0f)));
        det.class_id = best_class;
        det.x1 = (anchor_x - ltrb[0]) * stride;
        det.y1 = (anchor_y - ltrb[1]) * stride;
        det.x2 = (anchor_x + ltrb[2]) * stride;
        det.y2 = (anchor_y + ltrb[3]) * stride;
        detections.push_back(det);
      }
    }
  }

  std::sort(detections.begin(), detections.end(),
            [](const objdet::Box& a, const objdet::Box& b) { return a.score > b.score; });
  if (static_cast<int>(detections.size()) > max_detections) {
    detections.resize(static_cast<std::size_t>(max_detections));
  }
  return detections;
}

// Undo the letterbox so boxes are in source-frame coordinates.
std::vector<objdet::Box> to_original(const std::vector<objdet::Box>& detections,
                                     const Geometry& geometry, int width, int height) {
  std::vector<objdet::Box> mapped;
  mapped.reserve(detections.size());
  const auto map = [&geometry](float value, int offset, int limit) {
    const double out = (static_cast<double>(value) - offset) / geometry.scale;
    return static_cast<float>(std::clamp(out, 0.0, static_cast<double>(limit) - 1.0));
  };
  for (const auto& det : detections) {
    objdet::Box box = det;
    box.x1 = map(det.x1, geometry.dx, width);
    box.y1 = map(det.y1, geometry.dy, height);
    box.x2 = map(det.x2, geometry.dx, width);
    box.y2 = map(det.y2, geometry.dy, height);
    if (box.x2 > box.x1 && box.y2 > box.y1) {
      mapped.push_back(box);
    }
  }
  return mapped;
}

// --------------------------------------------------------------------------
// runtime plumbing
// --------------------------------------------------------------------------
// RTSP source, depayload, parse and hardware H.264 decode to NV12, in one
// group. The frames leave here and the batch is assembled by the app.
simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_options(const AppConfig& cfg, const std::string& url, int& width, int& height,
                     int& fps) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  (void)sima_examples::probe_rtsp_stream_info(url, probe_options, probe);
  sima_examples::require(probe.width > 0 && probe.height > 0,
                         "failed to probe RTSP frame size: " + url);
  width = probe.width;
  height = probe.height;
  fps = probe.fps > 0 ? probe.fps : 30;

  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = simaai::neat::FormatTag::NV12;
  opt.decoder_name = "decoder_" + std::to_string(width) + "x" + std::to_string(height);
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.fallback_h264_width = width;
  opt.fallback_h264_height = height;
  opt.fallback_h264_fps = fps;
  opt.source_fps = fps;
  opt.output_caps.enable = true;
  opt.output_caps.format = simaai::neat::FormatTag::NV12;
  opt.output_caps.width = width;
  opt.output_caps.height = height;
  opt.output_caps.fps = fps;
  opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  return opt;
}

simaai::neat::InputOptions h264_decode_input_options() {
  simaai::neat::InputOptions options;
  options.payload_type = simaai::neat::PayloadType::Encoded;
  options.format = simaai::neat::FormatTag::H264;
  options.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  return options;
}

simaai::neat::InputOptions h264_video_input_options() {
  auto options = h264_decode_input_options();
  options.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  return options;
}

simaai::neat::Graph build_encoded_source_graph(
    int index, const simaai::neat::nodes::groups::RtspDecodedInputOptions& options) {
  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded;
  encoded.url = options.url;
  encoded.codec = simaai::neat::nodes::groups::RtspCodec::H264;
  encoded.latency_ms = options.latency_ms;
  encoded.tcp = options.tcp;
  encoded.source_fps = options.source_fps;
  encoded.fallback_h264_width = options.fallback_h264_width;
  encoded.fallback_h264_height = options.fallback_h264_height;

  simaai::neat::Graph source("rtsp_source_" + std::to_string(index));
  source.add(simaai::neat::nodes::groups::RtspEncodedInput(encoded));
  return source;
}

simaai::neat::Graph build_decode_graph(
    int index, const simaai::neat::nodes::groups::RtspDecodedInputOptions& options) {
  simaai::neat::SimaDecodeOptions decode_options;
  decode_options.type = simaai::neat::SimaDecodeType::H264;
  decode_options.sima_allocator_type = options.sima_allocator_type;
  decode_options.out_format = options.out_format;
  decode_options.decoder_name = options.decoder_name;
  decode_options.raw_output = options.decoder_raw_output;
  decode_options.dec_width = options.fallback_h264_width;
  decode_options.dec_height = options.fallback_h264_height;
  decode_options.dec_fps = options.source_fps;

  simaai::neat::Graph decode("decode_" + std::to_string(index));
  decode.connect(simaai::neat::nodes::Input("decode_h264", h264_decode_input_options()),
                 simaai::neat::nodes::SimaDecode(decode_options));
  decode.add(simaai::neat::nodes::CapsRaw(
      "NV12", options.output_caps.width, options.output_caps.height, options.output_caps.fps,
      options.output_caps.memory));
  decode.add(simaai::neat::nodes::Output("frame_" + std::to_string(index),
                                         simaai::neat::OutputOptions::Latest()));
  return decode;
}

simaai::neat::Graph build_video_sender_graph(
    int index, const simaai::neat::nodes::groups::VideoSenderOptions& options) {
  simaai::neat::Graph video("video_sender_" + std::to_string(index));
  video.connect(simaai::neat::nodes::Input("video_h264", h264_video_input_options()),
                simaai::neat::nodes::groups::VideoSender(options));
  return video;
}

simaai::neat::GraphLinkOptions realtime_link(int index, int queue_depth = 3) {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.queue_depth = queue_depth;
  link.stream_id = "stream" + std::to_string(index);
  return link;
}

simaai::neat::RunOptions build_run_options() {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 4;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

// Block for one frame, then drain so the caller always gets the newest.
bool pull_latest(simaai::neat::Run& run, const std::string& output_name, int timeout_ms,
                 simaai::neat::Sample& out) {
  simaai::neat::PullError pull_error;
  auto status = run.pull(output_name, timeout_ms, out, &pull_error);
  if (status == simaai::neat::PullStatus::Timeout ||
      status == simaai::neat::PullStatus::Closed) {
    return false;
  }
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to pull " + output_name + ": " + pull_error.message);
  }
  for (;;) {
    simaai::neat::Sample newer;
    status = run.pull(output_name, 0, newer, &pull_error);
    if (status != simaai::neat::PullStatus::Ok) {
      return true;
    }
    out = std::move(newer);
  }
}

// Keeps one batch in flight so ingest overlaps inference.
//
// A frame pull blocks until the source produces its next picture — about 33 ms
// at 30 fps — and the letterbox costs a few ms more per lane. Doing that
// between dispatches would serialize it with the MLA and the decode. Instead
// the lanes for the next batch are filled on worker threads while the caller
// works on the current one.
class BatchPrefetcher {
public:
  BatchPrefetcher(simaai::neat::Run& run, std::vector<StreamRuntime>& streams, int net,
                  int batch_size, int timeout_ms)
      : run_(run), streams_(streams), net_(net), batch_size_(batch_size), timeout_ms_(timeout_ms) {
    const std::size_t lane_floats =
        static_cast<std::size_t>(net) * static_cast<std::size_t>(net) * 3U;
    for (auto& slot : slots_) {
      slot.data.assign(lane_floats * static_cast<std::size_t>(batch_size), 0.0f);
      slot.frames.resize(streams.size());
      slot.geometry.resize(streams.size());
      slot.samples.resize(streams.size());
    }
    submit(0);
  }

  // Hand back the filled batch and immediately start filling the other.
  BatchSlot* next() {
    // get(), unlike wait(), rethrows failures from fill() and its lane tasks.
    pending_.get();
    BatchSlot* ready = &slots_[static_cast<std::size_t>(slot_)];
    slot_ ^= 1;
    submit(slot_);
    return ready->ready ? ready : nullptr;
  }

  // The in-flight fill has lane threads sitting inside run.pull, so it has to
  // finish before the caller tears the Run down.
  void close() {
    closed_ = true;
    if (pending_.valid()) {
      pending_.get();
    }
  }

private:
  void submit(int slot_index) {
    // Never leave a previously completed slot publishable if the refill fails.
    slots_[static_cast<std::size_t>(slot_index)].ready = false;
    pending_ = std::async(std::launch::async, [this, slot_index] {
      fill(slots_[static_cast<std::size_t>(slot_index)]);
    });
  }

  void fill(BatchSlot& slot) {
    const std::size_t lane_floats =
        static_cast<std::size_t>(net_) * static_cast<std::size_t>(net_) * 3U;

    std::vector<std::future<bool>> lanes;
    lanes.reserve(streams_.size());
    for (auto& stream : streams_) {
      lanes.push_back(std::async(std::launch::async, [this, &slot, &stream, lane_floats] {
        return fill_lane(slot, stream, lane_floats);
      }));
    }
    bool ok = true;
    for (auto& lane : lanes) {
      ok = lane.get() && ok;
    }
    slot.ready = ok;

    // Fewer streams than the compiled batch: repeat the last lane so the MLA
    // still runs exactly one dispatch.
    for (std::size_t lane = streams_.size(); lane < static_cast<std::size_t>(batch_size_); ++lane) {
      std::copy_n(slot.data.data() + (streams_.size() - 1) * lane_floats, lane_floats,
                  slot.data.data() + lane * lane_floats);
    }
  }

  bool fill_lane(BatchSlot& slot, StreamRuntime& stream, std::size_t lane_floats) {
    if (closed_) {
      return false;
    }
    const auto lane = static_cast<std::size_t>(stream.index);
    simaai::neat::Sample sample;
    if (!pull_latest(run_, "frame_" + std::to_string(stream.index), timeout_ms_, sample)) {
      return false;
    }
    const auto tensors = simaai::neat::tensors_from_sample(sample, false);
    if (tensors.empty()) {
      return false;
    }
    cv::Mat bgr;
    std::string err;
    if (!sima_examples::nv12_to_bgr(tensors.front(), bgr, err)) {
      throw std::runtime_error("failed to convert decoded frame: " + err);
    }
    slot.geometry[lane] = letterbox_into(bgr, net_, slot.data.data() + lane * lane_floats);
    slot.frames[lane] = std::move(bgr);
    slot.samples[lane] = std::move(sample);
    return true;
  }

  simaai::neat::Run& run_;
  std::vector<StreamRuntime>& streams_;
  int net_ = 0;
  int batch_size_ = 0;
  int timeout_ms_ = 0;
  std::array<BatchSlot, 2> slots_;
  int slot_ = 0;
  std::future<void> pending_;
  std::atomic<bool> closed_{false};
};

// --------------------------------------------------------------------------
// output
// --------------------------------------------------------------------------
void send_metadata(StreamRuntime& stream, const simaai::neat::Sample& sample,
                   const std::vector<objdet::Box>& boxes,
                   const std::vector<std::string>& labels) {
  std::vector<sima_examples::MetadataBox> metadata_boxes;
  metadata_boxes.reserve(boxes.size());
  int object_index = 1;
  for (const auto& box : boxes) {
    sima_examples::MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = (box.class_id >= 0 && box.class_id < static_cast<int>(labels.size()))
                    ? labels[static_cast<std::size_t>(box.class_id)]
                    : "unknown";
    obj.confidence = box.score;
    obj.x = box.x1;
    obj.y = box.y1;
    obj.w = box.x2 - box.x1;
    obj.h = box.y2 - box.y1;
    metadata_boxes.push_back(obj);
  }
  const std::string data_json = sima_examples::metadata_boxes_data_json("objects", metadata_boxes);
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  nlohmann::json message;
  message["type"] = "object-detection";
  message["data"] = nlohmann::json::parse(data_json);
  message["timestamp"] = timestamp_ms;
  message["frame_id"] = frame_id;
  if (sample.pts_ns >= 0) {
    const auto ticks =
        (static_cast<__int128>(sample.pts_ns) * 90'000) / 1'000'000'000;
    message["_insight"]["rtp_timestamp"] =
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(ticks));
  }
  std::string err;
  if (!stream.metadata_sender->send_raw_json(message.dump(), &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
}

void save_debug_frame(const AppConfig& cfg, const StreamRuntime& stream, const cv::Mat& bgr,
                      const std::vector<objdet::Box>& boxes) {
  if (cfg.debug_dir.empty() || cfg.save_every <= 0 || stream.processed % cfg.save_every != 0) {
    return;
  }
  cv::Mat frame = bgr.clone();
  objdet::draw_boxes(frame, boxes, cfg.score_threshold, cv::Scalar(0, 255, 0), "");
  const auto out_path = cfg.debug_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                         std::to_string(stream.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), frame)) {
    std::cerr << "[warn] failed to write " << out_path.string() << "\n";
  }
}

// --------------------------------------------------------------------------
// application
// --------------------------------------------------------------------------
void build_streams(const AppConfig& cfg, simaai::neat::Graph& graph,
                   std::vector<StreamRuntime>& streams) {
  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    StreamRuntime stream;
    stream.index = static_cast<int>(index);
    stream.url = cfg.rtsp_urls[index];
    stream.source_options =
        build_source_options(cfg, stream.url, stream.frame_w, stream.frame_h, stream.fps);

    simaai::neat::MetadataSenderOptions metadata_options;
    metadata_options.host = cfg.insight_host;
    metadata_options.channel = stream.index;
    metadata_options.metadata_port_base = cfg.metadata_port_base;
    std::string metadata_err;
    stream.metadata_sender =
        std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
    sima_examples::require(stream.metadata_sender->ok(), metadata_err);

    auto source = build_encoded_source_graph(stream.index, stream.source_options);
    auto decoder = build_decode_graph(stream.index, stream.source_options);
    if (cfg.video_enabled) {
      auto branch = simaai::neat::graphs::Branch(
          "encoded_" + std::to_string(stream.index), {"decode_h264", "video_h264"});
      graph.connect(source, branch);
      graph.connect(branch, decoder, realtime_link(stream.index));

      auto video_options =
          simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(
              simaai::neat::nodes::groups::RtspCodec::H264);
      video_options.host = cfg.insight_host;
      video_options.channel = stream.index;
      video_options.video_port_base = cfg.video_port_base;
      video_options.async = true;
      stream.video_port = video_options.video_port();
      graph.connect(branch, build_video_sender_graph(stream.index, video_options),
                    realtime_link(stream.index));
    } else {
      graph.connect(source, decoder, realtime_link(stream.index));
    }

    std::cout << "[stream " << stream.index << "] rtsp=" << stream.url << " " << stream.frame_w
              << "x" << stream.frame_h << "@" << stream.fps << " video=";
    if (cfg.video_enabled) {
      std::cout << stream.video_port;
    } else {
      std::cout << "disabled";
    }
    std::cout << " metadata=" << stream.metadata_sender->metadata_port() << "\n";
    streams.push_back(std::move(stream));
  }
}

bool all_streams_done(const std::vector<StreamRuntime>& streams, int frame_limit) {
  if (frame_limit <= 0) {
    return false;
  }
  return std::all_of(streams.begin(), streams.end(), [frame_limit](const StreamRuntime& stream) {
    return stream.processed >= frame_limit;
  });
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);
  if (!cfg.debug_dir.empty()) {
    fs::create_directories(cfg.debug_dir);
  }

  const auto labels = load_labels(cfg.labels_path);
  simaai::neat::Model model(cfg.model_path);

  const auto input_specs = model.input_specs();
  sima_examples::require(!input_specs.empty(), "model exposes no inputs");
  const auto& input_shape = input_specs.front().shape;
  if (input_shape.size() != 4) {
    throw std::runtime_error("this example needs a batched model; compile with --batch_size 4");
  }
  const int batch_size = static_cast<int>(input_shape[0]);
  const int net = static_cast<int>(input_shape[1]);
  if (batch_size != kMaxStreams) {
    throw std::runtime_error("model batch size is " + std::to_string(batch_size) +
                             "; this example requires batch size " +
                             std::to_string(kMaxStreams));
  }

  const auto output_specs = model.output_specs();
  if (static_cast<int>(output_specs.size()) != kExpectedOutputs) {
    throw std::runtime_error("expected " + std::to_string(kExpectedOutputs) +
                             " YOLO26 head tensors, model has " +
                             std::to_string(output_specs.size()));
  }
  std::cout << "Model loaded: batch=" << batch_size << " net=" << net
            << " heads=" << output_specs.size() << "\n";

  simaai::neat::Graph graph;
  std::vector<StreamRuntime> streams;
  streams.reserve(cfg.rtsp_urls.size());
  build_streams(cfg, graph, streams);
  auto run = graph.build(build_run_options());

  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.streams = static_cast<int>(streams.size());
  profile.interval = cfg.profile_interval;

  BatchPrefetcher prefetcher(run, streams, net, batch_size, cfg.timeout_ms);

  while (g_stop_requested == 0 && !all_streams_done(streams, cfg.frames)) {
    const double start = sima_examples::time_ms();
    BatchSlot* batch = prefetcher.next();
    if (batch == nullptr) {
      const auto last_error = run.last_error();
      if (!last_error.empty()) {
        throw std::runtime_error("runtime graph error: " + last_error);
      }
      std::cerr << "[warn] timed out waiting for frames\n";
      continue;
    }
    const double batched = sima_examples::time_ms();

    // One dispatch for all lanes.
    const auto input = simaai::neat::Tensor::from_vector(
        batch->data, {batch_size, net, net, 3}, simaai::neat::TensorMemory::EV74);
    const auto outputs = model.run(simaai::neat::TensorList{input}, cfg.timeout_ms);
    if (static_cast<int>(outputs.size()) != kExpectedOutputs) {
      throw std::runtime_error("expected " + std::to_string(kExpectedOutputs) +
                               " head tensors, got " + std::to_string(outputs.size()));
    }
    // Materialise the heads on the host once per dispatch, not once per lane.
    std::vector<HeadTensor> heads;
    heads.reserve(outputs.size());
    for (const auto& tensor : outputs) {
      heads.push_back(HeadTensor{sima_examples::tensor_to_floats(tensor), tensor.shape});
    }
    const double inferred = sima_examples::time_ms();

    // Each lane's heads are an independent read of the same output tensors.
    std::vector<std::future<std::vector<objdet::Box>>> lanes;
    lanes.reserve(streams.size());
    for (const auto& stream : streams) {
      const int index = stream.index;
      const int frame_w = stream.frame_w;
      const int frame_h = stream.frame_h;
      lanes.push_back(std::async(std::launch::async, [&, index, frame_w, frame_h] {
        const auto lane_heads = heads_from_outputs(heads, index);
        auto detections = decode_lane(lane_heads, net, cfg.score_threshold, cfg.max_detections);
        return to_original(detections, batch->geometry[static_cast<std::size_t>(index)], frame_w,
                           frame_h);
      }));
    }
    std::vector<std::vector<objdet::Box>> per_stream;
    per_stream.reserve(lanes.size());
    for (auto& lane : lanes) {
      per_stream.push_back(lane.get());
    }
    const double decoded = sima_examples::time_ms();

    int found = 0;
    for (std::size_t i = 0; i < streams.size(); ++i) {
      auto& stream = streams[i];
      const auto& detections = per_stream[i];
      found += static_cast<int>(detections.size());
      ++stream.processed;
      if (stream.processed <= cfg.warmup_frames) {
        continue;
      }
      const auto lane = static_cast<std::size_t>(stream.index);
      const auto& sample = batch->samples[lane];
      const auto& frame = batch->frames[lane];
      send_metadata(stream, sample, detections, labels);
      save_debug_frame(cfg, stream, frame, detections);
    }
    profile.add(batched - start, inferred - batched, decoded - inferred,
                sima_examples::time_ms() - decoded, found);
  }

  profile.flush();
  prefetcher.close();
  run.close();
  for (auto& stream : streams) {
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
  }
  std::signal(SIGINT, previous_sigint);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions cli = parse_args(argc, argv);
    if (!fs::exists(cli.config_path)) {
      std::cerr << "Error: config file not found: " << cli.config_path << "\n";
      return 2;
    }

    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.rtsp_urls.size()
                << ", score_threshold=" << cfg.score_threshold
                << ", max_detections=" << cfg.max_detections << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
