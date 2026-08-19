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

#include "neat.h"
#include "neat/models.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using sima_examples::time_ms;

namespace {

/// MetadataSender rejects a payload above 65507 bytes, and the rejection surfaces as an error the
/// application has to handle mid-stream. Half of that leaves room for the envelope and keeps the
/// datagram count low enough for Insight to reassemble within its 250 ms window.
constexpr std::size_t kMetadataByteBudget = 32768;

/// One instance in frame pixels: `bbox` is the detection rectangle, `polygon` its silhouette.
struct MetadataSegment {
  std::string id;
  std::string label;
  float confidence = 0.0f;
  cv::Rect bbox;
  std::vector<cv::Point> polygon;
};

struct EncodedSegments {
  std::string data_json;
  int dropped = 0;
};

/// YOLO26 emits masks at one quarter of the model input per dimension, so a 160x160 head
/// corresponds to a 640x640 input.
constexpr int kMaskStride = 4;

/// Mask-head region covering `frame_rect`. The head is a fixed grid over the letterboxed model
/// input, so a frame rectangle reaches it through the same scale and padding the preprocessor used.
cv::Rect mask_rect_for_frame_rect(const cv::Rect& frame_rect, const cv::Size& frame_size,
                                  const cv::Size& mask_size) {
  const int model_w = mask_size.width * kMaskStride;
  const int model_h = mask_size.height * kMaskStride;
  const double scale =
      std::min(static_cast<double>(model_w) / static_cast<double>(frame_size.width),
               static_cast<double>(model_h) / static_cast<double>(frame_size.height));
  const double pad_x =
      (static_cast<double>(model_w) - static_cast<double>(frame_size.width) * scale) * 0.5;
  const double pad_y =
      (static_cast<double>(model_h) - static_cast<double>(frame_size.height) * scale) * 0.5;
  const auto to_mask_x = [&](double frame_x) {
    return (frame_x * scale + pad_x) * static_cast<double>(mask_size.width) /
           static_cast<double>(model_w);
  };
  const auto to_mask_y = [&](double frame_y) {
    return (frame_y * scale + pad_y) * static_cast<double>(mask_size.height) /
           static_cast<double>(model_h);
  };

  const int x0 = std::clamp(static_cast<int>(std::floor(to_mask_x(frame_rect.x))), 0,
                            std::max(0, mask_size.width - 1));
  const int y0 = std::clamp(static_cast<int>(std::floor(to_mask_y(frame_rect.y))), 0,
                            std::max(0, mask_size.height - 1));
  const int x1 = std::clamp(static_cast<int>(std::ceil(to_mask_x(frame_rect.x + frame_rect.width))),
                            x0 + 1, mask_size.width);
  const int y1 =
      std::clamp(static_cast<int>(std::ceil(to_mask_y(frame_rect.y + frame_rect.height))), y0 + 1,
                 mask_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

/// Mask-head region for `frame_rect`, resized to frame pixels.
cv::Mat project_letterbox_mask_roi(const cv::Mat& mask, const cv::Rect& frame_rect,
                                   const cv::Size& frame_size) {
  const cv::Rect mask_rect =
      mask_rect_for_frame_rect(frame_rect, frame_size, cv::Size(mask.cols, mask.rows));
  cv::Mat projected;
  cv::resize(mask(mask_rect), projected, frame_rect.size(), 0, 0, cv::INTER_LINEAR);
  return projected;
}

/// Frame-absolute silhouette of `mask` inside `frame_rect`, empty when the thresholded mask holds
/// nothing Insight can draw. `threshold` is a fraction of full scale, as `output.mask_threshold`
/// is. Upscaling before thresholding is what makes the outline match the rendered overlay.
std::vector<cv::Point> mask_polygon(const cv::Mat& mask, const cv::Rect& frame_rect,
                                    const cv::Size& frame_size, double threshold) {
  cv::Mat binary;
  cv::threshold(project_letterbox_mask_roi(mask, frame_rect, frame_size), binary, threshold * 255.0,
                255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return {};
  }
  const auto& largest =
      *std::max_element(contours.begin(), contours.end(),
                        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                          return cv::contourArea(a) < cv::contourArea(b);
                        });

  std::vector<cv::Point> polygon;
  cv::approxPolyDP(largest, polygon, 0.004 * cv::arcLength(largest, true), true);
  if (polygon.size() < 3) {
    return {};
  }
  // Contour points lie inside frame_rect, which is already clamped to the frame, so shifting them
  // into frame space cannot leave the image.
  for (auto& point : polygon) {
    point += frame_rect.tl();
  }
  return polygon;
}

/// `data` object of a `segmentation` metadata message. Segments that do not fit the byte budget are
/// dropped lowest-confidence first and counted.
EncodedSegments encode_segments(std::vector<MetadataSegment> segments) {
  // Stable, so segments tying on confidence are dropped in the same order the Python
  // implementation drops them.
  std::stable_sort(segments.begin(), segments.end(),
                   [](const MetadataSegment& a, const MetadataSegment& b) {
                     return a.confidence > b.confidence;
                   });

  nlohmann::json entries = nlohmann::json::array();
  std::size_t bytes = sizeof(R"({"segments":[]})") - 1;
  for (const auto& segment : segments) {
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : segment.polygon) {
      points.push_back({point.x, point.y});
    }
    nlohmann::json entry = {
        {"id", segment.id},
        {"label", segment.label},
        {"confidence", segment.confidence},
        {"bbox", {segment.bbox.x, segment.bbox.y, segment.bbox.width, segment.bbox.height}},
        {"mask_format", "polygon"},
        {"mask", std::move(points)},
    };
    const std::size_t entry_bytes = entry.dump().size() + 1;
    if (bytes + entry_bytes > kMetadataByteBudget) {
      break;
    }
    bytes += entry_bytes;
    entries.push_back(std::move(entry));
  }

  const int dropped = static_cast<int>(segments.size() - entries.size());
  return {nlohmann::json{{"segments", std::move(entries)}}.dump(), dropped};
}

enum class SourceType { Rtsp, Http };
enum class SourceCodec { H264, H265, Mjpeg };

struct AppConfig {
  std::string model_path;
  fs::path labels_path;
  std::string source_url;
  SourceType source_type = SourceType::Rtsp;
  SourceCodec source_codec = SourceCodec::H264;
  int latency_ms = 200;
  bool tcp = true;
  int source_fps = 0;
  bool ssl_strict = true;
  int frames = 0;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host = "127.0.0.1";
  int video_port = 9000;
  int metadata_port = 9100;
  fs::path save_dir;
  int save_every = 0;
  double mask_alpha = 0.55;
  double mask_threshold = 0.50;
  bool draw_boxes = true;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct SegmentationDetection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  cv::Mat mask;
};

struct ProfileWindow {
  bool enabled = false;
  int interval = 100;
  int frames = 0;
  int boxes = 0;
  int dropped_segments = 0;
  double start_ms = 0.0;
  double pull_ms = 0.0;
  double decode_ms = 0.0;
  double metadata_ms = 0.0;

  void add(double pull, double decode, double metadata, int box_count, int dropped) {
    if (!enabled) {
      return;
    }
    if (frames == 0) {
      start_ms = time_ms();
    }
    frames += 1;
    boxes += box_count;
    dropped_segments += dropped;
    pull_ms += pull;
    decode_ms += decode;
    metadata_ms += metadata;
    if (frames >= interval) {
      flush();
    }
  }

  void flush() {
    if (!enabled || frames <= 0) {
      return;
    }
    const double elapsed_ms = std::max(time_ms() - start_ms, 1e-6);
    const double n = static_cast<double>(frames);
    const double fps = static_cast<double>(frames) * 1000.0 / elapsed_ms;
    std::cout << "[profile] frames=" << frames << " output_fps=" << fps
              << " avg_pull_ms=" << pull_ms / n << " avg_decode_ms=" << decode_ms / n
              << " avg_metadata_ms=" << metadata_ms / n
              << " avg_instances=" << static_cast<double>(boxes) / n
              << " dropped_segments=" << dropped_segments << "\n";
    reset();
  }

  void reset() {
    frames = 0;
    boxes = 0;
    dropped_segments = 0;
    start_ms = 0.0;
    pull_ms = 0.0;
    decode_ms = 0.0;
    metadata_ms = 0.0;
  }
};

struct PipelineRuntime {
  std::unique_ptr<simaai::neat::Model> model;
  simaai::neat::Graph graph;
  simaai::neat::Run run;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  /// Run output the loop pulls: the segments alone, or the frame-joined bundle when saving.
  std::string output_name;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 30;
  int video_port = 0;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

SourceType parse_source_type(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "rtsp") {
    return SourceType::Rtsp;
  }
  if (lowered == "http" || lowered == "https") {
    return SourceType::Http;
  }
  throw std::runtime_error("source.type must be rtsp or http");
}

SourceCodec parse_source_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return SourceCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return SourceCodec::H265;
  }
  if (lowered == "mjpeg" || lowered == "jpeg") {
    return SourceCodec::Mjpeg;
  }
  throw std::runtime_error("source.codec must be h264/avc, h265/hevc, or mjpeg");
}

const char* source_type_name(SourceType value) {
  return value == SourceType::Rtsp ? "rtsp" : "http";
}

const char* source_codec_name(SourceCodec value) {
  if (value == SourceCodec::H264)
    return "h264";
  return value == SourceCodec::H265 ? "h265" : "mjpeg";
}

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

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.source_url.empty(), "source.url or source.rtsp_url must be set");
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "source.latency_ms must be >= 0");
  sima_examples::require(cfg.source_fps >= 0, "source.fps must be >= 0");
  if (cfg.source_type == SourceType::Http) {
    sima_examples::require(cfg.source_codec == SourceCodec::Mjpeg,
                           "source.codec must be mjpeg for source.type=http");
  }
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.profile_interval > 0, "runtime.profile_interval must be > 0");
  sima_examples::require(cfg.video_port > 0, "output.insight.video_port must be > 0");
  sima_examples::require(cfg.metadata_port > 0, "output.insight.metadata_port must be > 0");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
  sima_examples::require(cfg.mask_alpha >= 0.0 && cfg.mask_alpha <= 1.0,
                         "output.mask_alpha must be between 0 and 1");
  sima_examples::require(cfg.mask_threshold >= 0.0 && cfg.mask_threshold <= 1.0,
                         "output.mask_threshold must be between 0 and 1");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const fs::path default_labels =
      sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() /
      "coco_label.txt";

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  const std::string legacy_rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.source_url = raw.string_or("source.url", legacy_rtsp_url);
  cfg.source_type = parse_source_type(raw.string_or("source.type", "rtsp"));
  cfg.source_codec = parse_source_codec(raw.string_or("source.codec", "h264"));
  cfg.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.source_fps = raw.int_or("source.fps", 0);
  cfg.ssl_strict = raw.bool_or("source.ssl_strict", true);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  cfg.mask_alpha = raw.double_or("output.mask_alpha", 0.55);
  cfg.mask_threshold = raw.double_or("output.mask_threshold", 0.50);
  cfg.draw_boxes = raw.bool_or("output.draw_boxes", true);
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

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("float tensor byte size is not aligned");
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

std::vector<std::uint8_t> tensor_to_u8(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::UInt8) {
    throw std::runtime_error("expected UInt8 tensor");
  }
  return tensor.copy_dense_bytes_tight();
}

std::vector<SegmentationDetection>
decode_segmentation_output(const simaai::neat::TensorList& tensors, int frame_w, int frame_h,
                           int max_detections) {
  if (tensors.empty()) {
    throw std::runtime_error("model returned no segmentation tensors");
  }

  const auto decoded =
      simaai::neat::decode_segmentation(tensors, frame_w, frame_h, max_detections, false);
  std::vector<SegmentationDetection> detections;
  const size_t mask_bytes = 160U * 160U;
  for (const auto& item : decoded) {
    if (!item.boxes.shape.empty() && item.boxes.shape.front() == 0) {
      continue;
    }
    const auto boxes = tensor_to_floats(item.boxes);
    const auto masks = tensor_to_u8(item.masks);
    const int count = static_cast<int>(boxes.size() / 6U);
    for (int i = 0; i < count; ++i) {
      const float* row = boxes.data() + static_cast<size_t>(i) * 6U;
      if (row[2] <= row[0] || row[3] <= row[1]) {
        continue;
      }
      SegmentationDetection det;
      det.x1 = row[0];
      det.y1 = row[1];
      det.x2 = row[2];
      det.y2 = row[3];
      det.score = row[4];
      det.class_id = static_cast<int>(row[5]);
      if (masks.size() >= (static_cast<size_t>(i) + 1U) * mask_bytes) {
        cv::Mat mask(160, 160, CV_8UC1,
                     const_cast<std::uint8_t*>(masks.data() + static_cast<size_t>(i) * mask_bytes));
        det.mask = mask.clone();
      }
      detections.push_back(std::move(det));
      if (static_cast<int>(detections.size()) >= max_detections) {
        return detections;
      }
    }
  }
  return detections;
}

struct SourceGeometry {
  int width = 0;
  int height = 0;
  int fps = 0;
};

int fps_from_rate(const std::string& value) {
  if (value.empty() || value == "0/0" || value == "0/1")
    return 0;
  try {
    const auto slash = value.find('/');
    double fps = 0.0;
    if (slash == std::string::npos) {
      fps = std::stod(value);
    } else {
      const double den = std::stod(value.substr(slash + 1));
      if (den <= 0.0)
        return 0;
      fps = std::stod(value.substr(0, slash)) / den;
    }
    return fps > 0.0 ? static_cast<int>(std::lround(fps)) : 0;
  } catch (...) {
    return 0;
  }
}

std::string shell_quote(const std::string& value) {
  std::string out = "'";
  for (const char c : value) {
    out += c == '\'' ? "'\\''" : std::string(1, c);
  }
  out += "'";
  return out;
}

SourceGeometry probe_http_ffprobe_geometry(const AppConfig& cfg) {
  SourceGeometry geometry;
  std::string command =
      "ffprobe -v error -rw_timeout 5000000 -select_streams v:0 "
      "-show_entries stream=width,height,r_frame_rate,avg_frame_rate -of default=nw=1 ";
  if (!cfg.ssl_strict) {
    command += "-tls_verify 0 ";
  }
  command += shell_quote(cfg.source_url) + " 2>/dev/null";

  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    return geometry;
  }

  int avg_fps = 0;
  int r_fps = 0;
  std::array<char, 256> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
    std::string line(buffer.data());
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
      line.pop_back();
    }
    const auto eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    if (key == "width") {
      geometry.width = std::atoi(value.c_str());
    } else if (key == "height") {
      geometry.height = std::atoi(value.c_str());
    } else if (key == "avg_frame_rate") {
      avg_fps = fps_from_rate(value);
    } else if (key == "r_frame_rate") {
      r_fps = fps_from_rate(value);
    }
  }
  pclose(pipe);
  geometry.fps = avg_fps > 0 ? avg_fps : r_fps;
  return geometry;
}

void fill_missing_geometry(SourceGeometry& dst, const SourceGeometry& src) {
  if (dst.width <= 0)
    dst.width = src.width;
  if (dst.height <= 0)
    dst.height = src.height;
  if (dst.fps <= 0)
    dst.fps = src.fps;
}

void require_mjpeg_fps(const AppConfig& cfg, const SourceGeometry& geometry) {
  if (cfg.source_codec == SourceCodec::Mjpeg && geometry.fps <= 0) {
    throw std::runtime_error(
        "MJPEG source did not provide a valid frame rate; set source.fps or use a source with "
        "probeable FPS metadata");
  }
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
make_rtsp_source_options(const AppConfig& cfg, const SourceGeometry& geometry) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.codec = cfg.source_codec == SourceCodec::H264 ? simaai::neat::nodes::groups::RtspCodec::H264
              : cfg.source_codec == SourceCodec::H265
                  ? simaai::neat::nodes::groups::RtspCodec::H265
                  : simaai::neat::nodes::groups::RtspCodec::MJPEG;
  opt.source_fps = geometry.fps;
  if (cfg.source_codec == SourceCodec::H264) {
    opt.auto_caps_from_stream = true;
    opt.fallback_h264_width = geometry.width;
    opt.fallback_h264_height = geometry.height;
  } else if (cfg.source_codec == SourceCodec::H265) {
    opt.auto_caps_from_stream = true;
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  } else {
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  }
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions
make_http_mjpeg_source_options(const AppConfig& cfg, const SourceGeometry& geometry) {
  simaai::neat::nodes::groups::HttpMjpegDecodedInputOptions opt;
  opt.url = cfg.source_url;
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.source_fps = geometry.fps;
  opt.ssl_strict = cfg.ssl_strict;
  if (geometry.width > 0 && geometry.height > 0 && geometry.fps > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = geometry.width;
    opt.output_caps.height = geometry.height;
    opt.output_caps.fps = geometry.fps;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

simaai::neat::Graph make_source_graph(const AppConfig& cfg, const SourceGeometry& geometry) {
  if (cfg.source_type == SourceType::Rtsp) {
    return simaai::neat::nodes::groups::RtspDecodedInput(make_rtsp_source_options(cfg, geometry));
  }
  return simaai::neat::nodes::groups::HttpMjpegDecodedInput(
      make_http_mjpeg_source_options(cfg, geometry));
}

SourceGeometry probe_shared_rtsp_geometry(const AppConfig& cfg) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(cfg.source_url, probe_options, probe);

  SourceGeometry geometry;
  geometry.width = probe.width;
  geometry.height = probe.height;
  geometry.fps = probe.fps;
  return geometry;
}

SourceGeometry probe_rtsp_geometry(const AppConfig& cfg) {
  SourceGeometry geometry = probe_shared_rtsp_geometry(cfg);
  if (cfg.source_fps > 0) {
    geometry.fps = cfg.source_fps;
  }
  require_mjpeg_fps(cfg, geometry);
  return geometry;
}

SourceGeometry probe_decoded_source_geometry(const AppConfig& cfg, int fps) {
  SourceGeometry geometry;
  geometry.fps = fps;

  simaai::neat::Graph probe_graph("source_probe");
  probe_graph.add(make_source_graph(cfg, geometry));
  probe_graph.add(simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(1)));

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  simaai::neat::Run run = probe_graph.build(run_options);

  simaai::neat::Sample sample;
  simaai::neat::PullError pull_error;
  const auto status = run.pull("frame", 20000, sample, &pull_error);
  run.close();
  if (status != simaai::neat::PullStatus::Ok) {
    throw std::runtime_error("failed to probe decoded source frame: " + pull_error.message);
  }

  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (!tensors.empty()) {
    (void)sima_examples::infer_dims(tensors.front(), geometry.width, geometry.height);
  }
  return geometry;
}

SourceGeometry resolve_source_geometry(const AppConfig& cfg) {
  if (cfg.source_type == SourceType::Rtsp) {
    return probe_rtsp_geometry(cfg);
  }
  SourceGeometry geometry = probe_http_ffprobe_geometry(cfg);
  if (cfg.source_fps > 0) {
    geometry.fps = cfg.source_fps;
  }
  require_mjpeg_fps(cfg, geometry);
  if (geometry.width <= 0 || geometry.height <= 0) {
    fill_missing_geometry(geometry, probe_decoded_source_geometry(cfg, geometry.fps));
  }
  return geometry;
}

std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg,
                                                const SourceGeometry& geometry) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  if (geometry.width > 0 && geometry.height > 0) {
    opt.preprocess.input_max_width = geometry.width;
    opt.preprocess.input_max_height = geometry.height;
  }
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV26Seg;
  opt.score_threshold = cfg.min_score;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, opt);
}

cv::Mat tensor_bgr_from_decoded(const simaai::neat::Tensor& tensor) {
  cv::Mat bgr;
  std::string err;
  if (sima_examples::nv12_to_bgr(tensor, bgr, err)) {
    return bgr;
  }
  return tensor.to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
}

const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                       const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

const simaai::neat::Sample& joined_field(const simaai::neat::Sample& sample,
                                         const std::string& label, size_t bundle_index) {
  if (const auto* field = find_field(sample, label)) {
    return *field;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle && sample.fields.size() > bundle_index) {
    return sample.fields[bundle_index];
  }
  throw std::runtime_error("joined output missing " + label + " field");
}

simaai::neat::Tensor frame_tensor_from_sample(const simaai::neat::Sample& sample) {
  const auto tensors = simaai::neat::tensors_from_sample(joined_field(sample, "frame", 0U), true);
  return tensors.front();
}

simaai::neat::TensorList segment_tensors_from_sample(const simaai::neat::Sample& sample) {
  // Without save_dir there is nothing to combine, so the pulled sample is the segments payload.
  const simaai::neat::Sample& field = sample.kind == simaai::neat::SampleKind::Bundle
                                          ? joined_field(sample, "segments", 1U)
                                          : sample;
  return simaai::neat::tensors_from_sample(field, true);
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  return class_id >= 0 && class_id < static_cast<int>(labels.size()) ? labels[class_id] : "unknown";
}

cv::Scalar class_color(int class_id) {
  static const std::vector<cv::Scalar> palette = {
      cv::Scalar(56, 56, 255),  cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
      cv::Scalar(29, 178, 255), cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
      cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),  cv::Scalar(52, 147, 26),
      cv::Scalar(187, 212, 0),  cv::Scalar(255, 194, 0),   cv::Scalar(168, 153, 44),
  };
  return palette[static_cast<size_t>(std::max(class_id, 0)) % palette.size()];
}

cv::Rect frame_rect_for_detection(const SegmentationDetection& det, const cv::Size& frame_size) {
  const int x0 = std::clamp(static_cast<int>(std::floor(det.x1)), 0, frame_size.width - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(det.y1)), 0, frame_size.height - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil(det.x2)), x0 + 1, frame_size.width);
  const int y1 = std::clamp(static_cast<int>(std::ceil(det.y2)), y0 + 1, frame_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

void draw_box(cv::Mat& frame, const SegmentationDetection& det,
              const std::vector<std::string>& labels) {
  const cv::Rect rect = frame_rect_for_detection(det, frame.size());
  const cv::Scalar color = class_color(det.class_id);
  cv::rectangle(frame, rect, color, 2);
  cv::putText(frame,
              class_name(labels, det.class_id) + " " + std::to_string(det.score).substr(0, 4),
              cv::Point(rect.x, std::max(0, rect.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
              cv::LINE_AA);
}

cv::Mat overlay_segmentation(const cv::Mat& frame,
                             const std::vector<SegmentationDetection>& detections,
                             const std::vector<std::string>& labels, const AppConfig& cfg) {
  cv::Mat annotated = frame.clone();
  for (const auto& det : detections) {
    if (det.score < cfg.min_score || det.mask.empty()) {
      continue;
    }
    const cv::Rect frame_rect = frame_rect_for_detection(det, annotated.size());
    cv::Mat resized_mask = project_letterbox_mask_roi(det.mask, frame_rect, annotated.size());
    cv::Mat binary_mask;
    cv::threshold(resized_mask, binary_mask, cfg.mask_threshold * 255.0, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(binary_mask) > 0) {
      cv::Mat annotated_roi = annotated(frame_rect);
      cv::Mat mask_color(frame_rect.size(), annotated.type(), class_color(det.class_id));
      cv::Mat blended;
      cv::addWeighted(annotated_roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0,
                      blended);
      blended.copyTo(annotated_roi, binary_mask);

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      cv::drawContours(annotated_roi, contours, -1, class_color(det.class_id), 2);
    }
    if (cfg.draw_boxes) {
      draw_box(annotated, det, labels);
    }
  }
  return annotated;
}

std::vector<MetadataSegment>
build_metadata_segments(const std::vector<SegmentationDetection>& detections,
                        const std::vector<std::string>& labels, const cv::Size& frame_size,
                        double mask_threshold) {
  std::vector<MetadataSegment> segments;
  segments.reserve(detections.size());
  for (const auto& det : detections) {
    if (det.mask.empty()) {
      continue;
    }
    const cv::Rect rect = frame_rect_for_detection(det, frame_size);
    auto polygon = mask_polygon(det.mask, rect, frame_size, mask_threshold);
    if (polygon.empty()) {
      continue;
    }
    segments.push_back({"seg_" + std::to_string(segments.size() + 1),
                        class_name(labels, det.class_id), det.score, rect, std::move(polygon)});
  }
  return segments;
}

PipelineRuntime build_pipeline(const AppConfig& cfg) {
  PipelineRuntime runtime;
  const SourceGeometry geometry = resolve_source_geometry(cfg);
  runtime.frame_w = geometry.width;
  runtime.frame_h = geometry.height;
  runtime.output_fps = geometry.fps;
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe source frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to resolve source frame rate");

  runtime.model = make_model(cfg, geometry);
  runtime.labels = load_labels(cfg.labels_path);

  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      runtime.frame_w, runtime.frame_h, runtime.output_fps);
  video_options.host = cfg.insight_host;
  video_options.channel = 0;
  video_options.video_port_base = cfg.video_port;
  video_options.encoder.bitrate_kbps = 1000;
  runtime.video_port = video_options.video_port();

  // Insight correlates the RTP timestamp with the metadata timestamp, so the encoder and the
  // segments must stay in one Run and therefore on one GStreamer timeline.
  const bool save_frames = !cfg.save_dir.empty();
  auto source = make_source_graph(cfg, geometry);
  auto branch = save_frames ? simaai::neat::graphs::Branch("source", {"video", "model", "frame"})
                            : simaai::neat::graphs::Branch("source", {"video", "model"});

  simaai::neat::Graph video_graph("video");
  video_graph.connect(simaai::neat::nodes::Input("video"),
                      simaai::neat::nodes::groups::VideoSender(video_options));

  simaai::neat::Graph model_graph("model");
  model_graph.connect(simaai::neat::nodes::Input("model"), *runtime.model);

  simaai::neat::Graph segments_graph("segments");
  segments_graph.add(
      simaai::neat::nodes::Output("segments", simaai::neat::OutputOptions::EveryFrame(4)));

  runtime.graph.connect(source, branch);
  runtime.graph.connect(branch, video_graph);
  runtime.graph.connect(branch, model_graph);
  runtime.graph.connect(model_graph, segments_graph);
  if (save_frames) {
    simaai::neat::Graph frame_graph("frame");
    frame_graph.add(
        simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(4)));
    auto joined = simaai::neat::graphs::Combine({"frame", "segments"}, "segmentation_output",
                                                simaai::neat::CombinePolicy::ByFrame);
    runtime.graph.connect(branch, frame_graph);
    runtime.graph.connect(frame_graph, joined);
    runtime.graph.connect(segments_graph, joined);
  }
  runtime.output_name = save_frames ? "segmentation_output" : "segments";
  if (cfg.profile) {
    std::cout << "Backend:\n" << runtime.graph.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  runtime.run = runtime.graph.build(run_options);

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = 0;
  metadata_options.metadata_port_base = cfg.metadata_port;
  std::string metadata_err;
  runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);

  std::cout << "source=" << cfg.source_url << " type=" << source_type_name(cfg.source_type)
            << " codec=" << source_codec_name(cfg.source_codec) << " stream=" << runtime.frame_w
            << "x" << runtime.frame_h << "@" << runtime.output_fps
            << " insight=" << cfg.insight_host << " video=" << runtime.video_port
            << " metadata=" << runtime.metadata_sender->metadata_port() << " channel=0\n";
  return runtime;
}

/// Sends one `segmentation` message and reports how many segments the byte budget dropped.
int send_metadata(PipelineRuntime& runtime, const AppConfig& cfg,
                  const simaai::neat::Sample& sample,
                  const std::vector<SegmentationDetection>& detections) {
  const auto encoded = encode_segments(build_metadata_segments(
      detections, runtime.labels, cv::Size(runtime.frame_w, runtime.frame_h), cfg.mask_threshold));
  const int64_t ts_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!runtime.metadata_sender->send_metadata("segmentation", encoded.data_json, ts_ms, frame_id,
                                              &err)) {
    std::cerr << "[warn] insight metadata send failed: " << err << "\n";
  }
  return encoded.dropped;
}

void maybe_save_frame(const AppConfig& cfg, int processed, const simaai::neat::Sample& sample,
                      const std::vector<SegmentationDetection>& detections,
                      const std::vector<std::string>& labels) {
  if (cfg.save_dir.empty() || cfg.save_every <= 0 || processed % cfg.save_every != 0) {
    return;
  }
  const cv::Mat frame = tensor_bgr_from_decoded(frame_tensor_from_sample(sample));
  const cv::Mat annotated = overlay_segmentation(frame, detections, labels, cfg);
  const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), annotated)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

void run_pipeline(PipelineRuntime& runtime, const AppConfig& cfg) {
  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  int processed = 0;
  int dropped_total = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    const double pull_start = time_ms();
    const auto status = runtime.run.pull(runtime.output_name, 20000, sample, &pull_error);
    const double pull_end = time_ms();
    if (status == simaai::neat::PullStatus::Timeout) {
      std::cerr << "[warn] timed out waiting for segmentation output\n";
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      throw std::runtime_error("failed to pull segmentation output: " + pull_error.message);
    }

    const double decode_start = time_ms();
    const auto detections = decode_segmentation_output(
        segment_tensors_from_sample(sample), runtime.frame_w, runtime.frame_h, cfg.max_detections);
    const double decode_end = time_ms();

    const double metadata_start = time_ms();
    const int dropped = send_metadata(runtime, cfg, sample, detections);
    const double metadata_end = time_ms();
    if (dropped > 0 && dropped_total == 0) {
      std::cerr << "[warn] metadata byte budget exceeded, dropped " << dropped << " segments\n";
    }
    dropped_total += dropped;

    ++processed;
    maybe_save_frame(cfg, processed, sample, detections, runtime.labels);
    profile.add(pull_end - pull_start, decode_end - decode_start, metadata_end - metadata_start,
                static_cast<int>(detections.size()), dropped);
  }

  profile.flush();
  std::cout << "processed=" << processed << " dropped_segments=" << dropped_total
            << " video_sender=" << cfg.insight_host << ":" << runtime.video_port << "\n";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const CliOptions cli = parse_args(argc, argv);
    const AppConfig cfg = load_app_config(cli.config_path);
    if (cli.validate_config_only) {
      std::cout << "Config validated: " << cli.config_path << "\n";
      return 0;
    }
    if (!cfg.save_dir.empty()) {
      fs::create_directories(cfg.save_dir);
    }
    if (cfg.profile) {
      setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
      setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
      setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
    }

    PipelineRuntime runtime = build_pipeline(cfg);
    run_pipeline(runtime, cfg);
    runtime.run.close();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 2;
  }
}
