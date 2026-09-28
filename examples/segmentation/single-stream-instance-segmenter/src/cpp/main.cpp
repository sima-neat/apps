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
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

/// Both families emit masks at one quarter of the model input per dimension, so a 160x160
/// mask grid corresponds to a 640x640 input.
constexpr int kMaskStride = 4;

/// BoxDecode returns YOLO26 masks on a fixed grid, independent of the model input size.
constexpr int kYolo26MaskGrid = 160;

/// YOLOv8 head contract: three feature levels of box, class, and mask-coefficient tensors,
/// followed by the mask prototypes.
constexpr std::size_t kYolov8HeadTensors = 10;
constexpr std::array<int, 3> kYolov8Strides = {8, 16, 32};

/// Distribution-focal-loss bins per box side, and prototype/coefficient depth.
constexpr int kDflBins = 16;
constexpr int kMaskCoefficients = 32;

/// Model families this application decodes. YOLO26 decodes on the MLA through the packaged
/// BoxDecode route; YOLOv8 surfaces raw heads that are decoded here. Both produce the same
/// `SegmentationDetection` records, so everything downstream is family independent.
enum class ModelFamily { Yolo26, YoloV8 };

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
  ModelFamily model_family = ModelFamily::Yolo26;
  std::string model_path;
  fs::path labels_path;
  int input_size = 640;
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
  /// Separate decoded-frame output, used when frames are paired by this application instead
  /// of by a graph-side join. Empty when the graph joins them.
  std::string frame_output_name;
  /// Recent decoded frames, oldest first, waiting to be paired with their segments.
  std::deque<std::pair<std::int64_t, simaai::neat::Sample>> frames;
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

ModelFamily parse_model_family(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "yolo26" || lowered == "yolo-26" || lowered == "yolov26") {
    return ModelFamily::Yolo26;
  }
  if (lowered == "yolov8" || lowered == "yolo-v8" || lowered == "yolo_v8") {
    return ModelFamily::YoloV8;
  }
  throw std::runtime_error("model.family must be yolo26 or yolov8");
}

const char* model_family_name(ModelFamily value) {
  return value == ModelFamily::Yolo26 ? "yolo26" : "yolov8";
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
  sima_examples::require(cfg.input_size > 0 &&
                             cfg.input_size % (kMaskStride * kYolov8Strides.back()) == 0,
                         "model.input_size must be a positive multiple of " +
                             std::to_string(kMaskStride * kYolov8Strides.back()));
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
  cfg.model_family = parse_model_family(raw.string_or("model.family", "yolo26"));
  cfg.model_path = raw.string_or("model.path", "");
  cfg.input_size = raw.int_or("model.input_size", 640);
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

/// Detection rectangle in frame pixels, clamped to the frame and never empty.
cv::Rect frame_rect_for_detection(const SegmentationDetection& det, const cv::Size& frame_size) {
  const int x0 = std::clamp(static_cast<int>(std::floor(det.x1)), 0, frame_size.width - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(det.y1)), 0, frame_size.height - 1);
  const int x1 = std::clamp(static_cast<int>(std::ceil(det.x2)), x0 + 1, frame_size.width);
  const int y1 = std::clamp(static_cast<int>(std::ceil(det.y2)), y0 + 1, frame_size.height);
  return cv::Rect(x0, y0, x1 - x0, y1 - y0);
}

/// YOLO26 boundary: the MLA already ran BoxDecode, so this only unpacks its payload.
std::vector<SegmentationDetection>
decode_yolo26_segments(const simaai::neat::TensorList& tensors, int frame_w, int frame_h,
                       int max_detections) {
  if (tensors.empty()) {
    throw std::runtime_error("model returned no segmentation tensors");
  }

  const auto decoded =
      simaai::neat::decode_segmentation(tensors, frame_w, frame_h, max_detections, false);
  std::vector<SegmentationDetection> detections;
  const size_t mask_bytes = static_cast<size_t>(kYolo26MaskGrid) * kYolo26MaskGrid;
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
        cv::Mat mask(kYolo26MaskGrid, kYolo26MaskGrid, CV_8UC1,
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

/// One head tensor as a dense HWC float32 buffer, dropping a leading batch axis of 1.
struct TensorHWC {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;

  /// The `c` values stored for one cell.
  const float* row(int y, int x) const {
    const size_t index =
        (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) *
        static_cast<size_t>(c);
    return data.data() + index;
  }
};

TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& tensor) {
  TensorHWC out;
  if (tensor.shape.size() == 4) {
    if (tensor.shape[0] != 1) {
      throw std::runtime_error("only batch size 1 is supported");
    }
    out.h = static_cast<int>(tensor.shape[1]);
    out.w = static_cast<int>(tensor.shape[2]);
    out.c = static_cast<int>(tensor.shape[3]);
  } else if (tensor.shape.size() == 3) {
    out.h = static_cast<int>(tensor.shape[0]);
    out.w = static_cast<int>(tensor.shape[1]);
    out.c = static_cast<int>(tensor.shape[2]);
  } else {
    throw std::runtime_error("unexpected head tensor rank " +
                             std::to_string(tensor.shape.size()));
  }

  const auto floats = tensor_to_floats(tensor);
  const size_t elements =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  if (floats.size() < elements) {
    throw std::runtime_error("head tensor holds fewer elements than its shape declares");
  }
  out.data.assign(floats.begin(), floats.begin() + static_cast<std::ptrdiff_t>(elements));
  return out;
}

/// The packaged YOLOv8 heads, grouped by role.
struct Yolov8Heads {
  std::vector<TensorHWC> boxes;
  std::vector<TensorHWC> scores;
  std::vector<TensorHWC> coefficients;
  TensorHWC proto;
};

/// Groups the head tensors and checks them against `model.input_size`.
Yolov8Heads split_yolov8_heads(const simaai::neat::TensorList& tensors, int input_size) {
  if (tensors.size() < kYolov8HeadTensors) {
    throw std::runtime_error("YOLOv8 decode expects " + std::to_string(kYolov8HeadTensors) +
                             " head tensors, got " + std::to_string(tensors.size()));
  }

  Yolov8Heads heads;
  for (size_t level = 0; level < kYolov8Strides.size(); ++level) {
    heads.boxes.push_back(tensor_to_hwc_f32(tensors[level]));
    heads.scores.push_back(tensor_to_hwc_f32(tensors[level + 3U]));
    heads.coefficients.push_back(tensor_to_hwc_f32(tensors[level + 6U]));
  }
  heads.proto = tensor_to_hwc_f32(tensors[9]);

  if (heads.proto.c != kMaskCoefficients) {
    throw std::runtime_error("unexpected prototype channels " + std::to_string(heads.proto.c));
  }
  if (heads.proto.h != heads.proto.w || heads.proto.h * kMaskStride != input_size) {
    throw std::runtime_error("prototype grid " + std::to_string(heads.proto.h) + "x" +
                             std::to_string(heads.proto.w) +
                             " does not match model.input_size " + std::to_string(input_size));
  }
  for (size_t level = 0; level < kYolov8Strides.size(); ++level) {
    const int grid = input_size / kYolov8Strides[level];
    const TensorHWC& box = heads.boxes[level];
    const TensorHWC& score = heads.scores[level];
    const TensorHWC& coefficient = heads.coefficients[level];
    if (box.h != grid || box.w != grid || box.c != 4 * kDflBins) {
      throw std::runtime_error("unexpected box head shape for stride " +
                               std::to_string(kYolov8Strides[level]) + " at model.input_size " +
                               std::to_string(input_size));
    }
    if (score.h != grid || score.w != grid || score.c <= 0) {
      throw std::runtime_error("unexpected class head shape for stride " +
                               std::to_string(kYolov8Strides[level]));
    }
    if (coefficient.h != grid || coefficient.w != grid ||
        coefficient.c != kMaskCoefficients) {
      throw std::runtime_error("unexpected mask-coefficient head shape for stride " +
                               std::to_string(kYolov8Strides[level]));
    }
  }
  return heads;
}

/// Expected distance, in cells, of one distribution-focal-loss box side.
float dfl_distance(const float* logits) {
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int bin = 0; bin < kDflBins; ++bin) {
    max_logit = std::max(max_logit, logits[bin]);
  }
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (int bin = 0; bin < kDflBins; ++bin) {
    const float weight = std::exp(logits[bin] - max_logit);
    numerator += static_cast<float>(bin) * weight;
    denominator += weight;
  }
  return denominator > 0.0f ? numerator / denominator : 0.0f;
}

/// One YOLOv8 instance before NMS, in letterboxed model pixels.
struct Yolov8Candidate {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  std::array<float, kMaskCoefficients> coefficients{};
};

std::vector<Yolov8Candidate> yolov8_candidates(const Yolov8Heads& heads, int input_size,
                                               double min_score) {
  std::vector<Yolov8Candidate> candidates;
  for (size_t level = 0; level < kYolov8Strides.size(); ++level) {
    const TensorHWC& box = heads.boxes[level];
    const TensorHWC& score = heads.scores[level];
    const TensorHWC& coefficient = heads.coefficients[level];
    const float stride = static_cast<float>(input_size) / static_cast<float>(box.h);

    for (int y = 0; y < box.h; ++y) {
      for (int x = 0; x < box.w; ++x) {
        // The packaged class head already carries probabilities, so it is thresholded as is.
        const float* class_scores = score.row(y, x);
        int best_class = 0;
        float best_score = class_scores[0];
        for (int c = 1; c < score.c; ++c) {
          if (class_scores[c] > best_score) {
            best_score = class_scores[c];
            best_class = c;
          }
        }
        if (best_score < static_cast<float>(min_score)) {
          continue;
        }

        const float* sides = box.row(y, x);
        const float left = dfl_distance(sides) * stride;
        const float top = dfl_distance(sides + kDflBins) * stride;
        const float right = dfl_distance(sides + 2 * kDflBins) * stride;
        const float bottom = dfl_distance(sides + 3 * kDflBins) * stride;
        const float center_x = (static_cast<float>(x) + 0.5f) * stride;
        const float center_y = (static_cast<float>(y) + 0.5f) * stride;

        Yolov8Candidate candidate;
        candidate.x1 = std::clamp(center_x - left, 0.0f, static_cast<float>(input_size));
        candidate.y1 = std::clamp(center_y - top, 0.0f, static_cast<float>(input_size));
        candidate.x2 = std::clamp(center_x + right, 0.0f, static_cast<float>(input_size));
        candidate.y2 = std::clamp(center_y + bottom, 0.0f, static_cast<float>(input_size));
        candidate.score = best_score;
        candidate.class_id = best_class;
        const float* coefficient_row = coefficient.row(y, x);
        std::copy(coefficient_row, coefficient_row + kMaskCoefficients,
                  candidate.coefficients.begin());
        if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1) {
          candidates.push_back(candidate);
        }
      }
    }
  }
  return candidates;
}

float iou_xyxy(const Yolov8Candidate& a, const Yolov8Candidate& b) {
  const float width = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  const float height = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  const float intersection = width * height;
  const float union_area =
      (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection;
  return union_area > 0.0f ? intersection / union_area : 0.0f;
}

/// Greedy per-class NMS, highest score first, capped at `max_detections`.
std::vector<Yolov8Candidate> nms_per_class(std::vector<Yolov8Candidate> candidates, double nms_iou,
                                           int max_detections) {
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Yolov8Candidate& a, const Yolov8Candidate& b) {
                     return a.score > b.score;
                   });
  std::vector<Yolov8Candidate> kept;
  kept.reserve(static_cast<size_t>(max_detections));
  for (const auto& candidate : candidates) {
    if (static_cast<int>(kept.size()) >= max_detections) {
      break;
    }
    const bool suppressed =
        std::any_of(kept.begin(), kept.end(), [&](const Yolov8Candidate& keeper) {
          return keeper.class_id == candidate.class_id &&
                 iou_xyxy(keeper, candidate) > static_cast<float>(nms_iou);
        });
    if (!suppressed) {
      kept.push_back(candidate);
    }
  }
  return kept;
}

/// Prototype mask for one instance, on the same grid BoxDecode returns for YOLO26.
cv::Mat yolov8_instance_mask(const TensorHWC& proto,
                             const std::array<float, kMaskCoefficients>& coefficients,
                             const cv::Rect& frame_rect, const cv::Size& frame_size) {
  cv::Mat mask = cv::Mat::zeros(proto.h, proto.w, CV_8UC1);
  const cv::Rect mask_rect =
      mask_rect_for_frame_rect(frame_rect, frame_size, cv::Size(proto.w, proto.h));
  for (int y = mask_rect.y; y < mask_rect.y + mask_rect.height; ++y) {
    std::uint8_t* row = mask.ptr<std::uint8_t>(y);
    for (int x = mask_rect.x; x < mask_rect.x + mask_rect.width; ++x) {
      const float* prototypes = proto.row(y, x);
      float logit = 0.0f;
      for (int k = 0; k < kMaskCoefficients; ++k) {
        logit += prototypes[k] * coefficients[static_cast<size_t>(k)];
      }
      row[x] = cv::saturate_cast<std::uint8_t>(255.0f / (1.0f + std::exp(-logit)));
    }
  }
  return mask;
}

/// YOLOv8 boundary: decode raw heads here into the shared detection representation.
std::vector<SegmentationDetection>
decode_yolov8_segments(const simaai::neat::TensorList& tensors, int frame_w, int frame_h,
                       const AppConfig& cfg) {
  const Yolov8Heads heads = split_yolov8_heads(tensors, cfg.input_size);
  const auto kept = nms_per_class(yolov8_candidates(heads, cfg.input_size, cfg.min_score),
                                  cfg.nms_iou, cfg.max_detections);

  // Undo the letterbox the model preprocess applied, so boxes land in frame pixels.
  const double scale = std::min(static_cast<double>(cfg.input_size) / frame_w,
                                static_cast<double>(cfg.input_size) / frame_h);
  const double pad_x = (static_cast<double>(cfg.input_size) - frame_w * scale) * 0.5;
  const double pad_y = (static_cast<double>(cfg.input_size) - frame_h * scale) * 0.5;
  const cv::Size frame_size(frame_w, frame_h);

  std::vector<SegmentationDetection> detections;
  detections.reserve(kept.size());
  for (const auto& candidate : kept) {
    SegmentationDetection det;
    det.x1 = static_cast<float>(std::clamp((candidate.x1 - pad_x) / scale, 0.0, 1.0 * frame_w));
    det.y1 = static_cast<float>(std::clamp((candidate.y1 - pad_y) / scale, 0.0, 1.0 * frame_h));
    det.x2 = static_cast<float>(std::clamp((candidate.x2 - pad_x) / scale, 0.0, 1.0 * frame_w));
    det.y2 = static_cast<float>(std::clamp((candidate.y2 - pad_y) / scale, 0.0, 1.0 * frame_h));
    if (det.x2 <= det.x1 || det.y2 <= det.y1) {
      continue;
    }
    det.score = candidate.score;
    det.class_id = candidate.class_id;
    det.mask = yolov8_instance_mask(heads.proto, candidate.coefficients,
                                    frame_rect_for_detection(det, frame_size), frame_size);
    detections.push_back(std::move(det));
  }
  return detections;
}

/// The one place model family changes behavior.
std::vector<SegmentationDetection> decode_segments(const AppConfig& cfg,
                                                   const simaai::neat::TensorList& tensors,
                                                   int frame_w, int frame_h) {
  if (cfg.model_family == ModelFamily::Yolo26) {
    return decode_yolo26_segments(tensors, frame_w, frame_h, cfg.max_detections);
  }
  return decode_yolov8_segments(tensors, frame_w, frame_h, cfg);
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

/// Loads the segmentation package. Only the postprocess contract differs per family.
std::unique_ptr<simaai::neat::Model> make_model(const AppConfig& cfg,
                                                const SourceGeometry& geometry) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  // The decoder emits NV12; the model preprocess letterboxes it onto the model input.
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  if (geometry.width > 0 && geometry.height > 0) {
    opt.preprocess.input_max_width = geometry.width;
    opt.preprocess.input_max_height = geometry.height;
  }
  if (cfg.model_family == ModelFamily::Yolo26) {
    opt.preprocess.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    // BoxDecode runs on device and emits the segmentation payload this app unpacks.
    opt.decode_type = simaai::neat::BoxDecodeType::YoloV26Seg;
    opt.score_threshold = cfg.min_score;
    opt.nms_iou_threshold = cfg.nms_iou;
    opt.top_k = cfg.max_detections;
  } else {
    // YOLOv8 leaves decode_type unset so the route surfaces the raw float heads that
    // decode_yolov8_segments() consumes. The decode inverts a letterbox, so the resize policy
    // is requested rather than assumed, and the YOLO normalization is requested explicitly.
    opt.preprocess.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
    opt.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
  }
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
  // A graph-joined bundle carries the frame in a field; a separate frame output is the sample.
  const simaai::neat::Sample& field = sample.kind == simaai::neat::SampleKind::Bundle
                                          ? joined_field(sample, "frame", 0U)
                                          : sample;
  const auto tensors = simaai::neat::tensors_from_sample(field, true);
  return tensors.front();
}

/// How many decoded frames may wait for their segments. The segments branch trails the frame
/// branch by the model and host-decode latency, so this only has to cover that lag. Output
/// memory is owned on this route, so a retained frame costs memory, not a pipeline buffer.
constexpr std::size_t kFrameRingCapacity = 16;

/// Moves every frame the run has ready into the ring, dropping the oldest past capacity.
/// Draining every iteration is what keeps the frame output queue from backing up.
void drain_frames(PipelineRuntime& runtime) {
  while (true) {
    simaai::neat::Sample frame;
    simaai::neat::PullError pull_error;
    if (runtime.run.pull(runtime.frame_output_name, 0, frame, &pull_error) !=
        simaai::neat::PullStatus::Ok) {
      return;
    }
    const std::int64_t frame_id = frame.frame_id;
    runtime.frames.emplace_back(frame_id, std::move(frame));
    if (runtime.frames.size() > kFrameRingCapacity) {
      runtime.frames.pop_front();
    }
  }
}

/// Waits for the next segments sample while keeping the frame branch drained.
///
/// A single long blocking pull would leave the frame output queue unattended, and the frames
/// discarded there under the keep-latest policy are exactly the partners a saved frame needs,
/// so the wait is split into short slices with a drain between them.
simaai::neat::PullStatus pull_segments(PipelineRuntime& runtime, int timeout_ms,
                                       simaai::neat::Sample& sample,
                                       simaai::neat::PullError& pull_error) {
  if (runtime.frame_output_name.empty()) {
    return runtime.run.pull(runtime.output_name, timeout_ms, sample, &pull_error);
  }
  constexpr int kSliceMs = 20;
  const double deadline = time_ms() + timeout_ms;
  while (true) {
    drain_frames(runtime);
    const auto status = runtime.run.pull(runtime.output_name, kSliceMs, sample, &pull_error);
    if (status != simaai::neat::PullStatus::Timeout || time_ms() >= deadline) {
      return status;
    }
  }
}

/// The retained frame a segments sample was computed from, or null when it has aged out.
const simaai::neat::Sample* frame_for(const PipelineRuntime& runtime, std::int64_t frame_id) {
  if (frame_id < 0) {
    return nullptr;
  }
  for (auto it = runtime.frames.rbegin(); it != runtime.frames.rend(); ++it) {
    if (it->first == frame_id) {
      return &it->second;
    }
  }
  return nullptr;
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

  // The YOLO26 decode is a cheap payload unpack, so its output can queue frames. The YOLOv8
  // decode runs on the host and cannot match the source rate; queueing there pins the whole
  // detess stage output pool and starves it, so that output keeps only the newest sample.
  simaai::neat::Graph segments_graph("segments");
  segments_graph.add(simaai::neat::nodes::Output(
      "segments", cfg.model_family == ModelFamily::Yolo26
                      ? simaai::neat::OutputOptions::EveryFrame(4)
                      : simaai::neat::OutputOptions::EveryFrame(1)));

  runtime.graph.connect(source, branch);
  runtime.graph.connect(branch, video_graph);
  runtime.graph.connect(branch, model_graph);
  runtime.graph.connect(model_graph, segments_graph);
  if (save_frames) {
    simaai::neat::Graph frame_graph("frame");
    frame_graph.add(
        simaai::neat::nodes::Output("frame", simaai::neat::OutputOptions::EveryFrame(4)));
    runtime.graph.connect(branch, frame_graph);
    if (cfg.model_family == ModelFamily::Yolo26) {
      auto joined = simaai::neat::graphs::Combine({"frame", "segments"}, "segmentation_output",
                                                  simaai::neat::CombinePolicy::ByFrame);
      runtime.graph.connect(frame_graph, joined);
      runtime.graph.connect(segments_graph, joined);
      runtime.output_name = "segmentation_output";
    } else {
      // The graph-side join retains more model-output buffers than a YOLOv8 package's fixed
      // pool can serve, so this route publishes both streams and pairs them on frame_id below.
      runtime.frame_output_name = "frame";
      runtime.output_name = "segments";
    }
  } else {
    runtime.output_name = "segments";
  }
  if (cfg.profile) {
    std::cout << "Backend:\n" << runtime.graph.describe_backend() << "\n";
  }

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  // Measured on Modalix: the YOLOv8 head tensors must be copied out of the detess stage pool
  // at pull time, or the stage starves while the host decode runs. YOLO26 pulls one small
  // BoxDecode payload and keeps the cheaper zero-copy path.
  run_options.output_memory = cfg.model_family == ModelFamily::Yolo26
                                  ? simaai::neat::OutputMemory::ZeroCopy
                                  : simaai::neat::OutputMemory::Owned;
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
            << " codec=" << source_codec_name(cfg.source_codec)
            << " model=" << model_family_name(cfg.model_family) << " stream=" << runtime.frame_w
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

/// Whether this result is due an annotated frame.
bool save_due(const AppConfig& cfg, int processed) {
  return !cfg.save_dir.empty() && cfg.save_every > 0 && processed % cfg.save_every == 0;
}

/// Writes one annotated frame. Returns false when the decoded frame it needs is gone.
///
/// The YOLO26 route joins frames to results inside the graph and always has its partner. The
/// YOLOv8 route pairs them here, and a source faster than the model makes the two branches
/// retain different frames, so some results have no picture to annotate. Those are counted and
/// reported rather than silently skipped.
bool save_frame(const AppConfig& cfg, int processed, const simaai::neat::Sample* sample,
                const std::vector<SegmentationDetection>& detections,
                const std::vector<std::string>& labels) {
  if (sample == nullptr) {
    return false;
  }
  const cv::Mat frame = tensor_bgr_from_decoded(frame_tensor_from_sample(*sample));
  const cv::Mat annotated = overlay_segmentation(frame, detections, labels, cfg);
  const auto out_path = cfg.save_dir / ("frame_" + std::to_string(processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), annotated)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
    return false;
  }
  return true;
}

void run_pipeline(PipelineRuntime& runtime, const AppConfig& cfg) {
  ProfileWindow profile;
  profile.enabled = cfg.profile;
  profile.interval = cfg.profile_interval;

  int processed = 0;
  int dropped_total = 0;
  int saved = 0;
  int unpaired = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    const double pull_start = time_ms();
    const auto status = pull_segments(runtime, 20000, sample, pull_error);
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
    const auto detections =
        decode_segments(cfg, segment_tensors_from_sample(sample), runtime.frame_w, runtime.frame_h);
    const double decode_end = time_ms();

    const double metadata_start = time_ms();
    const int dropped = send_metadata(runtime, cfg, sample, detections);
    const double metadata_end = time_ms();
    if (dropped > 0 && dropped_total == 0) {
      std::cerr << "[warn] metadata byte budget exceeded, dropped " << dropped << " segments\n";
    }
    dropped_total += dropped;

    ++processed;
    if (save_due(cfg, processed)) {
      const simaai::neat::Sample* frame_sample = &sample;
      if (!runtime.frame_output_name.empty()) {
        drain_frames(runtime);
        frame_sample = frame_for(runtime, sample.frame_id);
      }
      if (save_frame(cfg, processed, frame_sample, detections, runtime.labels)) {
        ++saved;
      } else {
        ++unpaired;
      }
    }
    profile.add(pull_end - pull_start, decode_end - decode_start, metadata_end - metadata_start,
                static_cast<int>(detections.size()), dropped);
  }

  profile.flush();
  std::cout << "processed=" << processed << " dropped_segments=" << dropped_total
            << " saved=" << saved << " unpaired=" << unpaired
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
