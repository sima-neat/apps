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
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
#include <numeric>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

/// One model per stream. The task selects the host-side decode path and the Insight metadata
/// contract the stream publishes on.
enum class Task { Detection, Segmentation, Pose };

/// `streams[].decode` values. The family follows the shape of the archive's detection head, not
/// the model's version number: YOLO26 packages carry raw l/t/r/b distance heads, while the Model
/// Zoo YOLO11 and YOLOv8 packages carry the YOLOv8 head layout, so YOLO11 uses `yolov8`.
enum class DecodeFamily { YoloV8, Yolo26 };

const char* task_name(Task task) {
  switch (task) {
  case Task::Segmentation:
    return "segmentation";
  case Task::Pose:
    return "pose";
  case Task::Detection:
  default:
    return "detection";
  }
}

/// Insight overlay contract per task. Insight picks its renderer from this string.
const char* metadata_type(Task task) {
  switch (task) {
  case Task::Segmentation:
    return "segmentation";
  case Task::Pose:
    return "pose-estimation";
  case Task::Detection:
  default:
    return "object-detection";
  }
}

/// MetadataSender rejects a payload above 65507 bytes, and the rejection surfaces as an error the
/// application has to handle mid-stream. Half of that leaves room for the envelope and keeps the
/// datagram count low enough for Insight to reassemble within its 250 ms window.
constexpr std::size_t kMetadataByteBudget = 32768;

/// BoxDecode emits segmentation masks on a 160x160 grid covering the model's 640x640 letterboxed
/// input, so one mask cell spans four input pixels.
constexpr int kMaskStride = 4;
constexpr int kMaskGrid = 160;

/// COCO keypoint names in the order the BoxDecode pose payload emits them.
///
/// The order is fixed by the wire format, not by configuration: `decode_pose` returns a
/// `[N, 17, 3]` tensor whose second axis is positional. Insight joins skeleton edges by name,
/// so these strings are part of the published metadata contract.
constexpr std::array<const char*, 17> kCocoKeypointNames = {
    "nose",           "left_eye",   "right_eye",   "left_ear",   "right_ear",   "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
    "right_hip",      "left_knee",  "right_knee",  "left_ankle", "right_ankle"};

/// Skeleton edges drawn between keypoints, expressed as index pairs into `kCocoKeypointNames`.
/// Mirrors the COCO topology Insight renders.
constexpr std::array<std::pair<int, int>, 17> kCocoSkeleton = {{{0, 1},
                                                                {0, 2},
                                                                {1, 3},
                                                                {2, 4},
                                                                {0, 5},
                                                                {0, 6},
                                                                {5, 7},
                                                                {7, 9},
                                                                {6, 8},
                                                                {8, 10},
                                                                {5, 11},
                                                                {6, 12},
                                                                {11, 12},
                                                                {11, 13},
                                                                {13, 15},
                                                                {12, 14},
                                                                {14, 16}}};

/// The source frame's own timing, carried from the feeder to the consumer.
struct FrameStamp {
  int64_t pts_ns = -1;
  int64_t frame_id = -1;
};

/// One keypoint in source-frame pixel space. `visibility` is the decoder's per-joint confidence
/// in [0, 1]; the debug overlay uses it to hide uncertain joints.
struct Keypoint {
  float x = 0.0f;
  float y = 0.0f;
  float visibility = 0.0f;
};

/// One detected person: the bounding box that anchors the pose plus its 17 keypoints.
struct Pose {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  std::array<Keypoint, 17> keypoints{};
};

/// One segmented instance: the detection rectangle plus its 160x160 mask crop.
struct SegmentationDetection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
  cv::Mat mask;
};

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

/// Everything one stream's model produced for one frame. Exactly one vector is populated,
/// selected by the stream's task.
struct FrameResults {
  std::vector<objdet::Box> boxes;
  std::vector<SegmentationDetection> segments;
  std::vector<Pose> poses;

  [[nodiscard]] std::size_t size() const {
    return boxes.size() + segments.size() + poses.size();
  }
};

struct StreamConfig {
  int index = 0;
  std::string url;
  Task task = Task::Detection;
  DecodeFamily decode = DecodeFamily::Yolo26;
  std::string model_path;
};

struct AppConfig {
  std::vector<StreamConfig> streams;
  fs::path labels_path;
  /// Encoded RTSP path used for every stream in this application.
  simaai::neat::nodes::groups::RtspCodec codec = simaai::neat::nodes::groups::RtspCodec::H264;
  int latency_ms = 100;
  bool tcp = true;
  int frames = 0;
  int fps = 0;
  int max_inflight_per_stream = 4;
  double min_score = 0.30;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int warmup_frames = 30;
  std::string insight_host = "127.0.0.1";
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  bool video_enabled = true;
  double mask_threshold = 0.50;
  double mask_alpha = 0.55;
  double min_keypoint_visibility = 0.30;
  fs::path save_dir;
  int save_every = 0;
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct ProfileWindow {
  bool enabled = false;
  int stream_index = 0;
  Task task = Task::Detection;
  int interval = 100;
  int frames = 0;
  int instances = 0;
  double start_ms = 0.0;
  double result_pull_ms = 0.0;
  double metadata_send_ms = 0.0;

  void add(double result_pull, double metadata_send, int instance_count) {
    if (!enabled)
      return;
    if (frames == 0)
      start_ms = sima_examples::time_ms();
    ++frames;
    instances += instance_count;
    result_pull_ms += result_pull;
    metadata_send_ms += metadata_send;
    if (frames >= interval)
      flush();
  }

  void flush() {
    if (!enabled || frames == 0)
      return;
    const double elapsed = sima_examples::time_ms() - start_ms;
    const double output_fps = elapsed > 0.0 ? static_cast<double>(frames) * 1000.0 / elapsed : 0.0;
    const auto avg = [this](double value) { return value / static_cast<double>(frames); };
    std::cout << "[profile stream=" << stream_index << " task=" << task_name(task)
              << "] frames=" << frames << " output_fps=" << output_fps
              << " avg_result_pull_ms=" << avg(result_pull_ms)
              << " avg_metadata_send_ms=" << avg(metadata_send_ms)
              << " avg_instances=" << static_cast<double>(instances) / static_cast<double>(frames)
              << "\n";
    frames = 0;
    instances = 0;
    start_ms = 0.0;
    result_pull_ms = 0.0;
    metadata_send_ms = 0.0;
  }
};

struct StreamRuntime {
  int index = 0;
  Task task = Task::Detection;
  std::string url;
  std::string model_path;
  simaai::neat::nodes::groups::RtspDecodedInputOptions source_options;
  std::unique_ptr<simaai::neat::Model> model;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  std::vector<std::string> labels;
  ProfileWindow profile;
  std::optional<cv::Mat> latest_debug_frame;
  int frame_w = 0;
  int frame_h = 0;
  int output_fps = 0;
  int video_port = 0;
  int processed = 0;
  int dropped_segments = 0;
  bool closed = false;

  /// This stream's model, in a Run of its own. Four model graphs in one Run couple to each
  /// other: one stream stalls for good while its neighbours keep running at the source rate.
  simaai::neat::Graph model_graph;
  simaai::neat::Run model_run;
  /// Timing of frames handed to the model, oldest first.
  ///
  /// The model lives in its own Run, and a tensor pushed into it arrives with no PTS, so its
  /// results come back stamped -1. Insight correlates metadata to video by timestamp, so the
  /// camera's own PTS has to be carried across the Run boundary by hand. Push and pull are
  /// FIFO-paired under OverflowPolicy::Block, which makes a queue enough to re-pair them.
  std::mutex pending_mutex;
  std::deque<FrameStamp> pending;
  /// Guards latest_debug_frame: the feeder writes it while the consumer reads it.
  mutable std::mutex debug_frame_mutex;
  /// Set by the feeder so the consumer knows no more frames are coming.
  std::atomic<bool> feed_done{false};
  /// Frames handed to the model but not yet pulled back. This is the backpressure bound.
  /// OverflowPolicy does not provide one: push() returns as soon as the frame lands in the
  /// graph's internal edge queue, so a model slower than its camera runs latency into seconds.
  std::atomic<int> in_flight{0};
};

struct AppRuntime {
  /// Every stream's `rtsp -> decode` chain plus its video passthrough, in ONE graph and Run.
  /// Neat requests a decoder-admission lease only when a single graph holds more than one
  /// decoder, and without that lease four concurrent decoders do not reach the source rate.
  simaai::neat::Graph source_graph;
  simaai::neat::Run source_run;
  std::vector<std::unique_ptr<StreamRuntime>> streams;
};

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

simaai::neat::nodes::groups::RtspCodec parse_input_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return simaai::neat::nodes::groups::RtspCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return simaai::neat::nodes::groups::RtspCodec::H265;
  }
  throw std::runtime_error("input.codec must be h264/avc or h265/hevc");
}

Task parse_task(const std::string& value, std::size_t index) {
  const std::string lowered = lower_copy(sima_examples::trim_copy(value));
  if (lowered == "detection") {
    return Task::Detection;
  }
  if (lowered == "segmentation") {
    return Task::Segmentation;
  }
  if (lowered == "pose") {
    return Task::Pose;
  }
  throw std::runtime_error("streams[" + std::to_string(index) +
                           "].task must be one of detection, segmentation, pose");
}

DecodeFamily parse_decode(const std::string& value, std::size_t index) {
  const std::string lowered = lower_copy(sima_examples::trim_copy(value));
  if (lowered == "yolov8") {
    return DecodeFamily::YoloV8;
  }
  if (lowered == "yolo26") {
    return DecodeFamily::Yolo26;
  }
  throw std::runtime_error("streams[" + std::to_string(index) +
                           "].decode must be one of yolov8, yolo26");
}

const char* decode_name(DecodeFamily decode) {
  return decode == DecodeFamily::YoloV8 ? "yolov8" : "yolo26";
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

/// Split `url: rtsp://host/path` into its key and value. ScalarConfig cannot reach inside a YAML
/// sequence, so the stream list is parsed here.
bool split_key_value(const std::string& line, std::string& key, std::string& value) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) {
    return false;
  }
  key = sima_examples::trim_copy(line.substr(0, colon));
  value = unquote(line.substr(colon + 1));
  return !key.empty();
}

int leading_indent(const std::string& line) {
  int indent = 0;
  while (indent < static_cast<int>(line.size()) &&
         (line[static_cast<std::size_t>(indent)] == ' ' ||
          line[static_cast<std::size_t>(indent)] == '\t')) {
    ++indent;
  }
  return indent;
}

std::vector<StreamConfig> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + config_path.string());
  }

  std::vector<StreamConfig> streams;
  bool in_streams = false;
  int streams_indent = -1;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    const std::string line_without_comment = strip_inline_comment(raw_line);
    if (sima_examples::trim_copy(line_without_comment).empty()) {
      continue;
    }

    const int indent = leading_indent(line_without_comment);
    const std::string line = sima_examples::trim_copy(line_without_comment);

    if (in_streams && indent <= streams_indent && line.rfind("- ", 0) != 0) {
      in_streams = false;
    }
    if (!in_streams) {
      if (line == "streams:") {
        in_streams = true;
        streams_indent = indent;
      }
      continue;
    }

    std::string entry = line;
    if (entry.rfind("- ", 0) == 0) {
      streams.push_back(StreamConfig{static_cast<int>(streams.size()), "", Task::Detection,
                                     DecodeFamily::Yolo26, ""});
      entry = sima_examples::trim_copy(entry.substr(2));
    }
    if (streams.empty()) {
      throw std::runtime_error("streams entries must start with '- '");
    }

    std::string key;
    std::string value;
    if (!split_key_value(entry, key, value)) {
      throw std::runtime_error("streams entries must be mappings with url, task, and model");
    }
    StreamConfig& stream = streams.back();
    if (key == "url") {
      stream.url = value;
    } else if (key == "task") {
      stream.task = parse_task(value, streams.size() - 1);
    } else if (key == "decode") {
      stream.decode = parse_decode(value, streams.size() - 1);
    } else if (key == "model") {
      stream.model_path = value;
    }
  }

  if (streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  for (std::size_t index = 0; index < streams.size(); ++index) {
    sima_examples::require(!streams[index].url.empty(),
                           "streams[" + std::to_string(index) + "].url must be set");
    sima_examples::require(!streams[index].model_path.empty(),
                           "streams[" + std::to_string(index) + "].model must be set");
  }
  return streams;
}

void validate_config(const AppConfig& cfg) {
  sima_examples::require(!cfg.streams.empty(), "streams must be set");
  sima_examples::require(cfg.streams.size() <= 4, "this example supports up to four streams");
  sima_examples::require(!cfg.labels_path.empty(), "model.labels must be set");
  sima_examples::require(!cfg.insight_host.empty(), "output.insight.host must be set");
  sima_examples::require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  sima_examples::require(cfg.frames >= 0, "inference.frames must be >= 0");
  sima_examples::require(cfg.fps >= 0, "inference.fps must be >= 0");
  sima_examples::require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
                         "inference.max_inflight_per_stream must be -1 or > 0");
  sima_examples::require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
                         "inference.min_score must be between 0 and 1");
  sima_examples::require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0,
                         "inference.nms_iou must be between 0 and 1");
  sima_examples::require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  sima_examples::require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  sima_examples::require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  sima_examples::require(cfg.metadata_port_base > 0,
                         "output.insight.metadata_port_base must be > 0");
  sima_examples::require(cfg.mask_threshold >= 0.0 && cfg.mask_threshold <= 1.0,
                         "output.mask_threshold must be between 0 and 1");
  sima_examples::require(cfg.mask_alpha >= 0.0 && cfg.mask_alpha <= 1.0,
                         "output.mask_alpha must be between 0 and 1");
  sima_examples::require(cfg.min_keypoint_visibility >= 0.0 && cfg.min_keypoint_visibility <= 1.0,
                         "output.min_keypoint_visibility must be between 0 and 1");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "coco_label.txt";

  AppConfig cfg;
  cfg.streams = parse_streams(config_path);
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.codec = parse_input_codec(raw.string_or("input.codec", "h264"));
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.fps = raw.int_or("inference.fps", 0);
  cfg.max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 4);
  cfg.min_score = raw.double_or("inference.min_score", 0.30);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);
  cfg.mask_threshold = raw.double_or("output.mask_threshold", 0.50);
  cfg.mask_alpha = raw.double_or("output.mask_alpha", 0.55);
  cfg.min_keypoint_visibility = raw.double_or("output.min_keypoint_visibility", 0.30);
  cfg.save_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
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
  return palette[static_cast<std::size_t>(std::max(class_id, 0)) % palette.size()];
}

/// True when a decoded tensor carries no rows.
///
/// A frame with nothing detected decodes to a zero-row tensor. Copying one throws, because a
/// zero-byte payload has nothing to map, so treat "no rows" as an empty result.
bool tensor_is_empty(const simaai::neat::Tensor& tensor) {
  if (tensor.shape.empty()) {
    return true;
  }
  const int64_t elements = std::accumulate(tensor.shape.begin(), tensor.shape.end(), int64_t{1},
                                           std::multiplies<int64_t>());
  return elements == 0;
}

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& tensor) {
  if (tensor_is_empty(tensor)) {
    return {};
  }
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
  if (tensor_is_empty(tensor)) {
    return {};
  }
  if (tensor.dtype != simaai::neat::TensorDType::UInt8) {
    throw std::runtime_error("expected UInt8 tensor");
  }
  return tensor.copy_dense_bytes_tight();
}

bool extract_bbox_payload(const simaai::neat::Sample& sample, std::vector<std::uint8_t>& payload,
                          std::string& err) {
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : sample.fields) {
      if (extract_bbox_payload(field, payload, err)) {
        return true;
      }
    }
    err = "bundle missing BBOX field";
    return false;
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    simaai::neat::Sample tensor_sample = sample;
    tensor_sample.kind = simaai::neat::SampleKind::Tensor;
    tensor_sample.tensor = sample.tensors.front();
    tensor_sample.tensors.clear();
    return objdet::extract_bbox_payload(tensor_sample, payload, err);
  }
  return objdet::extract_bbox_payload(sample, payload, err);
}

/// Decode one segmentation sample into instances carrying a 160x160 mask each.
std::vector<SegmentationDetection> decode_segments(const simaai::neat::Sample& sample, int frame_w,
                                                   int frame_h, int max_detections) {
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    throw std::runtime_error("segmentation sample carried no tensors");
  }

  const auto decoded =
      simaai::neat::decode_segmentation(tensors, frame_w, frame_h, max_detections, false);
  std::vector<SegmentationDetection> detections;
  const std::size_t mask_bytes = static_cast<std::size_t>(kMaskGrid) * kMaskGrid;
  for (const auto& item : decoded) {
    if (!item.boxes.shape.empty() && item.boxes.shape.front() == 0) {
      continue;
    }
    const auto boxes = tensor_to_floats(item.boxes);
    const auto masks = tensor_to_u8(item.masks);
    const int count = static_cast<int>(boxes.size() / 6U);
    for (int i = 0; i < count; ++i) {
      const float* row = boxes.data() + static_cast<std::size_t>(i) * 6U;
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
      if (masks.size() >= (static_cast<std::size_t>(i) + 1U) * mask_bytes) {
        cv::Mat mask(
            kMaskGrid, kMaskGrid, CV_8UC1,
            const_cast<std::uint8_t*>(masks.data() + static_cast<std::size_t>(i) * mask_bytes));
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

/// Decode one pose sample into `Pose` records in source-frame pixel space.
///
/// `decode_pose` returns boxes as `[N, 6]` (x1, y1, x2, y2, score, class_id) and keypoints as
/// `[N, 17, 3]` (x, y, visibility), positionally aligned. Passing the frame size clamps box
/// coordinates to the frame; keypoints are emitted by the decoder unclamped, so the visibility
/// floor and the drawing code are what keep stray joints off the overlay.
std::vector<Pose> decode_poses(const simaai::neat::Sample& sample, int frame_w, int frame_h,
                               int max_poses) {
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    throw std::runtime_error("pose sample carried no tensors");
  }

  const auto decoded = simaai::neat::decode_pose(tensors, frame_w, frame_h, max_poses, false);
  std::vector<Pose> poses;
  for (const auto& item : decoded) {
    const auto boxes = tensor_to_floats(item.boxes);
    const auto keypoints = tensor_to_floats(item.keypoints);
    if (boxes.size() % 6U != 0 || keypoints.size() % (17U * 3U) != 0) {
      throw std::runtime_error("pose decode returned malformed tensor sizes");
    }
    const std::size_t box_count = boxes.size() / 6U;
    const std::size_t keypoint_count = keypoints.size() / (17U * 3U);
    if (box_count != keypoint_count) {
      throw std::runtime_error("pose decode returned " + std::to_string(box_count) + " boxes but " +
                               std::to_string(keypoint_count) + " keypoint sets");
    }

    for (std::size_t i = 0; i < box_count && poses.size() < static_cast<std::size_t>(max_poses);
         ++i) {
      const float* box = boxes.data() + i * 6U;
      Pose pose;
      pose.x1 = box[0];
      pose.y1 = box[1];
      pose.x2 = box[2];
      pose.y2 = box[3];
      pose.score = box[4];

      const float* points = keypoints.data() + i * 17U * 3U;
      for (std::size_t k = 0; k < 17U; ++k) {
        pose.keypoints[k] = Keypoint{points[k * 3U], points[k * 3U + 1U], points[k * 3U + 2U]};
      }
      poses.push_back(pose);
    }
  }
  return poses;
}

cv::Rect frame_rect_for_box(float x1, float y1, float x2, float y2, const cv::Size& frame_size) {
  const int x0 = std::clamp(static_cast<int>(std::floor(x1)), 0, frame_size.width - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(y1)), 0, frame_size.height - 1);
  const int x_end = std::clamp(static_cast<int>(std::ceil(x2)), x0 + 1, frame_size.width);
  const int y_end = std::clamp(static_cast<int>(std::ceil(y2)), y0 + 1, frame_size.height);
  return cv::Rect(x0, y0, x_end - x0, y_end - y0);
}

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
    const cv::Rect rect = frame_rect_for_box(det.x1, det.y1, det.x2, det.y2, frame_size);
    auto polygon = mask_polygon(det.mask, rect, frame_size, mask_threshold);
    if (polygon.empty()) {
      continue;
    }
    segments.push_back({"seg_" + std::to_string(segments.size() + 1),
                        class_name(labels, det.class_id), det.score, rect, std::move(polygon)});
  }
  return segments;
}

/// `data` object of a `segmentation` metadata message. Segments that do not fit the byte budget
/// are dropped lowest-confidence first and counted.
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
        {"confidence", std::round(segment.confidence * 1000.0f) / 1000.0f},
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

/// Serialize poses into the `data` object Insight's `pose-estimation` overlay consumes.
///
/// Pixel rounding and three-decimal confidence preserve overlay precision while keeping the
/// configured pose maximum within Core's logical metadata-message limit.
std::string pose_metadata_data_json(const std::vector<Pose>& poses) {
  nlohmann::json data;
  data["poses"] = nlohmann::json::array();
  int pose_index = 1;
  for (const auto& pose : poses) {
    nlohmann::json keypoints = nlohmann::json::array();
    for (std::size_t k = 0; k < pose.keypoints.size(); ++k) {
      const Keypoint& point = pose.keypoints[k];
      keypoints.push_back({{"name", kCocoKeypointNames[k]},
                           {"x", std::lround(point.x)},
                           {"y", std::lround(point.y)},
                           {"confidence", std::round(point.visibility * 1000.0f) / 1000.0f}});
    }
    data["poses"].push_back({{"id", "pose_" + std::to_string(pose_index++)},
                             {"label", "person"},
                             {"confidence", std::round(pose.score * 1000.0f) / 1000.0f},
                             {"bbox",
                              {std::lround(pose.x1), std::lround(pose.y1),
                               std::lround(std::max(0.0f, pose.x2 - pose.x1)),
                               std::lround(std::max(0.0f, pose.y2 - pose.y1))}},
                             {"keypoints", std::move(keypoints)}});
  }
  return data.dump();
}

std::vector<sima_examples::MetadataBox> build_metadata_boxes(const std::vector<objdet::Box>& boxes,
                                                             const std::vector<std::string>& labels,
                                                             int frame_w, int frame_h) {
  std::vector<sima_examples::MetadataBox> metadata_boxes;
  metadata_boxes.reserve(boxes.size());
  int object_index = 1;
  for (const auto& box : boxes) {
    int x1 = std::max(0, static_cast<int>(box.x1));
    int y1 = std::max(0, static_cast<int>(box.y1));
    int w = std::max(0, static_cast<int>(box.x2 - box.x1));
    int h = std::max(0, static_cast<int>(box.y2 - box.y1));
    if (x1 + w > frame_w)
      w = frame_w - x1;
    if (y1 + h > frame_h)
      h = frame_h - y1;

    sima_examples::MetadataBox obj;
    obj.id = "obj_" + std::to_string(object_index++);
    obj.label = class_name(labels, box.class_id);
    obj.confidence = box.score;
    obj.x = static_cast<float>(x1);
    obj.y = static_cast<float>(y1);
    obj.w = static_cast<float>(std::max(0, w));
    obj.h = static_cast<float>(std::max(0, h));
    metadata_boxes.push_back(obj);
  }
  return metadata_boxes;
}

void draw_boxes(cv::Mat& frame, const std::vector<objdet::Box>& boxes,
                const std::vector<std::string>& labels, double min_score) {
  for (const auto& box : boxes) {
    if (box.score < min_score) {
      continue;
    }
    const cv::Rect rect = frame_rect_for_box(box.x1, box.y1, box.x2, box.y2, frame.size());
    const cv::Scalar color = class_color(box.class_id);
    cv::rectangle(frame, rect, color, 2);
    cv::putText(frame,
                class_name(labels, box.class_id) + " " + std::to_string(box.score).substr(0, 4),
                cv::Point(rect.x, std::max(0, rect.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);
  }
}

void draw_segments(cv::Mat& frame, const std::vector<SegmentationDetection>& detections,
                   const std::vector<std::string>& labels, const AppConfig& cfg) {
  for (const auto& det : detections) {
    if (det.score < cfg.min_score || det.mask.empty()) {
      continue;
    }
    const cv::Rect frame_rect = frame_rect_for_box(det.x1, det.y1, det.x2, det.y2, frame.size());
    cv::Mat resized_mask = project_letterbox_mask_roi(det.mask, frame_rect, frame.size());
    cv::Mat binary_mask;
    cv::threshold(resized_mask, binary_mask, cfg.mask_threshold * 255.0, 255, cv::THRESH_BINARY);
    const cv::Scalar color = class_color(det.class_id);
    if (cv::countNonZero(binary_mask) > 0) {
      cv::Mat roi = frame(frame_rect);
      cv::Mat mask_color(frame_rect.size(), frame.type(), color);
      cv::Mat blended;
      cv::addWeighted(roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0, blended);
      blended.copyTo(roi, binary_mask);
    }
    cv::rectangle(frame, frame_rect, color, 2);
    cv::putText(frame,
                class_name(labels, det.class_id) + " " + std::to_string(det.score).substr(0, 4),
                cv::Point(frame_rect.x, std::max(0, frame_rect.y - 4)), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv::LINE_AA);
  }
}

void draw_poses(cv::Mat& frame, const std::vector<Pose>& poses, double min_visibility) {
  const cv::Scalar color(10, 249, 72);
  const cv::Scalar joint_color(255, 194, 0);
  for (const auto& pose : poses) {
    cv::rectangle(frame, frame_rect_for_box(pose.x1, pose.y1, pose.x2, pose.y2, frame.size()),
                  color, 2);

    const auto visible = [&](const Keypoint& point) {
      return point.visibility >= min_visibility && point.x >= 0.0f && point.y >= 0.0f &&
             point.x < static_cast<float>(frame.cols) && point.y < static_cast<float>(frame.rows);
    };

    for (const auto& [start, end] : kCocoSkeleton) {
      const Keypoint& a = pose.keypoints[static_cast<std::size_t>(start)];
      const Keypoint& b = pose.keypoints[static_cast<std::size_t>(end)];
      if (visible(a) && visible(b)) {
        cv::line(frame, cv::Point(static_cast<int>(a.x), static_cast<int>(a.y)),
                 cv::Point(static_cast<int>(b.x), static_cast<int>(b.y)), color, 2, cv::LINE_AA);
      }
    }
    for (const auto& point : pose.keypoints) {
      if (visible(point)) {
        cv::circle(frame, cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)), 3,
                   joint_color, -1);
      }
    }
  }
}

simaai::neat::nodes::groups::RtspDecodedInputOptions
build_source_options(const AppConfig& cfg, const std::string& url, int& fps_out, int& width_out,
                     int& height_out) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.payload_type = 96;
  probe_options.latency_ms = cfg.latency_ms;
  probe_options.rtsp_tcp = cfg.tcp;
  probe_options.debug = cfg.profile;
  (void)sima_examples::probe_rtsp_stream_info(url, probe_options, probe);

  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = url;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.payload_type = 96;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.auto_caps_from_stream = true;
  opt.codec = cfg.codec;
  if (probe.width > 0 && probe.height > 0) {
    opt.dec_width = probe.width;
    opt.dec_height = probe.height;
    if (cfg.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
      opt.fallback_h264_width = probe.width;
      opt.fallback_h264_height = probe.height;
    }
    width_out = probe.width;
    height_out = probe.height;
  }
  if (probe.fps > 0) {
    opt.source_fps = probe.fps;
    fps_out = probe.fps;
  }
  if (width_out > 0 && height_out > 0 && fps_out > 0) {
    opt.output_caps.enable = true;
    opt.output_caps.format = "NV12";
    opt.output_caps.width = width_out;
    opt.output_caps.height = height_out;
    opt.output_caps.fps = fps_out;
    opt.output_caps.memory = simaai::neat::CapsMemory::Any;
  }
  return opt;
}

bool output_caps_enabled(
    const simaai::neat::nodes::groups::RtspDecodedInputOptions::OutputCaps& caps) {
  return caps.enable || caps.width > 0 || caps.height > 0 || caps.fps > 0;
}

simaai::neat::FormatTag encoded_format_tag(simaai::neat::nodes::groups::RtspCodec codec) {
  return codec == simaai::neat::nodes::groups::RtspCodec::H265 ? simaai::neat::FormatTag::H265
                                                               : simaai::neat::FormatTag::H264;
}

simaai::neat::InputOptions
encoded_decode_input_options(simaai::neat::nodes::groups::RtspCodec codec) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = encoded_format_tag(codec);
  opt.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;
  return opt;
}

simaai::neat::InputOptions
encoded_video_input_options(simaai::neat::nodes::groups::RtspCodec codec) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Encoded;
  opt.format = encoded_format_tag(codec);
  opt.memory_policy = simaai::neat::InputMemoryPolicy::SystemMemory;
  return opt;
}

simaai::neat::Graph
build_encoded_source_graph(const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph source("rtsp_encoded_source");

  simaai::neat::nodes::groups::RtspEncodedInputOptions encoded_opt;
  encoded_opt.url = opt.url;
  encoded_opt.codec = opt.codec;
  encoded_opt.latency_ms = opt.latency_ms;
  encoded_opt.tcp = opt.tcp;
  encoded_opt.source_fps = opt.source_fps;
  if (opt.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
    encoded_opt.fallback_h264_width = opt.fallback_h264_width;
    encoded_opt.fallback_h264_height = opt.fallback_h264_height;
  }
  source.add(simaai::neat::nodes::groups::RtspEncodedInput(encoded_opt));
  return source;
}

simaai::neat::Graph
build_decode_graph(const std::string& input_name,
                   const simaai::neat::nodes::groups::RtspDecodedInputOptions& opt) {
  simaai::neat::Graph decode("decode");
  const bool use_h265 = opt.codec == simaai::neat::nodes::groups::RtspCodec::H265;

  simaai::neat::SimaDecodeOptions dec;
  dec.type = use_h265 ? simaai::neat::SimaDecodeType::H265 : simaai::neat::SimaDecodeType::H264;
  dec.sima_allocator_type = opt.sima_allocator_type;
  dec.out_format = opt.out_format;
  dec.decoder_name = opt.decoder_name;
  dec.raw_output = opt.decoder_raw_output;
  dec.next_element = opt.decoder_next_element;
  dec.dec_width = opt.dec_width;
  dec.dec_height = opt.dec_height;
  dec.dec_fps = opt.source_fps;
  dec.num_buffers = opt.num_buffers;
  dec.input_buffers = opt.decoder_input_buffers;
  dec.decoder_tuning = opt.decoder_tuning;
  dec.memory_opt = opt.decoder_memory_opt;

  decode.connect(simaai::neat::nodes::Input(input_name, encoded_decode_input_options(opt.codec)),
                 simaai::neat::nodes::SimaDecode(dec));
  if (opt.use_videoconvert) {
    decode.add(simaai::neat::nodes::VideoConvert());
  }
  if (opt.use_videoscale) {
    decode.add(simaai::neat::nodes::VideoScale());
  }
  if (output_caps_enabled(opt.output_caps)) {
    const auto& caps = opt.output_caps;
    decode.add(
        simaai::neat::nodes::CapsRaw(caps.format, caps.width, caps.height, caps.fps, caps.memory));
  }
  if (!opt.extra_fragment.empty()) {
    decode.add(simaai::neat::nodes::Custom(opt.extra_fragment));
  }
  return decode;
}

simaai::neat::Graph
build_video_sender_graph(const std::string& input_name,
                         simaai::neat::nodes::groups::RtspCodec codec,
                         const simaai::neat::nodes::groups::VideoSenderOptions& video_options) {
  simaai::neat::Graph video("video_sender");
  video.connect(simaai::neat::nodes::Input(input_name, encoded_video_input_options(codec)),
                simaai::neat::nodes::groups::VideoSender(video_options));
  return video;
}

/// Map a stream's task and declared head layout onto an on-device BoxDecode family.
simaai::neat::BoxDecodeType decode_type_for(Task task, DecodeFamily decode) {
  const bool yolo26 = decode == DecodeFamily::Yolo26;
  switch (task) {
  case Task::Pose:
    return yolo26 ? simaai::neat::BoxDecodeType::YoloV26Pose
                  : simaai::neat::BoxDecodeType::YoloV8Pose;
  case Task::Segmentation:
    return yolo26 ? simaai::neat::BoxDecodeType::YoloV26Seg
                  : simaai::neat::BoxDecodeType::YoloV8Seg;
  case Task::Detection:
  default:
    return yolo26 ? simaai::neat::BoxDecodeType::YoloV26 : simaai::neat::BoxDecodeType::YoloV8;
  }
}

std::unique_ptr<simaai::neat::Model> build_model(const AppConfig& cfg, const StreamConfig& stream,
                                                 int frame_w, int frame_h) {
  simaai::neat::Model::Options model_opt;
  model_opt.preprocess.kind = simaai::neat::InputKind::Image;
  model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
  model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  if (frame_w > 0 && frame_h > 0) {
    model_opt.preprocess.input_max_width = frame_w;
    model_opt.preprocess.input_max_height = frame_h;
  }
  model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  model_opt.decode_type = decode_type_for(stream.task, stream.decode);
  if (stream.task == Task::Pose) {
    // YOLO pose ships single-class ("person") score heads. The packaged MPK still declares the
    // 80-class detector metadata, so state the real class count rather than inheriting it.
    model_opt.num_classes = 1;
  }
  model_opt.score_threshold = cfg.min_score;
  model_opt.nms_iou_threshold = cfg.nms_iou;
  model_opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(stream.model_path, model_opt);
}

simaai::neat::RunOptions build_source_run_options() {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 4;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

/// Block, not KeepLatest: the feeder already bounds frames in flight, and Block keeps push and
/// pull strictly FIFO-paired so a pulled result belongs to the frame that was pushed.
simaai::neat::RunOptions build_model_run_options(int model_queue_depth) {
  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = std::max(1, model_queue_depth);
  run_options.overflow_policy = simaai::neat::OverflowPolicy::Block;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return run_options;
}

bool save_frames_enabled(const AppConfig& cfg) {
  return !cfg.save_dir.empty() && cfg.save_every > 0;
}

std::string stream_id_for(int stream_index) {
  return "stream" + std::to_string(stream_index);
}

std::string source_frame_name(int stream_index) {
  return "frame_" + std::to_string(stream_index);
}

/// Every model graph uses the same endpoint names; they live in separate Runs.
constexpr const char* kModelInput = "image";
constexpr const char* kModelOutput = "results";

simaai::neat::GraphLinkOptions realtime_link(int stream_index, int queue_depth) {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.queue_depth = queue_depth;
  link.stream_id = stream_id_for(stream_index);
  return link;
}

/// Wrap one stream's model in its own graph. Every stream owns a distinct model archive.
/// Copy a decoded frame out of the zero-copy decoder pool into a tensor the model input accepts.
///
/// The frame has to leave the pool: it outlives the pulled sample, and holding pool buffers
/// starves the decoder. It also has to be EV74-placed, because the model's input route is
/// device-visible and rejects CPU-backed tensors rather than silently copying them.
simaai::neat::Tensor copy_nv12_for_model(const simaai::neat::Tensor& input) {
  int width = 0;
  int height = 0;
  if (!input.is_nv12() || !sima_examples::infer_dims(input, width, height)) {
    throw std::runtime_error("expected an NV12 frame with valid dimensions");
  }
  const auto bytes = input.copy_nv12_contiguous();
  if (bytes.empty()) {
    throw std::runtime_error("NV12 copy produced no bytes");
  }
  auto out = simaai::neat::Tensor::from_vector(bytes, {static_cast<std::int64_t>(bytes.size())},
                                               simaai::neat::TensorMemory::EV74);
  std::string err;
  if (!sima_examples::init_nv12_tensor_meta(out, width, height, err)) {
    throw std::runtime_error(err);
  }
  return out;
}

/// The host pushes decoded frames in here, so the input is a plain NV12 image endpoint.
simaai::neat::InputOptions make_nv12_input_options(int width, int height, int fps) {
  simaai::neat::InputOptions opt;
  opt.payload_type = simaai::neat::PayloadType::Image;
  opt.format = simaai::neat::FormatTag::NV12;
  opt.width = width;
  opt.height = height;
  opt.depth = 1;
  opt.max_width = width;
  opt.max_height = height;
  opt.max_depth = 1;
  opt.fps_n = std::max(1, fps);
  opt.fps_d = 1;
  return opt;
}

simaai::neat::Graph build_model_graph(const AppConfig& cfg, StreamRuntime& stream) {
  simaai::neat::Graph graph("model_" + std::to_string(stream.index));
  graph.add(simaai::neat::nodes::Input(
      kModelInput, make_nv12_input_options(stream.frame_w, stream.frame_h, stream.output_fps)));
  graph.add(*stream.model);
  // At least as many buffers as frames in flight, or finished results back up inside the graph
  // and throttle the pipelining the feeder is paying for.
  graph.add(simaai::neat::nodes::Output(
      kModelOutput,
      simaai::neat::OutputOptions::EveryFrame(std::max(1, cfg.max_inflight_per_stream))));
  return graph;
}

simaai::neat::nodes::groups::VideoSenderOptions make_video_options(const AppConfig& cfg,
                                                                   int stream_index) {
  auto video_options = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(cfg.codec);
  video_options.host = cfg.insight_host;
  video_options.channel = stream_index;
  video_options.video_port_base = cfg.video_port_base;
  video_options.async = true;
  return video_options;
}

void build_stream_runtime(const AppConfig& cfg, const StreamConfig& stream_cfg,
                          const std::vector<std::string>& labels, StreamRuntime& runtime) {
  runtime.index = stream_cfg.index;
  runtime.task = stream_cfg.task;
  runtime.url = stream_cfg.url;
  runtime.model_path = stream_cfg.model_path;
  runtime.source_options = build_source_options(cfg, stream_cfg.url, runtime.output_fps,
                                                runtime.frame_w, runtime.frame_h);
  sima_examples::require(runtime.frame_w > 0 && runtime.frame_h > 0,
                         "failed to probe RTSP frame dimensions");
  sima_examples::require(runtime.output_fps > 0, "failed to probe RTSP frame rate");
  if (cfg.fps > 0) {
    runtime.output_fps = cfg.fps;
  }

  runtime.model = build_model(cfg, stream_cfg, runtime.frame_w, runtime.frame_h);
  runtime.labels = labels;
  runtime.profile.enabled = cfg.profile;
  runtime.profile.stream_index = stream_cfg.index;
  runtime.profile.task = stream_cfg.task;
  if (cfg.video_enabled) {
    runtime.video_port = make_video_options(cfg, stream_cfg.index).video_port();
  }

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = stream_cfg.index;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string metadata_err;
  runtime.metadata_sender =
      std::make_unique<simaai::neat::MetadataSender>(metadata_options, &metadata_err);
  sima_examples::require(runtime.metadata_sender->ok(), metadata_err);

  std::cout << "[stream " << stream_cfg.index << "] task=" << task_name(stream_cfg.task)
            << " decode=" << decode_name(stream_cfg.decode)
            << " model=" << fs::path(stream_cfg.model_path).filename().string()
            << " rtsp=" << stream_cfg.url << " stream=" << runtime.frame_w << "x" << runtime.frame_h
            << "@" << runtime.output_fps << " insight=" << cfg.insight_host << " video=";
  if (cfg.video_enabled) {
    std::cout << runtime.video_port;
  } else {
    std::cout << "disabled";
  }
  std::cout << " metadata=" << runtime.metadata_sender->metadata_port() << "\n";
}

/// Add one stream's `rtsp -> decode -> Output` chain, plus its video passthrough, to the shared
/// source graph.
///
/// Video stays in-graph as an encoded passthrough: Insight correlates the RTP timestamp with the
/// metadata timestamp, and re-encoding on the host would cost a copy this example does not need.
void connect_source_stream(AppRuntime& app, const AppConfig& cfg, StreamRuntime& stream) {
  auto source = build_encoded_source_graph(stream.source_options);
  auto decoder = build_decode_graph("decode_h264", stream.source_options);

  simaai::neat::Graph frames("frames_" + std::to_string(stream.index));
  // A few buffers, and dropping: Insight can only draw an overlay on a frame it has metadata
  // for, so a frame skipped here shows up as a blink in the viewer. Depth absorbs a brief
  // feeder hiccup; dropping still sheds load if the model genuinely cannot keep up.
  auto frame_output = simaai::neat::OutputOptions::EveryFrame(4);
  frame_output.drop = true;
  frames.add(simaai::neat::nodes::Output(source_frame_name(stream.index), frame_output));

  if (cfg.video_enabled) {
    auto encoded_branch = simaai::neat::graphs::Branch("encoded", {"decode_h264", "video_h264"});
    app.source_graph.connect(source, encoded_branch);
    app.source_graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3));

    const auto video_options = make_video_options(cfg, stream.index);
    app.source_graph.connect(encoded_branch,
                             build_video_sender_graph("video_h264", cfg.codec, video_options),
                             realtime_link(stream.index, 3));
  } else {
    app.source_graph.connect(source, decoder, realtime_link(stream.index, 3));
  }
  app.source_graph.connect(decoder, frames);
}

FrameResults decode_results(const AppConfig& cfg, const StreamRuntime& stream,
                            const simaai::neat::Sample& sample) {
  FrameResults results;
  switch (stream.task) {
  case Task::Segmentation:
    results.segments = decode_segments(sample, stream.frame_w, stream.frame_h, cfg.max_detections);
    break;
  case Task::Pose:
    results.poses = decode_poses(sample, stream.frame_w, stream.frame_h, cfg.max_detections);
    break;
  case Task::Detection:
  default: {
    std::vector<std::uint8_t> payload;
    std::string err;
    if (!extract_bbox_payload(sample, payload, err)) {
      throw std::runtime_error("stream " + std::to_string(stream.index) +
                               " bbox extract failed: " + err);
    }
    results.boxes = objdet::parse_boxes_strict(payload, stream.frame_w, stream.frame_h,
                                               cfg.max_detections, false);
    break;
  }
  }
  return results;
}

/// Publish one message on this stream's Insight channel, typed by the stream's task. Returns how
/// many segments the byte budget dropped, which is always zero outside segmentation.
int send_metadata(const AppConfig& cfg, StreamRuntime& stream, const FrameStamp& stamp,
                  const FrameResults& results) {
  std::string data_json;
  int dropped = 0;
  switch (stream.task) {
  case Task::Segmentation: {
    auto encoded = encode_segments(build_metadata_segments(results.segments, stream.labels,
                                                           cv::Size(stream.frame_w, stream.frame_h),
                                                           cfg.mask_threshold));
    data_json = std::move(encoded.data_json);
    dropped = encoded.dropped;
    break;
  }
  case Task::Pose:
    data_json = pose_metadata_data_json(results.poses);
    break;
  case Task::Detection:
  default:
    data_json = sima_examples::metadata_boxes_data_json(
        "objects",
        build_metadata_boxes(results.boxes, stream.labels, stream.frame_w, stream.frame_h));
    break;
  }

  const int64_t timestamp_ms = stamp.pts_ns >= 0 ? stamp.pts_ns / 1'000'000 : -1;
  const std::string frame_id = stamp.frame_id >= 0 ? std::to_string(stamp.frame_id) : "";
  std::string err;
  if (!stream.metadata_sender->send_metadata(metadata_type(stream.task), data_json, timestamp_ms,
                                             frame_id, &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
  return dropped;
}

void maybe_save_debug_frame(const AppConfig& cfg, const StreamRuntime& stream,
                            const FrameResults& results) {
  if (!save_frames_enabled(cfg) || stream.processed % cfg.save_every != 0) {
    return;
  }
  cv::Mat frame;
  {
    std::lock_guard<std::mutex> lock(stream.debug_frame_mutex);
    if (!stream.latest_debug_frame || stream.latest_debug_frame->empty()) {
      return;
    }
    frame = stream.latest_debug_frame->clone();
  }
  switch (stream.task) {
  case Task::Segmentation:
    draw_segments(frame, results.segments, stream.labels, cfg);
    break;
  case Task::Pose:
    draw_poses(frame, results.poses, cfg.min_keypoint_visibility);
    break;
  case Task::Detection:
  default:
    draw_boxes(frame, results.boxes, stream.labels, cfg.min_score);
    break;
  }

  const auto out_path = cfg.save_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                        std::to_string(stream.processed) + ".jpg");
  if (!cv::imwrite(out_path.string(), frame)) {
    std::cerr << "[warn] failed to write output frame: " << out_path.string() << "\n";
  }
}

void process_output_sample(const AppConfig& cfg, StreamRuntime& stream,
                           const simaai::neat::Sample& sample, const FrameStamp& stamp,
                           double result_pull_ms) {
  if (cfg.frames > 0 && stream.processed >= cfg.frames) {
    return;
  }

  const FrameResults results = decode_results(cfg, stream, sample);

  ++stream.processed;
  if (stream.processed <= cfg.warmup_frames) {
    return;
  }

  const double metadata_start = sima_examples::time_ms();
  stream.dropped_segments += send_metadata(cfg, stream, stamp, results);
  const double metadata_end = sima_examples::time_ms();
  maybe_save_debug_frame(cfg, stream, results);
  stream.profile.add(result_pull_ms, metadata_end - metadata_start,
                     static_cast<int>(results.size()));
}

/// Feeder: pull this stream's decoded frames from the shared source Run and hand them to its
/// model, never letting more than `max_inflight_per_stream` frames sit inside the model graph.
///
/// The gate is the real backpressure. `OverflowPolicy::Block` does not bound in-graph buffering:
/// push() returns as soon as the frame lands in the graph's internal edge queue, so a model
/// slower than its camera accumulates seconds of latency while still reporting the source rate.
/// Frames that arrive while the gate is closed are dropped, which is what a live camera wants.
void run_stream_feeder(AppRuntime& app, const AppConfig& cfg, StreamRuntime& stream) {
  constexpr int kPullTimeoutMs = 50;
  const std::string frame_name = source_frame_name(stream.index);
  const int depth = std::max(1, cfg.max_inflight_per_stream);

  while (g_stop_requested == 0 && !stream.closed) {
    simaai::neat::Sample frame_sample;
    simaai::neat::PullError pull_error;
    const auto status = app.source_run.pull(frame_name, kPullTimeoutMs, frame_sample, &pull_error);
    if (status == simaai::neat::PullStatus::Timeout) {
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      std::cerr << "[ERR] stream " << stream.index << " source pull: " << pull_error.message
                << "\n";
      break;
    }
    // Wait for a slot rather than discarding the frame: every frame dropped here is a frame
    // Insight renders with no overlay. Bounded, so a wedged model cannot hang the feeder.
    for (int waited = 0; stream.in_flight.load() >= depth && waited < 50 && g_stop_requested == 0;
         ++waited) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (stream.in_flight.load() >= depth) {
      continue;
    }

    const auto tensors = simaai::neat::tensors_from_sample(frame_sample, false);
    if (tensors.empty()) {
      continue;
    }
    simaai::neat::Tensor frame;
    try {
      frame = copy_nv12_for_model(tensors.front());
    } catch (const std::exception& e) {
      std::cerr << "[warn] stream " << stream.index << " frame copy: " << e.what() << "\n";
      continue;
    }
    if (save_frames_enabled(cfg)) {
      cv::Mat bgr;
      std::string err;
      if (sima_examples::nv12_to_bgr(frame, bgr, err)) {
        std::lock_guard<std::mutex> lock(stream.debug_frame_mutex);
        stream.latest_debug_frame = std::move(bgr);
      }
    }

    // Recorded before the push: under Block the result can come back the instant push()
    // returns, and a consumer that found the queue empty would have nothing to pair it with.
    {
      std::lock_guard<std::mutex> lock(stream.pending_mutex);
      stream.pending.push_back(FrameStamp{frame_sample.pts_ns, frame_sample.frame_id});
    }
    stream.in_flight.fetch_add(1);
    if (!stream.model_run.push(kModelInput, simaai::neat::TensorList{frame})) {
      stream.in_flight.fetch_sub(1);
      std::lock_guard<std::mutex> lock(stream.pending_mutex);
      if (!stream.pending.empty()) {
        stream.pending.pop_back();
      }
      break;
    }
  }
  stream.feed_done.store(true);
}

/// Consumer: pull this stream's results back and publish them. Split from the feeder so the
/// model holds several frames at once; a single thread doing push-then-pull would serialise
/// every stage and cap the stream at its slowest one.
void run_stream_consumer(const AppConfig& cfg, StreamRuntime& stream) {
  constexpr int kPullTimeoutMs = 50;

  while (g_stop_requested == 0 && (cfg.frames <= 0 || stream.processed < cfg.frames)) {
    const double pull_start = sima_examples::time_ms();
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    const auto status = stream.model_run.pull(kModelOutput, kPullTimeoutMs, sample, &pull_error);
    const double pull_end = sima_examples::time_ms();

    if (status == simaai::neat::PullStatus::Timeout) {
      // Nothing in flight and the feeder has stopped: there is nothing left to wait for.
      if (stream.feed_done.load() && stream.in_flight.load() <= 0) {
        break;
      }
      continue;
    }
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      std::cerr << "[ERR] stream " << stream.index << " model pull: " << pull_error.message << "\n";
      break;
    }
    stream.in_flight.fetch_sub(1);
    FrameStamp stamp;
    {
      std::lock_guard<std::mutex> lock(stream.pending_mutex);
      if (!stream.pending.empty()) {
        stamp = stream.pending.front();
        stream.pending.pop_front();
      }
    }
    process_output_sample(cfg, stream, sample, stamp, pull_end - pull_start);
  }
  stream.closed = true;
}

void run_app(const AppConfig& cfg) {
  g_stop_requested = 0;
  auto previous_sigint = std::signal(SIGINT, request_stop);
  if (cfg.profile) {
    setenv("SIMA_GST_ELEMENT_TIMINGS", "1", 0);
    setenv("SIMA_GST_FLOW_DEBUG", "1", 0);
    setenv("SIMA_GST_BOUNDARY_PROBES", "1", 0);
  }
  if (save_frames_enabled(cfg)) {
    fs::create_directories(cfg.save_dir);
  }

  const auto labels = load_labels(cfg.labels_path);
  AppRuntime app;
  app.streams.reserve(cfg.streams.size());
  for (const auto& stream_cfg : cfg.streams) {
    app.streams.push_back(std::make_unique<StreamRuntime>());
    StreamRuntime& stream = *app.streams.back();
    build_stream_runtime(cfg, stream_cfg, labels, stream);
    connect_source_stream(app, cfg, stream);
    stream.model_graph = build_model_graph(cfg, stream);
    stream.model_run =
        stream.model_graph.build(build_model_run_options(cfg.max_inflight_per_stream));
  }

  if (cfg.profile) {
    std::cout << "Backend:\n" << app.source_graph.describe_backend() << "\n";
  }

  // Built last, and only once every model is loaded: build() starts the pipeline, so building the
  // sources earlier lets the first camera stream while the remaining models load, filling its
  // edge queue and killing it before anything pulls.
  app.source_run = app.source_graph.build(build_source_run_options());

  std::vector<std::thread> workers;
  workers.reserve(app.streams.size() * 2);
  for (auto& stream : app.streams) {
    StreamRuntime* rt = stream.get();
    workers.emplace_back([&app, &cfg, rt] { run_stream_feeder(app, cfg, *rt); });
    workers.emplace_back([&cfg, rt] { run_stream_consumer(cfg, *rt); });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (auto& stream : app.streams) {
    stream->model_run.close();
  }
  app.source_run.close();

  for (auto& stream : app.streams) {
    stream->profile.flush();
    std::cout << "[stream " << stream->index << "] task=" << task_name(stream->task)
              << " processed=" << stream->processed;
    if (stream->task == Task::Segmentation) {
      std::cout << " dropped_segments=" << stream->dropped_segments;
    }
    std::cout << "\n";
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
      std::string tasks;
      for (const auto& stream : cfg.streams) {
        if (!tasks.empty()) {
          tasks += ",";
        }
        tasks += task_name(stream.task);
      }
      std::cout << "Config validated: " << cli.config_path << " (streams=" << cfg.streams.size()
                << ", tasks=" << tasks
                << ", max_inflight_per_stream=" << cfg.max_inflight_per_stream << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
