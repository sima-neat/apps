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
#include "support/runtime/pull_status.h"

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
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kMaxStreams = 4;
constexpr int kPullTimeoutMs = 50;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

struct AppConfig {
  std::string model_path;
  std::string labels_path;
  std::vector<std::string> rtsp_urls;
  simaai::neat::nodes::groups::RtspCodec codec = simaai::neat::nodes::groups::RtspCodec::H264;
  bool tcp = true;
  int latency_ms = 100;
  int frames = 0;
  int max_inflight_per_stream = 4;
  int max_inflight_total = 16;
  double min_score = 0.30;
  int max_detections = 300;
  std::string insight_host;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  fs::path debug_dir;
  int save_every = 0;
};

struct Stream {
  int index = 0;
  std::string url;
  int width = 0;
  int height = 0;
  int fps = 0;
  std::unique_ptr<simaai::neat::MetadataSender> metadata_sender;
  int processed = 0;
  cv::Mat latest_frame;
};

// One detection in source-frame pixels; a positive angle turns the box clockwise.
struct OrientedBox {
  double cx = 0.0;
  double cy = 0.0;
  double width = 0.0;
  double height = 0.0;
  double angle_radians = 0.0;
  double score = 0.0;
  int class_id = 0;
};

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

simaai::neat::nodes::groups::RtspCodec parse_codec(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (value == "h264" || value == "avc" || value == "h.264") {
    return simaai::neat::nodes::groups::RtspCodec::H264;
  }
  if (value == "h265" || value == "hevc" || value == "h.265") {
    return simaai::neat::nodes::groups::RtspCodec::H265;
  }
  throw std::runtime_error("input.codec must be h264/avc or h265/hevc");
}

// ScalarConfig reads scalars only, so the `streams:` list is read here.
std::vector<std::string> parse_streams(const fs::path& config_path) {
  std::ifstream input(config_path);
  std::vector<std::string> streams;
  bool in_streams = false;
  for (std::string line; std::getline(input, line);) {
    const std::string text = sima_examples::trim_copy(line.substr(0, line.find(" #")));
    if (text.empty() || text.front() == '#') {
      continue;
    }
    if (!in_streams) {
      in_streams = text == "streams:";
      continue;
    }
    if (text.front() != '-') {
      break;
    }
    std::string url = sima_examples::trim_copy(text.substr(1));
    if (url.size() >= 2 && (url.front() == '"' || url.front() == '\'') &&
        url.back() == url.front()) {
      url = url.substr(1, url.size() - 2);
    }
    if (sima_examples::trim_copy(url).empty()) {
      throw std::runtime_error("streams[" + std::to_string(streams.size()) +
                               "] must be a non-empty string");
    }
    streams.push_back(url);
  }
  if (streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  return streams;
}

void validate_config(const AppConfig& cfg) {
  using sima_examples::require;
  require(!cfg.model_path.empty(), "model.path must be set");
  require(!cfg.labels_path.empty(), "model.labels must be set");
  require(cfg.rtsp_urls.size() <= kMaxStreams, "streams supports up to four entries");
  require(!cfg.insight_host.empty(), "output.insight.host must be set");
  require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0");
  require(cfg.frames >= 0, "inference.frames must be >= 0");
  require(cfg.max_inflight_per_stream == -1 || cfg.max_inflight_per_stream > 0,
          "inference.max_inflight_per_stream must be -1 or > 0");
  require(cfg.max_inflight_total == -1 || cfg.max_inflight_total > 0,
          "inference.max_inflight_total must be -1 or > 0");
  require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
          "inference.min_score must be between 0 and 1");
  require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  require(cfg.video_port_base > 0, "output.insight.video_port_base must be > 0");
  require(cfg.metadata_port_base > 0, "output.insight.metadata_port_base must be > 0");
  require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& config_path) {
  const auto raw = sima_examples::ScalarConfig::load(config_path);
  const auto default_labels =
      fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" / "dota_label.txt";

  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  cfg.rtsp_urls = parse_streams(config_path);
  cfg.codec = parse_codec(raw.string_or("input.codec", "h264"));
  cfg.tcp = raw.bool_or("input.tcp", true);
  cfg.latency_ms = raw.int_or("input.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.max_inflight_per_stream = raw.int_or("inference.max_inflight_per_stream", 4);
  cfg.max_inflight_total = raw.int_or("inference.max_inflight_total", 16);
  cfg.min_score = raw.double_or("inference.min_score", 0.30);
  cfg.max_detections = raw.int_or("inference.max_detections", 300);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.debug_dir = raw.string_or("output.debug_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  validate_config(cfg);
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream input(labels_path);
  if (!input.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }
  std::vector<std::string> labels;
  for (std::string line; std::getline(input, line);) {
    line = sima_examples::trim_copy(line);
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

bool save_frames_enabled(const AppConfig& cfg) {
  return !cfg.debug_dir.empty() && cfg.save_every > 0;
}

simaai::neat::nodes::groups::VideoSenderOptions video_options(const AppConfig& cfg, int index) {
  auto opt = simaai::neat::nodes::groups::VideoSenderOptions::Passthrough(cfg.codec);
  opt.host = cfg.insight_host;
  opt.channel = index;
  opt.video_port_base = cfg.video_port_base;
  opt.async = false;
  return opt;
}

Stream open_stream(const AppConfig& cfg, int index, const std::string& url) {
  Stream stream;
  stream.index = index;
  stream.url = url;

  sima_examples::RtspProbeOptions probe_options;
  probe_options.rtsp_tcp = cfg.tcp;
  sima_examples::RtspStreamInfo probe;
  sima_examples::require(sima_examples::probe_rtsp_stream_info(url, probe_options, probe) &&
                             probe.fps > 0,
                         "failed to probe RTSP frame size and rate: " + url);
  stream.width = probe.width;
  stream.height = probe.height;
  stream.fps = probe.fps;

  simaai::neat::MetadataSenderOptions metadata_options;
  metadata_options.host = cfg.insight_host;
  metadata_options.channel = index;
  metadata_options.metadata_port_base = cfg.metadata_port_base;
  std::string err;
  stream.metadata_sender = std::make_unique<simaai::neat::MetadataSender>(metadata_options, &err);
  sima_examples::require(stream.metadata_sender->ok(), err);

  std::cout << "[stream " << index << "] rtsp=" << url << " stream=" << stream.width << "x"
            << stream.height << "@" << stream.fps << " insight=" << cfg.insight_host
            << " video=" << video_options(cfg, index).video_port()
            << " metadata=" << stream.metadata_sender->metadata_port() << std::endl;
  return stream;
}

std::unique_ptr<simaai::neat::Model> build_model(const AppConfig& cfg) {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::NV12;
  // COCO_YOLO letterboxes, which keeps one scale on both axes so rotated boxes map back.
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.decode_type = simaai::neat::BoxDecodeType::YoloV26Obb;
  opt.score_threshold = cfg.min_score;
  // YOLO26 needs no NMS, and the oriented-box decoder requires it disabled.
  opt.nms_iou_threshold = 0.0;
  opt.top_k = cfg.max_detections;
  return std::make_unique<simaai::neat::Model>(cfg.model_path, opt);
}

simaai::neat::nodes::groups::RtspEncodedInputOptions source_options(const AppConfig& cfg,
                                                                    const Stream& stream) {
  simaai::neat::nodes::groups::RtspEncodedInputOptions opt;
  opt.url = stream.url;
  opt.codec = cfg.codec;
  opt.latency_ms = cfg.latency_ms;
  opt.tcp = cfg.tcp;
  opt.source_fps = stream.fps;
  if (cfg.codec == simaai::neat::nodes::groups::RtspCodec::H264) {
    opt.fallback_h264_width = stream.width;
    opt.fallback_h264_height = stream.height;
  }
  return opt;
}

simaai::neat::SimaDecodeOptions decoder_options(const AppConfig& cfg, const Stream& stream) {
  simaai::neat::SimaDecodeOptions opt;
  opt.type = cfg.codec == simaai::neat::nodes::groups::RtspCodec::H265
                 ? simaai::neat::SimaDecodeType::H265
                 : simaai::neat::SimaDecodeType::H264;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.raw_output = true;
  opt.dec_width = stream.width;
  opt.dec_height = stream.height;
  opt.dec_fps = stream.fps;
  return opt;
}

std::string stream_id(int index) {
  return "stream" + std::to_string(index);
}

// Keeps each stream's newest frames and tags them so results find their stream.
simaai::neat::GraphLinkOptions detector_link(const AppConfig& cfg, int index) {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = cfg.max_inflight_per_stream;
  link.max_inflight_total = cfg.max_inflight_total;
  link.stream_id = stream_id(index);
  return link;
}

// Keeps only the newest frame, so a slow consumer drops frames instead of slowing the stream.
simaai::neat::GraphLinkOptions latest_frame_link() {
  simaai::neat::GraphLinkOptions link;
  link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

std::string debug_frame_output(int index) {
  return "debug_frame_" + std::to_string(index);
}

// Feeds every stream into one shared detector; Core renders a single pipeline.
simaai::neat::Graph build_graph(const AppConfig& cfg, const std::vector<Stream>& streams,
                                simaai::neat::Model& model) {
  auto detector_input = model.input_appsrc_options(false);
  detector_input.block = true;
  simaai::neat::Graph detector("detector");
  detector.connect(simaai::neat::nodes::Input("detector_frame", detector_input), model);

  simaai::neat::Graph graph;
  for (const auto& stream : streams) {
    simaai::neat::Graph source("rtsp_encoded_source");
    source.add(simaai::neat::nodes::groups::RtspEncodedInput(source_options(cfg, stream)));

    simaai::neat::Graph decoder("decode");
    decoder.add(simaai::neat::nodes::SimaDecode(decoder_options(cfg, stream)));
    decoder.add(simaai::neat::nodes::CapsRaw("NV12", stream.width, stream.height, stream.fps,
                                             simaai::neat::CapsMemory::Any));
    // Named so connect() can match the shared model's "detector_frame" input.
    decoder.add(simaai::neat::nodes::Output("detector_frame"));

    simaai::neat::Graph video("video_sender");
    video.add(simaai::neat::nodes::groups::VideoSender(video_options(cfg, stream.index)));

    // A plain source link lets Core tee the encoded stream to the decoder and the sender.
    graph.connect(source, decoder);
    graph.connect(decoder, detector, detector_link(cfg, stream.index));
    graph.connect(source, video, latest_frame_link());
    if (save_frames_enabled(cfg)) {
      simaai::neat::Graph frames("debug_frame");
      frames.connect(simaai::neat::nodes::Input("debug_frame"),
                     simaai::neat::nodes::Output(debug_frame_output(stream.index),
                                                 simaai::neat::OutputOptions::EveryFrame(4)));
      graph.connect(decoder, frames, latest_frame_link());
    }
  }

  simaai::neat::Graph detections("detections");
  detections.add(
      simaai::neat::nodes::Output("detections", simaai::neat::OutputOptions::EveryFrame(4)));
  graph.connect(detector, detections);
  return graph;
}

simaai::neat::RunOptions run_options() {
  simaai::neat::RunOptions opt;
  opt.preset = simaai::neat::RunPreset::Realtime;
  opt.queue_depth = 4;
  opt.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  opt.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  return opt;
}

// Reads the decoder's oriented boxes, already mapped to source-frame pixels.
std::vector<OrientedBox> decode_boxes(const simaai::neat::Sample& sample) {
  const auto tensors = simaai::neat::tensors_from_sample(sample, false);
  if (tensors.empty()) {
    throw std::runtime_error("detection sample carried no tensors");
  }
  const auto result = simaai::neat::decode_obb(tensors).front();
  std::vector<OrientedBox> boxes(static_cast<std::size_t>(result.scores.shape.at(0)));
  // A frame without detections decodes to empty tensors, which have no data to read.
  if (boxes.empty()) {
    return boxes;
  }
  // decode_obb returns dense CPU tensors: geometry is [N, 5], scores and class ids are [N].
  const float* geometry = result.geometry.data_ptr<float>();
  const float* scores = result.scores.data_ptr<float>();
  const int32_t* class_ids = result.class_ids.data_ptr<int32_t>();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const float* row = geometry + i * 5;
    boxes[i] = {row[0], row[1], row[2], row[3], row[4], scores[i], class_ids[i]};
  }
  return boxes;
}

std::string label_for(const std::vector<std::string>& labels, int class_id) {
  return class_id >= 0 && class_id < static_cast<int>(labels.size()) ? labels[class_id] : "unknown";
}

// Builds one Insight `object-detection` entry. `oriented_bbox` is what Insight draws. `bbox`
// is the box's upright extent, which places the label and is what viewers without
// oriented-box support draw instead. Values are rounded so that a frame with hundreds of
// boxes stays under the 64 KB limit of one metadata message.
nlohmann::json insight_object(int index, const OrientedBox& box,
                              const std::vector<std::string>& labels) {
  const double cos = std::abs(std::cos(box.angle_radians));
  const double sin = std::abs(std::sin(box.angle_radians));
  const double extent_w = box.width * cos + box.height * sin;
  const double extent_h = box.width * sin + box.height * cos;
  // Rounds halves to even, like the Python implementation, so both publish the same values.
  const auto round_to = [](double value, double scale) {
    return std::nearbyint(value * scale) / scale;
  };
  const auto px = [&](double value) { return round_to(value, 10.0); };
  return {
      {"id", "obj_" + std::to_string(index)},
      {"label", label_for(labels, box.class_id)},
      {"confidence", round_to(box.score, 1000.0)},
      {"bbox", {px(box.cx - extent_w / 2), px(box.cy - extent_h / 2), px(extent_w), px(extent_h)}},
      {"oriented_bbox",
       {px(box.cx), px(box.cy), px(box.width), px(box.height),
        round_to(box.angle_radians, 10000.0)}}};
}

void send_metadata(Stream& stream, const simaai::neat::Sample& sample,
                   const std::vector<OrientedBox>& boxes, const std::vector<std::string>& labels) {
  nlohmann::json objects = nlohmann::json::array();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    objects.push_back(insight_object(static_cast<int>(i) + 1, boxes[i], labels));
  }
  const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
  const std::string frame_id = sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "";
  std::string err;
  if (!stream.metadata_sender->send_metadata("object-detection",
                                             nlohmann::json{{"objects", objects}}.dump(),
                                             timestamp_ms, frame_id, &err)) {
    std::cerr << "[warn] stream " << stream.index << " metadata send failed: " << err << "\n";
  }
}

void draw_boxes(cv::Mat& frame, const std::vector<OrientedBox>& boxes,
                const std::vector<std::string>& labels) {
  const cv::Scalar green(0, 255, 0);
  for (const auto& box : boxes) {
    const cv::RotatedRect rect(cv::Point2f(box.cx, box.cy), cv::Size2f(box.width, box.height),
                               static_cast<float>(box.angle_radians * 180.0 / M_PI));
    std::array<cv::Point2f, 4> corners_f;
    rect.points(corners_f.data());
    std::vector<cv::Point> corners;
    for (const auto& corner : corners_f) {
      corners.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::polylines(frame, corners, true, green, 2);
    const cv::Rect extent = cv::boundingRect(corners);
    const std::string text =
        cv::format("%s %.2f", label_for(labels, box.class_id).c_str(), box.score);
    cv::putText(frame, text, cv::Point(extent.x, std::max(0, extent.y - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, green, 1);
  }
}

void save_debug_frame(const AppConfig& cfg, const Stream& stream,
                      const std::vector<OrientedBox>& boxes,
                      const std::vector<std::string>& labels) {
  if (stream.latest_frame.empty() || stream.processed % cfg.save_every != 0) {
    return;
  }
  cv::Mat frame = stream.latest_frame.clone();
  draw_boxes(frame, boxes, labels);
  const auto path = cfg.debug_dir / ("stream_" + std::to_string(stream.index) + "_frame_" +
                                     std::to_string(stream.processed) + ".jpg");
  if (!cv::imwrite(path.string(), frame)) {
    std::cerr << "[warn] failed to write debug frame: " << path.string() << "\n";
  }
}

// Keeps each stream's newest decoded frame for the next saved detection.
void drain_debug_frames(simaai::neat::Run& run, std::vector<Stream>& streams) {
  for (auto& stream : streams) {
    const std::string output_name = debug_frame_output(stream.index);
    for (;;) {
      simaai::neat::Sample sample;
      simaai::neat::PullError pull_error;
      const auto status = run.pull(output_name, 0, sample, &pull_error);
      if (!sima_examples::pull_status_has_sample(status, output_name, pull_error,
                                                 run.last_error())) {
        break;
      }
      const auto tensors = simaai::neat::tensors_from_sample(sample, false);
      std::string err;
      if (!tensors.empty() &&
          !sima_examples::nv12_to_bgr(tensors.front(), stream.latest_frame, err)) {
        std::cerr << "[warn] failed to convert debug frame: " << err << "\n";
      }
    }
  }
}

Stream& stream_for(const simaai::neat::Sample& sample, std::vector<Stream>& streams) {
  for (auto& stream : streams) {
    if (sample.stream_id == stream_id(stream.index)) {
      return stream;
    }
  }
  throw std::runtime_error("detection sample has an unknown stream id: '" + sample.stream_id + "'");
}

bool all_streams_done(const std::vector<Stream>& streams, int frame_limit) {
  return frame_limit > 0 && std::all_of(streams.begin(), streams.end(), [&](const Stream& s) {
           return s.processed >= frame_limit;
         });
}

void run_app(const AppConfig& cfg) {
  if (save_frames_enabled(cfg)) {
    fs::create_directories(cfg.debug_dir);
  }

  const auto labels = load_labels(cfg.labels_path);
  const auto model = build_model(cfg);
  std::vector<Stream> streams;
  for (std::size_t index = 0; index < cfg.rtsp_urls.size(); ++index) {
    streams.push_back(open_stream(cfg, static_cast<int>(index), cfg.rtsp_urls[index]));
  }
  auto graph = build_graph(cfg, streams, *model);

  auto run = graph.build(run_options());
  const auto previous_sigint = std::signal(SIGINT, request_stop);
  while (g_stop_requested == 0 && !all_streams_done(streams, cfg.frames)) {
    if (save_frames_enabled(cfg)) {
      drain_debug_frames(run, streams);
    }
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &pull_error);
    if (!sima_examples::pull_status_has_sample(status, "detections", pull_error,
                                               run.last_error())) {
      continue;
    }
    Stream& stream = stream_for(sample, streams);
    if (cfg.frames > 0 && stream.processed >= cfg.frames) {
      continue;
    }
    const auto boxes = decode_boxes(sample);
    ++stream.processed;
    send_metadata(stream, sample, boxes, labels);
    if (save_frames_enabled(cfg)) {
      save_debug_frame(cfg, stream, boxes, labels);
    }
  }
  std::signal(SIGINT, previous_sigint);
  run.close();

  for (const auto& stream : streams) {
    std::cout << "[stream " << stream.index << "] processed=" << stream.processed << "\n";
  }
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
      std::cout << "Config validated: " << cli.config_path.string()
                << " (streams=" << cfg.rtsp_urls.size() << ")\n";
      return 0;
    }
    run_app(cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
