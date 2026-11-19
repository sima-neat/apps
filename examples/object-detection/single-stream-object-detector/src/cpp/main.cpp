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

// Detects objects in one RTSP, HTTP MJPEG, or USB camera stream with YOLO26, shown in Insight.
// The decoded frames feed the model. Insight receives the video and every frame's boxes.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kPullTimeoutMs = 100;
// RTP/JPEG (RFC 2435) carries a USB camera's JPEG to Insight in 8-pixel blocks, 2040 at most.
constexpr int kRtpJpegMaxSide = 2040;

struct AppConfig {
  fs::path model_path;
  fs::path labels_path;
  std::string input_type; // rtsp, http, or usb.
  std::string source;
  std::string codec; // mjpeg for HTTP and USB, which carry only MJPEG.
  int width = 0;
  int height = 0;
  int fps = 0;             // 0 reads the RTSP stream's frame rate with a probe.
  bool rtsp_tcp = false;   // RTSP only.
  int rtsp_latency_ms = 0; // RTSP only.
  int frames = 0;          // 0 runs until a stop signal.
  double min_score = 0.0;
  double nms_iou = 0.0;
  int max_detections = 0;
  std::string insight_host;
  int video_port_base = 0;
  int metadata_port_base = 0;
  bool ssl_strict = true; // HTTP only.
};

struct CliOptions {
  fs::path config_path;
  bool validate_config_only = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  options.config_path = fs::exists(adjacent_config)
                            ? adjacent_config
                            : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
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

AppConfig load_app_config(const fs::path& path) {
  const auto config = sima_examples::Config::load(path);
  AppConfig cfg;
  cfg.input_type = config.string("input.type");
  if (cfg.input_type != "rtsp" && cfg.input_type != "http" && cfg.input_type != "usb") {
    throw std::runtime_error("input.type must be rtsp, http, or usb");
  }
  const bool rtsp = cfg.input_type == "rtsp";
  // Only the keys of input.type are read, so reject_unread_keys() fails on another type's.
  cfg.model_path = fs::absolute(path).parent_path() / config.string("model.path");
  cfg.labels_path = config.string("model.labels");
  cfg.source = config.string("input.source");
  cfg.codec = rtsp ? config.string("input.codec") : "mjpeg";
  cfg.width = config.integer("input.width");
  cfg.height = config.integer("input.height");
  cfg.fps = config.integer("input.fps");
  if (rtsp) {
    cfg.rtsp_tcp = config.boolean("input.rtsp.tcp");
    cfg.rtsp_latency_ms = config.integer("input.rtsp.latency_ms");
  }
  cfg.frames = config.integer("inference.frames");
  cfg.min_score = config.number("inference.min_score");
  cfg.nms_iou = config.number("inference.nms_iou");
  cfg.max_detections = config.integer("inference.max_detections");
  cfg.insight_host = config.string("output.insight.host");
  cfg.video_port_base = config.integer("output.insight.video_port_base");
  cfg.metadata_port_base = config.integer("output.insight.metadata_port_base");
  if (cfg.input_type == "http") {
    cfg.ssl_strict = config.boolean("app.ssl_strict");
  }
  config.reject_unread_keys();
  if (cfg.codec != "h264" && cfg.codec != "h265" && cfg.codec != "mjpeg") {
    throw std::runtime_error("input.codec must be h264, h265, or mjpeg");
  }
  if (!rtsp && cfg.fps <= 0) {
    throw std::runtime_error("input.fps must be above 0 for an HTTP or USB source");
  }
  if (cfg.frames < 0) {
    throw std::runtime_error("inference.frames must be 0 or more");
  }
  if (cfg.max_detections <= 0) {
    throw std::runtime_error("inference.max_detections must be above 0");
  }
  // Core candidate: Core rejects a larger JPEG only when the video sender's caps fail to
  // negotiate, with no message that names the RTP/JPEG limit.
  const auto rtp_jpeg_side = [](int side) {
    return side > 0 && side <= kRtpJpegMaxSide && side % 8 == 0;
  };
  if (cfg.input_type == "usb" && !(rtp_jpeg_side(cfg.width) && rtp_jpeg_side(cfg.height))) {
    throw std::runtime_error("input.width and input.height of a USB camera must be multiples of "
                             "8 and at most " +
                             std::to_string(kRtpJpegMaxSide) + " (RTP/JPEG)");
  }
  return cfg;
}

groups::RtspCodec rtsp_codec(const AppConfig& cfg) {
  return cfg.codec == "h264"   ? groups::RtspCodec::H264
         : cfg.codec == "h265" ? groups::RtspCodec::H265
                               : groups::RtspCodec::MJPEG;
}

// YOLO26 on the decoded frames: NV12, or NV16 from a 4:2:2 USB camera.
neat::Model::Options model_options(const AppConfig& cfg, neat::FormatTag frame_format) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = frame_format == neat::FormatTag::NV16
                                                      ? neat::PreprocessColorFormat::NV16
                                                      : neat::PreprocessColorFormat::NV12;
  options.preprocess.input_max_width = cfg.width;
  options.preprocess.input_max_height = cfg.height;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = cfg.min_score;
  options.nms_iou_threshold = cfg.nms_iou;
  options.top_k = cfg.max_detections;
  return options;
}

groups::RtspEncodedInputOptions rtsp_options(const AppConfig& cfg, int fps) {
  groups::RtspEncodedInputOptions options;
  options.url = cfg.source;
  options.codec = rtsp_codec(cfg);
  options.tcp = cfg.rtsp_tcp;
  options.latency_ms = cfg.rtsp_latency_ms;
  options.source_fps = fps;
  if (cfg.codec == "h264") {
    options.fallback_h264_width = cfg.width;
    options.fallback_h264_height = cfg.height;
  }
  return options;
}

// Receive the HTTP MJPEG stream and decode it to NV12 frames of the configured size.
groups::HttpMjpegDecodedInputOptions http_options(const AppConfig& cfg) {
  groups::HttpMjpegDecodedInputOptions options;
  options.url = cfg.source;
  options.source_fps = cfg.fps;
  options.ssl_strict = cfg.ssl_strict;
  options.output_caps.enable = true;
  options.output_caps.width = cfg.width;
  options.output_caps.height = cfg.height;
  options.output_caps.fps = cfg.fps;
  return options;
}

// One of the camera's MJPEG modes, exactly: width, height, and frame rate.
neat::UsbCameraInputOptions usb_camera_options(const AppConfig& cfg) {
  neat::UsbCameraInputOptions options;
  options.camera_name = cfg.source;
  options.width = static_cast<std::uint32_t>(cfg.width);
  options.height = static_cast<std::uint32_t>(cfg.height);
  options.framerate_num = static_cast<std::uint32_t>(cfg.fps);
  return options;
}

neat::Graph camera_graph(const AppConfig& cfg, const neat::UsbCameraInfo& camera) {
  neat::Graph graph("usb_camera");
  graph.add(neat::nodes::UsbCameraInput(usb_camera_options(cfg), camera));
  return graph;
}

// Hardware decode of the RTSP stream or the camera's JPEG: NV12, or NV16 for a 4:2:2 camera.
neat::Graph decoder_graph(const AppConfig& cfg, int fps, neat::FormatTag frame_format) {
  neat::SimaDecodeOptions decode;
  decode.type = cfg.codec == "h264"   ? neat::SimaDecodeType::H264
                : cfg.codec == "h265" ? neat::SimaDecodeType::H265
                                      : neat::SimaDecodeType::MJPEG;
  decode.out_format = frame_format;
  decode.dec_width = cfg.width;
  decode.dec_height = cfg.height;
  decode.dec_fps = fps;
  if (cfg.codec == "mjpeg") {
    // Leaves free decode surfaces while inference and the video encoder hold frames.
    decode.num_buffers = 32;
  }
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  return graph;
}

// Decoded frames -> model -> "detections" output.
neat::Graph detector_graph(neat::Model& model) {
  neat::Graph graph("detector");
  graph.add(model.graph());
  graph.add(neat::nodes::Output("detections"));
  return graph;
}

// H.264, H.265, and a USB camera's JPEG go to Insight as received; decoded HTTP and RTSP MJPEG
// frames are encoded as H.264.
groups::VideoSenderOptions video_options(const AppConfig& cfg, int fps) {
  auto options = cfg.input_type == "usb"
                     ? groups::VideoSenderOptions::Passthrough(groups::RtspCodec::MJPEG)
                 : cfg.codec == "mjpeg" ? groups::VideoSenderOptions::FromRaw({.fps = fps})
                                        : groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg));
  options.host = cfg.insight_host;
  options.video_port_base = cfg.video_port_base;
  return options;
}

// Bound the decoded frames the H.264 encoder holds, so it cannot starve the decoder.
neat::GraphLinkOptions encoder_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = 2;
  link.max_inflight_total = 2;
  return link;
}

// Send the camera's newest JPEG, so a slow video sender cannot hold back the camera.
neat::GraphLinkOptions camera_video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

neat::MetadataSenderOptions metadata_options(const AppConfig& cfg) {
  neat::MetadataSenderOptions options;
  options.host = cfg.insight_host;
  options.metadata_port_base = cfg.metadata_port_base;
  return options;
}

neat::RunOptions run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// The configured frame rate, or the RTSP stream's when input.fps is 0. The probe also reads the
// frame size, which must match input.width and input.height.
int stream_fps(const AppConfig& cfg) {
  if (cfg.fps > 0) {
    return cfg.fps;
  }
  const auto info = sima_examples::probe_rtsp_stream_info(cfg.source, cfg.rtsp_tcp);
  // A probe that read no size (0x0) leaves the check to the configured size.
  if (info.width > 0 && info.height > 0 && (info.width != cfg.width || info.height != cfg.height)) {
    throw std::runtime_error("the stream is " + std::to_string(info.width) + "x" +
                             std::to_string(info.height) +
                             ", but input.width and input.height are " + std::to_string(cfg.width) +
                             "x" + std::to_string(cfg.height));
  }
  return info.fps;
}

// The boxes in a detections sample, whose first tensor is the BBOX payload.
std::vector<neat::Box> decode_boxes(const neat::Sample& sample, const AppConfig& cfg) {
  // Core candidate: decode_bbox_tensor takes a Tensor, so each example picks the BBOX tensor out
  // of the Sample itself.
  const neat::Tensor* bbox = nullptr;
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    bbox = &sample.tensors.front();
  } else if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value()) {
    bbox = &*sample.tensor;
  } else {
    throw std::runtime_error("expected a BBOX tensor in the detections sample");
  }
  return neat::decode_bbox_tensor(*bbox, cfg.width, cfg.height, cfg.max_detections,
                                  /*strict=*/true)
      .boxes;
}

// Send every frame's boxes to Insight until inference.frames or a stop signal; return the count.
int send_detections(neat::Run& run, neat::MetadataSender& metadata_sender,
                    const std::vector<std::string>& labels, const AppConfig& cfg,
                    const std::atomic<bool>& stop) {
  int processed = 0;
  while (!stop && (cfg.frames == 0 || processed < cfg.frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }

    const std::string payload = sima_examples::detection_egress::serialize(
        decode_boxes(sample, cfg), labels, cfg.width, cfg.height,
        sima_examples::detection_egress::frame_metadata(sample, 0));
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_sender.send_raw_json(payload, &send_error) && !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++processed;
  }
  return processed;
}

int run_app(const CliOptions& cli) {
  if (!fs::exists(cli.config_path)) {
    std::cerr << "Error: config file not found: " << cli.config_path.string() << "\n";
    return 2;
  }
  const AppConfig cfg = load_app_config(cli.config_path);
  if (cli.validate_config_only) {
    std::cout << "Config validated: " << cli.config_path.string() << " (input=" << cfg.input_type
              << ", codec=" << cfg.codec << ", " << cfg.width << "x" << cfg.height
              << ", fps=" << cfg.fps << ")\n";
    return 0;
  }

  const auto labels = sima_examples::load_labels(cfg.labels_path);
  const int fps = stream_fps(cfg);
  const auto camera = cfg.input_type == "usb"
                          ? std::optional(neat::usb_camera_info(usb_camera_options(cfg)))
                          : std::nullopt;
  const neat::FormatTag frame_format = camera ? camera->decoded_format : neat::FormatTag::NV12;
  neat::Model model(cfg.model_path.string(), model_options(cfg, frame_format));
  const neat::Graph detector = detector_graph(model);
  const neat::Graph video = groups::VideoSender(video_options(cfg, fps));

  neat::Graph graph;
  neat::Graph frames;
  if (cfg.input_type == "http") {
    frames = groups::HttpMjpegDecodedInput(http_options(cfg));
    graph.connect(frames, video, encoder_link());
  } else if (cfg.input_type == "usb") {
    const neat::Graph source = camera_graph(cfg, *camera);
    frames = decoder_graph(cfg, fps, frame_format);
    graph.connect(source, frames);
    graph.connect(source, video, camera_video_link());
  } else {
    const neat::Graph source = groups::RtspEncodedInput(rtsp_options(cfg, fps));
    frames = decoder_graph(cfg, fps, frame_format);
    graph.connect(source, frames);
    if (cfg.codec == "mjpeg") {
      graph.connect(frames, video, encoder_link());
    } else {
      // A plain link lets Core tee the encoded stream to the decoder and the video sender.
      graph.connect(source, video);
    }
  }
  graph.connect(frames, detector);

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(cfg), &error);
  if (!metadata_sender.ok()) {
    throw std::runtime_error("metadata sender failed: " + error);
  }
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options());
  std::cout << "source=" << cfg.source << " type=" << cfg.input_type << " codec=" << cfg.codec
            << " stream=" << cfg.width << "x" << cfg.height << "@" << fps
            << " insight=" << cfg.insight_host << " video=" << cfg.video_port_base
            << " metadata=" << cfg.metadata_port_base << "\n";
  const int processed = send_detections(run, metadata_sender, labels, cfg, stop);
  run.close();

  // Core candidate: on a pipe or a file, text still in the stdout buffer when the process exits
  // is lost although the exit code is 0, so std::endl flushes each result line.
  std::cout << "processed=" << processed << " video_sender=" << cfg.insight_host << ":"
            << cfg.video_port_base << std::endl;
  const auto failures = metadata_sender.stats().send_failures;
  if (failures > 0) {
    std::cout << failures << " metadata messages dropped" << std::endl;
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run_app(parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "[ERR] " << error.what() << "\n";
    return 1;
  }
}
