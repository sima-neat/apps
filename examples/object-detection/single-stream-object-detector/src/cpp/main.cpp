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

// Detects objects in one RTSP, HTTP MJPEG, or USB camera stream with YOLO26 and shows them in
// Insight. The decoded frames feed the model, and Insight receives the video and every frame's
// boxes.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/detection_egress.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"

#include <csignal>
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

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct AppConfig {
  fs::path model_path;
  fs::path labels_path;
  std::string source_type; // input.type: rtsp, http, or usb.
  std::string source;
  std::string codec; // mjpeg for HTTP and USB, which carry only MJPEG.
  int width = 0;
  int height = 0;
  int fps = 0;             // 0 reads an RTSP stream's frame rate with a probe.
  bool rtsp_tcp = false;   // RTSP only.
  int rtsp_latency_ms = 0; // RTSP only.
  int frames = 0;          // 0 runs until Ctrl+C.
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
  cfg.source_type = config.string("input.type");
  if (cfg.source_type != "rtsp" && cfg.source_type != "http" && cfg.source_type != "usb") {
    throw std::runtime_error("input.type must be rtsp, http, or usb");
  }
  const bool rtsp = cfg.source_type == "rtsp";
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
  if (cfg.source_type == "http") {
    cfg.ssl_strict = config.boolean("app.ssl_strict");
  }
  config.reject_unread_keys();
  if (cfg.codec != "h264" && cfg.codec != "h265" && cfg.codec != "mjpeg") {
    throw std::runtime_error("input.codec must be h264, h265, or mjpeg");
  }
  if (!rtsp && cfg.fps <= 0) {
    throw std::runtime_error("input.fps must be above 0 for an HTTP or USB source");
  }
  const auto rtp_jpeg_side = [](int side) {
    return side > 0 && side <= kRtpJpegMaxSide && side % 8 == 0;
  };
  if (cfg.source_type == "usb" && !(rtp_jpeg_side(cfg.width) && rtp_jpeg_side(cfg.height))) {
    throw std::runtime_error("input.width and input.height of a USB camera must be multiples of "
                             "8 and at most " +
                             std::to_string(kRtpJpegMaxSide) + " (RTP/JPEG)");
  }
  return cfg;
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

// Receive the RTSP stream and decode it to NV12 frames of the configured size.
groups::RtspDecodedInputOptions rtsp_options(const AppConfig& cfg, int fps) {
  groups::RtspDecodedInputOptions options;
  options.url = cfg.source;
  options.codec = cfg.codec == "h264"   ? groups::RtspCodec::H264
                  : cfg.codec == "h265" ? groups::RtspCodec::H265
                                        : groups::RtspCodec::MJPEG;
  options.tcp = cfg.rtsp_tcp;
  options.latency_ms = cfg.rtsp_latency_ms;
  options.source_fps = fps;
  if (cfg.codec == "h264") {
    options.fallback_h264_width = cfg.width;
    options.fallback_h264_height = cfg.height;
  } else {
    options.dec_width = cfg.width;
    options.dec_height = cfg.height;
  }
  options.output_caps.enable = true;
  options.output_caps.width = cfg.width;
  options.output_caps.height = cfg.height;
  options.output_caps.fps = fps;
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

// Hardware decode of the camera's JPEG in its own chroma layout, NV12 or NV16.
neat::Graph usb_decoder_graph(const AppConfig& cfg, neat::FormatTag frame_format) {
  neat::SimaDecodeOptions decode;
  decode.type = neat::SimaDecodeType::MJPEG;
  decode.out_format = frame_format;
  decode.dec_width = cfg.width;
  decode.dec_height = cfg.height;
  decode.dec_fps = cfg.fps;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  return graph;
}

// Decoded frames -> model -> "detections" output.
neat::Graph detector_graph(neat::Model& model) {
  neat::Graph graph("detector");
  graph.add(model.graph());
  graph.add(neat::nodes::Output("detections", neat::OutputOptions::EveryFrame(4)));
  return graph;
}

// Re-encode decoded RTSP and HTTP frames as H.264; send a USB camera's JPEG unchanged.
groups::VideoSenderOptions video_options(const AppConfig& cfg, int fps) {
  auto options = cfg.source_type == "usb"
                     ? groups::VideoSenderOptions::Passthrough(groups::RtspCodec::MJPEG)
                     : groups::VideoSenderOptions::FromRaw({.fps = fps});
  if (cfg.source_type != "usb") {
    options.encoder.bitrate_kbps = 1000;
  }
  options.host = cfg.insight_host;
  options.video_port_base = cfg.video_port_base;
  return options;
}

// Send the newest frames and bound those the video sender holds, so it cannot starve the
// decoder or the camera.
neat::GraphLinkOptions video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  link.max_inflight_per_stream = 2;
  link.max_inflight_total = 2;
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
  options.queue_depth = 3;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// The stream's frame rate: the configured one, or a probe of an RTSP stream when input.fps is 0.
int stream_fps(const AppConfig& cfg) {
  if (cfg.fps > 0) {
    return cfg.fps;
  }
  return sima_examples::probe_rtsp_stream_info(cfg.source, cfg.rtsp_tcp).fps;
}

// The boxes in a detections sample, whose first tensor is the BBOX payload.
std::vector<neat::Box> decode_boxes(const neat::Sample& sample, const AppConfig& cfg) {
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

sima_examples::detection_egress::FrameMetadata frame_metadata(const neat::Sample& sample) {
  sima_examples::detection_egress::FrameMetadata metadata;
  metadata.stream_index = 0;
  metadata.stream_id = sample.stream_id;
  metadata.frame_id = sample.frame_id;
  metadata.pts_ns = sample.pts_ns;
  metadata.dts_ns = sample.dts_ns;
  metadata.duration_ns = sample.duration_ns;
  metadata.input_seq = sample.input_seq;
  metadata.orig_input_seq = sample.orig_input_seq;
  if (sample.pts_ns >= 0) {
    metadata.rtp_timestamp =
        sima_examples::detection_egress::rtp_timestamp_from_pts_ns(sample.pts_ns);
  }
  return metadata;
}

// Send every frame's boxes to Insight until inference.frames or Ctrl+C; return the count.
int send_detections(neat::Run& run, neat::MetadataSender& metadata_sender,
                    const std::vector<std::string>& labels, const AppConfig& cfg) {
  int processed = 0;
  while (g_stop_requested == 0 && (cfg.frames == 0 || processed < cfg.frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }

    const std::string payload = sima_examples::detection_egress::serialize(
        decode_boxes(sample, cfg), labels, cfg.width, cfg.height, frame_metadata(sample));
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
    std::cout << "Config validated: " << cli.config_path.string() << " (source=" << cfg.source_type
              << ")\n";
    return 0;
  }

  const auto labels = sima_examples::load_labels(cfg.labels_path);
  const int fps = stream_fps(cfg);
  const auto camera = cfg.source_type == "usb"
                          ? std::optional(neat::usb_camera_info(usb_camera_options(cfg)))
                          : std::nullopt;
  const neat::FormatTag frame_format = camera ? camera->decoded_format : neat::FormatTag::NV12;
  neat::Model model(cfg.model_path.string(), model_options(cfg, frame_format));
  const neat::Graph detector = detector_graph(model);
  const neat::Graph video = groups::VideoSender(video_options(cfg, fps));

  neat::Graph graph;
  neat::Graph frames;
  if (camera) {
    const neat::Graph source = camera_graph(cfg, *camera);
    frames = usb_decoder_graph(cfg, frame_format);
    graph.connect(source, video, video_link());
    graph.connect(source, frames);
  } else {
    frames = cfg.source_type == "rtsp" ? groups::RtspDecodedInput(rtsp_options(cfg, fps))
                                       : groups::HttpMjpegDecodedInput(http_options(cfg));
    graph.connect(frames, video, video_link());
  }
  graph.connect(frames, detector);

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(cfg), &error);
  if (!metadata_sender.ok()) {
    throw std::runtime_error("metadata sender failed: " + error);
  }
  std::signal(SIGINT, request_stop);
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options());
  std::cout << "source=" << cfg.source << " type=" << cfg.source_type << " codec=" << cfg.codec
            << " stream=" << cfg.width << "x" << cfg.height << "@" << fps
            << " insight=" << cfg.insight_host << " video=" << cfg.video_port_base
            << " metadata=" << cfg.metadata_port_base << "\n";
  const int processed = send_detections(run, metadata_sender, labels, cfg);
  run.close();

  // std::endl flushes: on a pipe or a file, text still buffered when the process exits is
  // lost, although the exit code is 0.
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
