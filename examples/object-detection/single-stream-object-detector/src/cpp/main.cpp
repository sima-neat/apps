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
#include "sima_examples/example_utils.h"
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

struct AppConfig {
  fs::path model_path;
  fs::path labels_path;
  std::string input_type; // rtsp, http, or usb.
  std::string source;
  std::string codec; // mjpeg for HTTP and USB, which carry only MJPEG.
  int width = 0;     // For RTSP, each 0 reads that value from the stream with a probe.
  int height = 0;
  int fps = 0;
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
  // Core sends to 127.0.0.1 when the host is empty.
  if (cfg.insight_host.empty()) {
    throw std::runtime_error("output.insight.host must not be empty");
  }
  if (cfg.frames < 0) {
    throw std::runtime_error("inference.frames must be 0 or more");
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

groups::RtspEncodedInputOptions rtsp_options(const AppConfig& cfg) {
  groups::RtspEncodedInputOptions options;
  options.url = cfg.source;
  options.codec = rtsp_codec(cfg);
  options.tcp = cfg.rtsp_tcp;
  options.latency_ms = cfg.rtsp_latency_ms;
  options.source_fps = cfg.fps;
  if (cfg.codec == "h264") {
    options.fallback_h264_width = cfg.width;
    options.fallback_h264_height = cfg.height;
    options.fallback_h264_fps = cfg.fps;
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
neat::Graph decoder_graph(const AppConfig& cfg, neat::FormatTag frame_format) {
  neat::SimaDecodeOptions decode;
  decode.type = cfg.codec == "h264"   ? neat::SimaDecodeType::H264
                : cfg.codec == "h265" ? neat::SimaDecodeType::H265
                                      : neat::SimaDecodeType::MJPEG;
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
  graph.add(neat::nodes::Output("detections"));
  return graph;
}

// RTSP H.264, H.265, and MJPEG and a USB camera's JPEG go to Insight as received; decoded HTTP
// MJPEG frames are encoded as H.264.
groups::VideoSenderOptions video_options(const AppConfig& cfg) {
  // Core candidate: no HTTP MJPEG input keeps the stream encoded, so its decoded frames are
  // encoded again for Insight. Passthrough(MJPEG) carries RTP/JPEG, at most 2040 pixels per side.
  auto options = cfg.input_type == "http"
                     ? groups::VideoSenderOptions::FromRaw({.fps = cfg.fps})
                     : groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg));
  options.host = cfg.insight_host;
  options.video_port_base = cfg.video_port_base;
  return options;
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

// The config with an RTSP stream's frame size and rate for each of input.width, input.height,
// and input.fps it leaves at 0. HTTP and USB use the configured values.
AppConfig with_stream_info(AppConfig cfg) {
  if (cfg.input_type != "rtsp") {
    return cfg;
  }
  const auto info = sima_examples::probe_rtsp_stream_info(
      cfg.source, cfg.rtsp_tcp, sima_examples::RtspStreamInfo{cfg.width, cfg.height, cfg.fps});
  cfg.width = info.width;
  cfg.height = info.height;
  cfg.fps = info.fps;
  return cfg;
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
  const AppConfig configured = load_app_config(cli.config_path);
  if (cli.validate_config_only) {
    std::cout << "Config validated: " << cli.config_path.string()
              << " (input=" << configured.input_type << ", codec=" << configured.codec << ", "
              << configured.width << "x" << configured.height << ", fps=" << configured.fps
              << ")\n";
    return 0;
  }

  const auto labels = sima_examples::load_labels(configured.labels_path);
  const AppConfig cfg = with_stream_info(configured);
  const auto camera = cfg.input_type == "usb"
                          ? std::optional(neat::usb_camera_info(usb_camera_options(cfg)))
                          : std::nullopt;
  const neat::FormatTag frame_format = camera ? camera->decoded_format : neat::FormatTag::NV12;
  neat::Model model(cfg.model_path.string(), model_options(cfg, frame_format));
  const neat::Graph detector = detector_graph(model);
  const neat::Graph video = groups::VideoSender(video_options(cfg));

  neat::Graph graph;
  neat::Graph frames;
  if (cfg.input_type == "http") {
    frames = groups::HttpMjpegDecodedInput(http_options(cfg));
    graph.connect(frames, video);
  } else if (cfg.input_type == "usb") {
    const neat::Graph source = camera_graph(cfg, *camera);
    frames = decoder_graph(cfg, frame_format);
    graph.connect(source, frames);
    graph.connect(source, video, camera_video_link());
  } else {
    const neat::Graph source = groups::RtspEncodedInput(rtsp_options(cfg));
    frames = decoder_graph(cfg, frame_format);
    graph.connect(source, frames);
    // A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(source, video);
  }
  graph.connect(frames, detector);

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(cfg), &error);
  sima_examples::require(metadata_sender.ok(), error);
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options());
  std::cout << "source=" << cfg.source << " type=" << cfg.input_type << " codec=" << cfg.codec
            << " stream=" << cfg.width << "x" << cfg.height << "@" << cfg.fps
            << " insight=" << cfg.insight_host << " video=" << cfg.video_port_base
            << " metadata=" << cfg.metadata_port_base << "\n";
  const int processed = send_detections(run, metadata_sender, labels, cfg, stop);
  run.close();

  // Core candidate: after an RTSP or HTTP MJPEG run, the process exits with code 0 without
  // flushing stdout, so on a pipe or a file the lines below are lost unless std::endl flushes
  // them. Reproduce: inference.frames: 30 on an RTSP MJPEG source, piped through cat.
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
