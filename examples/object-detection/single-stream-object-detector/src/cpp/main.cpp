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
#include <map>
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
const std::map<std::string, groups::RtspCodec> kRtspCodecs = {
    {"h264", groups::RtspCodec::H264},
    {"h265", groups::RtspCodec::H265},
    {"mjpeg", groups::RtspCodec::MJPEG},
};
const std::map<std::string, neat::SimaDecodeType> kDecodeTypes = {
    {"h264", neat::SimaDecodeType::H264},
    {"h265", neat::SimaDecodeType::H265},
    {"mjpeg", neat::SimaDecodeType::MJPEG},
};
// The codec of each input type: HTTP and USB carry only MJPEG, and RTSP names it in input.codec.
const std::map<std::string, std::string> kInputCodecs = {
    {"rtsp", ""}, {"http", "mjpeg"}, {"usb", "mjpeg"}};

fs::path parse_args(int argc, char** argv) {
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  fs::path config_path = fs::exists(adjacent_config)
                             ? adjacent_config
                             : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else {
      throw std::runtime_error("usage: " + std::string(argv[0]) + " [--config <path>]");
    }
  }
  return config_path;
}

// YOLO26 on the decoded frames: NV12, or NV16 from a 4:2:2 USB camera.
neat::Model::Options model_options(const sima_examples::Config& config,
                                   const sima_examples::RtspStreamInfo& stream,
                                   neat::FormatTag frame_format) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = frame_format == neat::FormatTag::NV16
                                                      ? neat::PreprocessColorFormat::NV16
                                                      : neat::PreprocessColorFormat::NV12;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = neat::BoxDecodeType::YoloV26;
  options.score_threshold = config.number("inference.min_score");
  options.nms_iou_threshold = config.number("inference.nms_iou");
  options.top_k = config.integer("inference.max_detections");
  return options;
}

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = config.string("input.source");
  options.codec = kRtspCodecs.at(config.string("input.codec"));
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  // With input.fps at 0, Core also takes the caps from the stream: above 60 fps, caps set from
  // the probed values alone failed to negotiate in some runs.
  options.auto_caps_from_stream = config.integer("input.fps") == 0;
  options.source_fps = stream.fps;
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
    options.fallback_h264_fps = stream.fps;
  }
  return options;
}

// Receive the HTTP MJPEG stream and decode it to NV12 frames of the configured size.
groups::HttpMjpegDecodedInputOptions http_options(const sima_examples::Config& config,
                                                  const sima_examples::RtspStreamInfo& stream) {
  groups::HttpMjpegDecodedInputOptions options;
  options.url = config.string("input.source");
  options.source_fps = stream.fps;
  options.ssl_strict = config.boolean("app.ssl_strict");
  options.output_caps.enable = true;
  options.output_caps.width = stream.width;
  options.output_caps.height = stream.height;
  options.output_caps.fps = stream.fps;
  return options;
}

// One of the camera's MJPEG modes, exactly: width, height, and frame rate.
neat::UsbCameraInputOptions usb_camera_options(const sima_examples::Config& config,
                                               const sima_examples::RtspStreamInfo& stream) {
  neat::UsbCameraInputOptions options;
  options.camera_name = config.string("input.source");
  options.width = static_cast<std::uint32_t>(stream.width);
  options.height = static_cast<std::uint32_t>(stream.height);
  options.framerate_num = static_cast<std::uint32_t>(stream.fps);
  return options;
}

neat::Graph camera_graph(const neat::UsbCameraInputOptions& options,
                         const neat::UsbCameraInfo& camera) {
  neat::Graph graph("usb_camera");
  graph.add(neat::nodes::UsbCameraInput(options, camera));
  return graph;
}

// Hardware decode of the RTSP stream or the camera's JPEG: NV12, or NV16 for a 4:2:2 camera.
neat::Graph decoder_graph(const std::string& codec, const sima_examples::RtspStreamInfo& stream,
                          neat::FormatTag frame_format) {
  neat::SimaDecodeOptions decode;
  decode.type = kDecodeTypes.at(codec);
  decode.out_format = frame_format;
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
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
groups::VideoSenderOptions video_options(const sima_examples::Config& config,
                                         const std::string& input_type, const std::string& codec,
                                         const sima_examples::RtspStreamInfo& stream) {
  // Core candidate: no HTTP MJPEG input keeps the stream encoded, so its decoded frames are
  // encoded again for Insight. Passthrough(MJPEG) carries RTP/JPEG, at most 2040 pixels per side.
  auto options = input_type == "http"
                     ? groups::VideoSenderOptions::FromRaw({.fps = stream.fps})
                     : groups::VideoSenderOptions::Passthrough(kRtspCodecs.at(codec));
  options.host = config.string("output.insight.host");
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
}

// Send the camera's newest JPEG, so a slow video sender cannot hold back the camera.
neat::GraphLinkOptions camera_video_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

neat::MetadataSenderOptions metadata_options(const sima_examples::Config& config) {
  neat::MetadataSenderOptions options;
  options.host = config.string("output.insight.host");
  options.metadata_port_base = config.integer("output.insight.metadata_port_base");
  return options;
}

neat::RunOptions run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// The boxes in a detections sample, whose first tensor is the BBOX payload.
std::vector<neat::Box> decode_boxes(const neat::Sample& sample,
                                    const sima_examples::RtspStreamInfo& stream,
                                    int max_detections) {
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
  return neat::decode_bbox_tensor(*bbox, stream.width, stream.height, max_detections,
                                  /*strict=*/true)
      .boxes;
}

// Send every frame's boxes to Insight until `frames` (0: no limit) or a stop signal; return the
// count.
int send_detections(neat::Run& run, neat::MetadataSender& metadata_sender,
                    const std::vector<std::string>& labels,
                    const sima_examples::RtspStreamInfo& stream, int max_detections, int frames,
                    const std::atomic<bool>& stop) {
  int processed = 0;
  while (!stop && (frames == 0 || processed < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }

    const std::string payload = sima_examples::detection_egress::serialize(
        decode_boxes(sample, stream, max_detections), labels, stream.width, stream.height,
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

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const std::string input_type = config.string("input.type");
  std::string codec = kInputCodecs.at(input_type);
  if (codec.empty()) {
    codec = config.string("input.codec");
  }
  const sima_examples::RtspStreamInfo configured{
      config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")};
  // An RTSP stream's size and rate come from the probe for each value the config leaves at 0;
  // HTTP and USB use the configured values.
  const auto stream =
      input_type == "rtsp"
          ? sima_examples::probe_rtsp_stream_info(config.string("input.source"),
                                                  config.boolean("input.rtsp.tcp"), configured)
          : configured;
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  const auto camera_options =
      input_type == "usb" ? std::optional(usb_camera_options(config, stream)) : std::nullopt;
  const auto camera =
      camera_options ? std::optional(neat::usb_camera_info(*camera_options)) : std::nullopt;
  const neat::FormatTag frame_format = camera ? camera->decoded_format : neat::FormatTag::NV12;
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config, stream, frame_format));
  const neat::Graph detector = detector_graph(model);
  const neat::Graph video = groups::VideoSender(video_options(config, input_type, codec, stream));

  neat::Graph graph;
  neat::Graph frames;
  if (input_type == "http") {
    frames = groups::HttpMjpegDecodedInput(http_options(config, stream));
    graph.connect(frames, video);
  } else if (input_type == "usb") {
    const neat::Graph source = camera_graph(*camera_options, *camera);
    frames = decoder_graph(codec, stream, frame_format);
    graph.connect(source, frames);
    graph.connect(source, video, camera_video_link());
  } else {
    const neat::Graph source = groups::RtspEncodedInput(rtsp_options(config, stream));
    frames = decoder_graph(codec, stream, frame_format);
    graph.connect(source, frames);
    // A plain link lets Core tee the encoded stream to the decoder and the video sender.
    graph.connect(source, video);
  }
  graph.connect(frames, detector);

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(config), &error);
  sima_examples::require(metadata_sender.ok(), error);
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_detections throws.
  neat::Run run = graph.build(run_options());
  const std::string host = config.string("output.insight.host");
  const int video_port = config.integer("output.insight.video_port_base");
  std::cout << "source=" << config.string("input.source") << " type=" << input_type
            << " codec=" << codec << " stream=" << stream.width << "x" << stream.height << "@"
            << stream.fps << " insight=" << host << " video=" << video_port
            << " metadata=" << config.integer("output.insight.metadata_port_base") << "\n";
  const int processed = send_detections(run, metadata_sender, labels, stream,
                                        config.integer("inference.max_detections"),
                                        config.integer("inference.frames"), stop);
  run.close();

  // Core candidate: after an RTSP or HTTP MJPEG run, the process exits with code 0 without
  // flushing stdout, so on a pipe or a file the lines below are lost unless std::endl flushes
  // them. Reproduce: inference.frames: 30 on an RTSP MJPEG source, piped through cat.
  std::cout << "processed=" << processed << " video_sender=" << host << ":" << video_port
            << std::endl;
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
