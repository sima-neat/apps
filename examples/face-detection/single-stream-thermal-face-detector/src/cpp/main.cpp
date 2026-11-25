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

// Detects faces and their five landmarks in one thermal RTSP stream with yolov5s-face, shown in
// Insight. The decoded frames branch to Insight's video and to the model. The model's six raw
// heads are decoded on the host, and Insight receives every frame's faces as a pose overlay.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/yolov5_face.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kPullTimeoutMs = 100;

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

// yolov5s-face on NV12 frames: letterbox to the 800x800 canvas and scale to 0-1 on the device.
// No box decode: the model outputs its raw heads.
neat::Model::Options model_options() {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  return options;
}

// Receive the H.264 stream and decode it to NV12 frames of the stream's size and rate.
groups::RtspDecodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspDecodedInputOptions options;
  options.url = config.string("input.source");
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  options.fallback_h264_width = stream.width;
  options.fallback_h264_height = stream.height;
  options.fallback_h264_fps = stream.fps;
  options.output_caps.enable = true;
  options.output_caps.width = stream.width;
  options.output_caps.height = stream.height;
  options.output_caps.fps = stream.fps;
  return options;
}

// The "video" branch: decoded frames encoded as H.264 for Insight.
neat::Graph video_graph(const sima_examples::Config& config,
                        const sima_examples::RtspStreamInfo& stream) {
  neat::SimaEncodeOptions encode;
  encode.fps = stream.fps;
  auto options = groups::VideoSenderOptions::FromRaw(encode);
  options.host = config.string("output.insight.host");
  options.video_port_base = config.integer("output.insight.video_port_base");
  neat::Graph graph("video");
  graph.connect(neat::nodes::Input("video"), groups::VideoSender(options));
  return graph;
}

// The "model" branch: decoded frames into the model.
neat::Graph model_graph(neat::Model& model) {
  neat::Graph graph("model");
  graph.connect(neat::nodes::Input("model"), model);
  return graph;
}

// The "detections" output the pull loop reads.
neat::Graph detections_graph() {
  neat::Graph graph("detections");
  graph.add(neat::nodes::Output("detections"));
  return graph;
}

// Run the model's postprocessing adapters, such as dequantize, on the EV74 vision processor,
// which leaves the host CPU to the face decode.
neat::GraphOptions graph_options() {
  neat::GraphOptions options;
  options.advanced_execution.postprocess_target = "EV74";
  return options;
}

// Pass on only the newest frame, so a slow branch cannot hold back the stream.
neat::GraphLinkOptions live_link() {
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
  // One more queued frame than the Realtime preset's 2.
  options.queue_depth = 3;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// The model's output tensors as float split heads, from a Tensor, TensorSet, or Bundle sample.
std::vector<yolov5_face::SplitHead> split_heads(const neat::Sample& sample) {
  std::vector<yolov5_face::SplitHead> heads;
  for (const neat::Tensor& tensor : neat::tensors_from_sample(sample)) {
    if (tensor.dtype != neat::TensorDType::Float32) {
      throw std::runtime_error("expected Float32 split heads");
    }
    // Copied straight into floats, which is alignment-safe where a reinterpret_cast of a byte
    // buffer could be an unaligned load.
    yolov5_face::SplitHead head{{tensor.shape.begin(), tensor.shape.end()},
                                std::vector<float>(tensor.dense_bytes_tight() / sizeof(float))};
    sima_examples::require(
        tensor.copy_dense_bytes_tight_to(reinterpret_cast<uint8_t*>(head.values.data()),
                                         head.values.size() * sizeof(float)),
        "failed to read an output tensor");
    heads.push_back(std::move(head));
  }
  return heads;
}

// Send every frame's faces to Insight until `frames` (0: no limit) or a stop signal; return the
// count.
int send_faces(neat::Run& run, neat::MetadataSender& metadata_sender,
               const sima_examples::Config& config, const std::string& label,
               const sima_examples::RtspStreamInfo& stream, int frames,
               const std::atomic<bool>& stop) {
  const auto min_score = static_cast<float>(config.number("inference.min_score"));
  const auto nms_iou = static_cast<float>(config.number("inference.nms_iou"));
  const int max_detections = config.integer("inference.max_detections");
  const auto letterbox = yolov5_face::letterbox(stream.width, stream.height);
  int processed = 0;
  while (!stop && (frames == 0 || processed < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("detections", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "detections", error, run.last_error())) {
      continue;
    }
    const auto faces = yolov5_face::to_frame(
        yolov5_face::decode_faces(split_heads(sample), min_score, nms_iou, max_detections),
        letterbox, stream.width, stream.height);
    // The frame's presentation time lets Insight draw the faces on the frame they came from.
    const int64_t timestamp_ms = sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1;
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_sender.send_metadata(
            "pose-estimation", yolov5_face::poses_json(faces, label), timestamp_ms,
            std::to_string(std::max<int64_t>(sample.frame_id, 0)), &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++processed;
  }
  return processed;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const sima_examples::RtspStreamInfo configured{
      config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")};
  // The stream's size and rate come from the probe for each value the config leaves at 0.
  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"), configured);
  // The label file names the one class, face.
  const std::string label = sima_examples::load_labels(config.string("model.labels")).front();
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options());

  const neat::Graph source = groups::RtspDecodedInput(rtsp_options(config, stream));
  const neat::Graph branch = neat::graphs::Branch("source", {"video", "model"});
  const neat::Graph video = video_graph(config, stream);
  const neat::Graph detector = model_graph(model);

  neat::Graph graph(graph_options());
  graph.connect(source, branch);
  graph.connect(branch, video, live_link());
  graph.connect(branch, detector, live_link());
  graph.connect(detector, detections_graph());

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(config), &error);
  sima_examples::require(metadata_sender.ok(), error);
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_faces throws.
  neat::Run run = graph.build(run_options());
  const std::string host = config.string("output.insight.host");
  const int video_port = config.integer("output.insight.video_port_base");
  std::cout << "source=" << config.string("input.source") << " stream=" << stream.width << "x"
            << stream.height << "@" << stream.fps << " insight=" << host << " video=" << video_port
            << " metadata=" << config.integer("output.insight.metadata_port_base") << "\n";
  const int processed = send_faces(run, metadata_sender, config, label, stream,
                                   config.integer("inference.frames"), stop);
  run.close();

  // std::endl flushes: single-stream-object-detector found these lines lost on a pipe without it.
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
