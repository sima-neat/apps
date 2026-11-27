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

// Flags surface defects in one RTSP stream with FastFlow and outlines them in Insight.
// The decoded frames feed the model, and the host turns its anomaly map into defect regions.
// Insight receives the video as received and every frame's regions as polygons.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/anomaly_regions.h"

#include <opencv2/core.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

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
// The mean each package expects, in RGB order on 0..1 pixels. fastflow_demo takes its training
// set's mean; an anomalib package normalizes inside its graph.
const std::map<std::string, std::array<float, 3>> kNormalizeMeans = {
    {"fastflow_demo", {0.3893437F, 0.35421189F, 0.36142577F}},
    {"anomalib", {0.0F, 0.0F, 0.0F}},
};

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

// EV74 converts each NV12 frame, stretches it to the model input, and normalizes it.
// Stretching makes the map cover the whole frame, so a region scales back to the frame by its
// width and height. A square frame, as FastFlow is trained on, is not distorted.
neat::Model::Options model_options(const sima_examples::Config& config,
                                   const sima_examples::RtspStreamInfo& stream) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.normalize.mean = config.choice("model.id", kNormalizeMeans);
  options.preprocess.normalize.has_explicit_stats = true;
  return options;
}

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = config.string("input.source");
  options.codec = config.choice("input.codec", kRtspCodecs);
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  options.source_fps = stream.fps;
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size.
neat::Graph decoder_graph(const sima_examples::Config& config,
                          const sima_examples::RtspStreamInfo& stream) {
  neat::SimaDecodeOptions decode;
  decode.type = config.choice("input.codec", kDecodeTypes);
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  return graph;
}

// Decoded frames -> FastFlow -> "anomaly_map" output.
neat::Graph detector_graph(neat::Model& model) {
  neat::Graph graph("detector");
  graph.add(model.graph());
  graph.add(neat::nodes::Output("anomaly_map"));
  return graph;
}

// Send the stream to Insight as received.
groups::VideoSenderOptions video_options(const sima_examples::Config& config) {
  auto options = groups::VideoSenderOptions::Passthrough(config.choice("input.codec", kRtspCodecs));
  options.host = config.string("output.insight.host");
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
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

fastflow::RegionOptions region_options(const sima_examples::Config& config,
                                       const sima_examples::RtspStreamInfo& stream) {
  fastflow::RegionOptions options;
  options.width = stream.width;
  options.height = stream.height;
  options.threshold = config.number("app.anomaly_threshold");
  options.min_region_px = config.integer("app.min_region_px");
  options.max_regions = config.integer("inference.max_detections");
  return options;
}

// The map of an "anomaly_map" sample: per pixel, the probability it shows no good part.
cv::Mat anomaly_map(const neat::Sample& sample, int map_side) {
  const neat::Tensor tensor = neat::tensors_from_sample(sample).front();
  cv::Mat map(map_side, map_side, CV_32FC1);
  const std::size_t bytes = map.total() * sizeof(float);
  sima_examples::require(tensor.dtype == neat::TensorDType::Float32 &&
                             tensor.dense_bytes_tight() == bytes &&
                             tensor.copy_dense_bytes_tight_to(map.data, bytes),
                         "the model must return a Float32 map of " + std::to_string(map_side) +
                             "x" + std::to_string(map_side) + " values");
  return map;
}

struct RegionCounts {
  int processed = 0; // Frames processed.
  int flagged = 0;   // Frames with a defect.
};

// Send every frame's defect regions to Insight until `frames` (0: no limit) or a stop signal.
RegionCounts send_regions(neat::Run& run, neat::MetadataSender& metadata_sender, int map_side,
                          const fastflow::RegionOptions& options, int frames,
                          const std::atomic<bool>& stop) {
  RegionCounts counts;
  while (!stop && (frames == 0 || counts.processed < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("anomaly_map", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "anomaly_map", error, run.last_error())) {
      continue;
    }
    const auto regions = fastflow::defect_regions(anomaly_map(sample, map_side), options);
    // A frame without a defect sends no segments, which clears the last outline. A nonblocking
    // send drops the message while the socket buffer is full; the drops are counted in the
    // sender's stats and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_sender.send_metadata("segmentation", fastflow::segments_json(regions),
                                       sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1,
                                       sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "",
                                       &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++counts.processed;
    counts.flagged += regions.empty() ? 0 : 1;
  }
  return counts;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(config, stream));
  const int map_side = static_cast<int>(model.output_specs().at(0).shape.at(1));

  const neat::Graph rtsp = groups::RtspEncodedInput(rtsp_options(config, stream));
  const neat::Graph decoder = decoder_graph(config, stream);
  const neat::Graph video = groups::VideoSender(video_options(config));
  neat::Graph graph;
  graph.connect(rtsp, decoder);
  // A plain link lets Core tee the encoded stream to the decoder and the video sender.
  graph.connect(rtsp, video);
  graph.connect(decoder, detector_graph(model));

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(config), &error);
  sima_examples::require(metadata_sender.ok(), error);
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_regions throws.
  neat::Run run = graph.build(run_options());
  const std::string host = config.string("output.insight.host");
  const int video_port = config.integer("output.insight.video_port_base");
  std::cout << "source=" << config.string("input.source")
            << " codec=" << config.string("input.codec") << " stream=" << stream.width << "x"
            << stream.height << "@" << stream.fps << " map=" << map_side << "x" << map_side
            << " insight=" << host << " video=" << video_port
            << " metadata=" << config.integer("output.insight.metadata_port_base") << std::endl;
  const RegionCounts counts =
      send_regions(run, metadata_sender, map_side, region_options(config, stream),
                   config.integer("inference.frames"), stop);
  run.close();

  // std::endl flushes: after an RTSP run the process can exit without flushing stdout, which
  // loses these lines on a pipe or a file.
  std::cout << "processed=" << counts.processed << " flagged=" << counts.flagged
            << " video_sender=" << host << ":" << video_port << std::endl;
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
