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

// Finds SuperPoint feature points in one H.264 RTSP stream, shown in Insight.
// The decoded frames feed the model. Insight receives the video as received and every frame's
// feature points.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/feature_points.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

constexpr int kPullTimeoutMs = 100;

struct FeatureCounts {
  int frames = 0;
  std::int64_t points = 0;
  std::int64_t descriptor_dim = 0;
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

// SuperPoint on the gray frames, stretched to the model input, with its keypoints and descriptors
// decoded.
neat::Model::Options model_options(const sima_examples::RtspStreamInfo& stream) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::GRAY8;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  // The default mean 0 and stddev 1, marked explicit, turn normalization on: pixels / 255.
  options.preprocess.normalize.has_explicit_stats = true;
  options.decode_type = neat::BoxDecodeType::SuperPoint;
  // A65V1 sets the decoder's numerics and its defaults: score threshold 0.1 while
  // score_threshold is 0, and border margin 0. Auto would also give A65V1 for this package,
  // which names no profile, but would follow a package that does. The 600-point limit is
  // Core's default top_k for every profile.
  options.superpoint.profile = neat::SuperPointProfile::A65V1;
  options.processcvu.post_run_target = "A65";
  return options;
}

// The frame the decoded keypoints are in: the model input.
superpoint::FrameSize model_size(const neat::Model& model) {
  const auto contract = model.resolved_preprocess_plan().mla_contract;
  sima_examples::require(contract.width > 0 && contract.height > 0,
                         "SuperPoint model input size is unresolved");
  return {contract.width, contract.height};
}

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = config.string("input.source");
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  // With input.fps at 0, Core also takes the caps from the stream: above 60 fps, caps set from
  // the probed values alone failed to negotiate in some runs.
  options.auto_caps_from_stream = config.integer("input.fps") == 0;
  options.source_fps = stream.fps;
  options.fallback_h264_width = stream.width;
  options.fallback_h264_height = stream.height;
  return options;
}

// Hardware decode of the H.264 stream to NV12 frames. The "frames" output keeps only the newest,
// so a pull loop slower than the stream cannot hold back the video sender.
neat::Graph decoder_graph(const sima_examples::RtspStreamInfo& stream) {
  neat::SimaDecodeOptions decode;
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  graph.add(neat::nodes::Output("frames", neat::OutputOptions::Latest()));
  return graph;
}

// The H.264 stream goes to Insight as received.
groups::VideoSenderOptions video_options(const sima_examples::Config& config) {
  auto options = groups::VideoSenderOptions::Passthrough(groups::RtspCodec::H264);
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

// A gray image as the model's input tensor, copied into EV74 memory, from_cv_mat's default.
neat::Tensor gray_tensor(const cv::Mat& gray) {
  return neat::Tensor::from_cv_mat(gray, neat::ImageSpec::PixelFormat::GRAY8);
}

// The Y plane of a decoded NV12 frame: the gray image SuperPoint reads. The only copy is the one
// into EV74 memory, made from a view of the mapped frame while the mapping lives.
neat::Tensor luma(const neat::Sample& sample, const sima_examples::RtspStreamInfo& stream) {
  // Core candidate: Preproc converts no YUV format to gray, and a VideoConvert before the model
  // drops the frame metadata Preproc needs, so the host hands the model the Y plane.
  const auto tensors = neat::tensors_from_sample(sample, /*require_nonempty=*/false);
  sima_examples::require(!tensors.empty(), "decoded frame has no tensor");
  const auto nv12 = tensors.front().map_nv12_read();
  sima_examples::require(nv12.has_value(), "decoded frame is not NV12");
  const int width = nv12->view.width;
  const int height = nv12->view.height;
  sima_examples::require(width == stream.width && height == stream.height,
                         "frame size " + std::to_string(width) + "x" + std::to_string(height) +
                             " differs from the stream's " + std::to_string(stream.width) + "x" +
                             std::to_string(stream.height) + "; the stream must keep one size");
  // from_cv_mat only reads the view, so dropping const here writes nothing.
  const cv::Mat y_plane(height, width, CV_8UC1, const_cast<std::uint8_t*>(nv12->view.y),
                        static_cast<std::size_t>(nv12->view.y_stride));
  return gray_tensor(y_plane);
}

// Run the model on every frame and send its feature points to Insight until `frames` (0: no
// limit) or a stop signal; return the frame count, the point count, and the descriptor length.
FeatureCounts send_feature_points(neat::Run& run, neat::Model::Runner& runner,
                                  neat::MetadataSender& metadata_sender,
                                  const superpoint::FrameSize& model_size,
                                  const sima_examples::RtspStreamInfo& stream, int frames,
                                  const std::atomic<bool>& stop) {
  FeatureCounts counts;
  while (!stop && (frames == 0 || counts.frames < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("frames", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "frames", error, run.last_error())) {
      continue;
    }

    const auto features =
        neat::decode_superpoint(runner.run(neat::TensorList{luma(sample, stream)})).at(0);
    const auto points = superpoint::source_points(superpoint::floats(features.keypoints),
                                                  model_size, {stream.width, stream.height});
    // A nonblocking send drops the message while the socket buffer is full; the drops are
    // counted in the sender's stats and reported when the run ends. Any other failure throws.
    // The frame's timestamp lets Insight draw the points on the video frame they belong to.
    std::string send_error;
    if (!metadata_sender.send_metadata("pose-estimation", superpoint::points_json(points),
                                       sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1,
                                       sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "",
                                       &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++counts.frames;
    counts.points += static_cast<std::int64_t>(points.size());
    counts.descriptor_dim = features.descriptors.shape.at(1);
  }
  return counts;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  // The stream's size and rate come from the probe for each value the config leaves at 0.
  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model model(model_path.string(), model_options(stream));
  auto runner = model.build(
      neat::TensorList{gray_tensor(cv::Mat::zeros(stream.height, stream.width, CV_8UC1))});
  const neat::Graph video = groups::VideoSender(video_options(config));

  neat::Graph graph;
  const neat::Graph source = groups::RtspEncodedInput(rtsp_options(config, stream));
  graph.connect(source, decoder_graph(stream));
  // A plain link lets Core tee the encoded stream to the decoder and the video sender.
  graph.connect(source, video);

  std::string error;
  neat::MetadataSender metadata_sender(metadata_options(config), &error);
  sima_examples::require(metadata_sender.ok(), error);
  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's and the runner's destructors close them when send_feature_points throws.
  neat::Run run = graph.build(run_options());
  const std::string host = config.string("output.insight.host");
  const int video_port = config.integer("output.insight.video_port_base");
  std::cout << "source=" << config.string("input.source") << " stream=" << stream.width << "x"
            << stream.height << "@" << stream.fps << " insight=" << host << " video=" << video_port
            << " metadata=" << config.integer("output.insight.metadata_port_base") << "\n";
  const auto counts = send_feature_points(run, runner, metadata_sender, model_size(model), stream,
                                          config.integer("inference.frames"), stop);
  run.close();
  runner.close();

  const double average = counts.frames ? static_cast<double>(counts.points) / counts.frames : 0.0;
  // std::endl flushes the summary when it is written, so a pipe or a file gets it even if the
  // process later ends without flushing stdout (sima-neat/apps#673).
  std::cout << "frames=" << counts.frames << " average_points=" << std::fixed
            << std::setprecision(1) << average << " descriptor_dim=" << counts.descriptor_dim
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
