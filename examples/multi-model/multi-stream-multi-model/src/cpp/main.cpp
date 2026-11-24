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

// Runs a different YOLO model on each of up to four RTSP streams, shown in Insight. Each stream's
// model detects objects, segments them, or estimates poses. Insight draws the results over that
// stream's video, one channel per stream.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/labels.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/insight_metadata.h"
#include "utils/pending_results.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

const std::map<std::string, groups::RtspCodec> kRtspCodecs = {
    {"h264", groups::RtspCodec::H264},
    {"h265", groups::RtspCodec::H265},
};
const std::map<std::string, neat::SimaDecodeType> kDecodeTypes = {
    {"h264", neat::SimaDecodeType::H264},
    {"h265", neat::SimaDecodeType::H265},
};
// Insight metadata types: Insight draws boxes, mask outlines, or skeletons for each.
constexpr const char* kDetection = "object-detection";
constexpr const char* kSegmentation = "segmentation";
constexpr const char* kPose = "pose-estimation";
// Each app.model_ids value: the BoxDecode type of the package's heads, and the Insight metadata
// type of its results. YOLO11 packages carry the YOLOv8 heads.
const std::map<std::string, std::pair<neat::BoxDecodeType, std::string>> kModelIds = {
    {"yolov8", {neat::BoxDecodeType::YoloV8, kDetection}},
    {"yolov8-seg", {neat::BoxDecodeType::YoloV8Seg, kSegmentation}},
    {"yolov8-pose", {neat::BoxDecodeType::YoloV8Pose, kPose}},
    {"yolo26", {neat::BoxDecodeType::YoloV26, kDetection}},
    {"yolo26-seg", {neat::BoxDecodeType::YoloV26Seg, kSegmentation}},
    {"yolo26-pose", {neat::BoxDecodeType::YoloV26Pose, kPose}},
};
// The key that holds a message's entries, for each metadata type.
const std::map<std::string, std::string> kDataKeys = {
    {kDetection, "objects"},
    {kSegmentation, "segments"},
    {kPose, "poses"},
};
// How long a stream's decoded frames may arrive with no result before the app stops with an
// error. Not measured on a DevKit: a result takes a fraction of a second.
constexpr double kResultDeadlineS = 10.0;
// The same for the first result, which also waits for Core to build the rest of the pipeline on
// the first frame (EdgeRouter.cpp). Not measured on a DevKit.
constexpr double kFirstResultDeadlineS = 30.0;
// Decoded frames each stream's decoder keeps: the 4 its model may hold, and 8 more to decode into
// while those are in use. 12 as before the conversion, not re-measured with this graph.
constexpr int kDecoderBuffers = 12;
// The pull loop polls every output without waiting, and sleeps this long after a round that
// pulled no result.
constexpr std::chrono::milliseconds kIdleSleep{2};

// What the pull loop keeps for one stream.
struct Stream {
  sima_examples::RtspStreamInfo info;
  std::string metadata_type;
  neat::MetadataSender metadata_sender;
  pending_results::PendingResults pending;
  int processed = 0;
  int left_out = 0;
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

// A YOLO model on the decoded NV12 frames of one stream, decoded by `decode_type`.
neat::Model::Options model_options(const sima_examples::Config& config,
                                   const sima_examples::RtspStreamInfo& stream,
                                   neat::BoxDecodeType decode_type, int max_detections) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  // Lets streams above Core's 1920x1080 default in (PreprocessPlan.h).
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.preset = neat::NormalizePreset::COCO_YOLO;
  options.decode_type = decode_type;
  // The pose heads score one class, person, while the packages declare the 80 COCO classes.
  if (decode_type == neat::BoxDecodeType::YoloV8Pose ||
      decode_type == neat::BoxDecodeType::YoloV26Pose) {
    options.num_classes = 1;
  }
  options.score_threshold = static_cast<float>(config.number("inference.min_score"));
  options.nms_iou_threshold = static_cast<float>(config.number("inference.nms_iou"));
  options.top_k = max_detections;
  return options;
}

std::string results_output(std::size_t index) {
  return "results_" + std::to_string(index);
}

std::string frames_output(std::size_t index) {
  return "frames_" + std::to_string(index);
}

// One stream's model: "model_frame" input -> model -> "results_<index>" output.
neat::Graph model_graph(const neat::Model& model, std::size_t index) {
  neat::Graph frames;
  frames.add(neat::nodes::Input("model_frame", model.input_appsrc_options(false)));

  neat::Graph results;
  results.add(neat::nodes::Output(results_output(index)));

  neat::Graph stages = model.graph();
  neat::Graph graph;
  graph.connect(frames, stages);
  graph.connect(stages, results);
  return graph;
}

groups::RtspEncodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const std::string& url,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspEncodedInputOptions options;
  options.url = url;
  options.codec = config.choice("input.codec", kRtspCodecs);
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  // 0 accepts every RTP payload type. The default accepts only 96, and a camera may send another
  // dynamic type.
  options.payload_type = 0;
  options.source_fps = config.integer("input.fps");
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
    if (options.source_fps > 0) {
      // H.264 caps from input.fps and the probed size, instead of the stream's own.
      options.auto_caps_from_stream = false;
    } else {
      // The rate the stream's caps fall back to; with input.fps set it has no effect.
      options.fallback_h264_fps = stream.fps;
    }
  }
  return options;
}

// Hardware decode to NV12 frames at the stream's size, ending at "model_frame".
neat::Graph decoder_graph(const sima_examples::Config& config,
                          const sima_examples::RtspStreamInfo& stream) {
  neat::SimaDecodeOptions decode;
  decode.type = config.choice("input.codec", kDecodeTypes);
  // Names the output buffer "decoder", the default Model::Options::upstream_name.
  decode.decoder_name = "decoder";
  decode.dec_width = stream.width;
  decode.dec_height = stream.height;
  decode.dec_fps = stream.fps;
  decode.num_buffers = kDecoderBuffers;
  neat::Graph graph("decoder");
  graph.add(neat::nodes::SimaDecode(std::move(decode)));
  graph.add(neat::nodes::CapsRaw("NV12", stream.width, stream.height, config.integer("input.fps")));
  graph.add(neat::nodes::Output("model_frame"));
  return graph;
}

// The newest decoded frame of one stream, which tells the pull loop the decoder is sending.
neat::Graph frames_graph(std::size_t index) {
  // Core gap: a Run reports no count of the frames a model holds, so the pull loop watches each
  // decoder's frames to tell a stalled model from a paused camera.
  neat::Graph graph;
  graph.add(neat::nodes::Output(frames_output(index), neat::OutputOptions::Latest()));
  return graph;
}

// Pass on the newest frame, so a slow consumer cannot hold back the stream.
//
// Into a model, Core lets a stream hold 4 decoded frames at once, and caps the frames on one link
// at min(4 * streams, 8), which is 4 for the one stream per model (EdgeRouter.cpp).
neat::GraphLinkOptions realtime_link() {
  neat::GraphLinkOptions link;
  link.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  return link;
}

// Send the encoded stream to Insight unchanged, on video port base + index.
groups::VideoSenderOptions video_options(const sima_examples::Config& config, std::size_t index) {
  auto options = groups::VideoSenderOptions::Passthrough(config.choice("input.codec", kRtspCodecs));
  options.host = config.string("output.insight.host");
  options.channel = static_cast<int>(index);
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
}

neat::MetadataSenderOptions metadata_options(const sima_examples::Config& config,
                                             std::size_t index) {
  neat::MetadataSenderOptions options;
  options.host = config.string("output.insight.host");
  options.channel = static_cast<int>(index);
  options.metadata_port_base = config.integer("output.insight.metadata_port_base");
  return options;
}

neat::RunOptions run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  // Auto copies a result that waits at an output while another one is queued there, unless it is
  // DMA-BUF or framework memory (RunCoreGraphStart.cpp, GraphRuntimeUtil.cpp).
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

// Each stream's frame size and rate: the configured values, and a probe's for each 0.
std::vector<sima_examples::RtspStreamInfo> stream_info(const sima_examples::Config& config,
                                                       const std::vector<std::string>& sources) {
  const sima_examples::RtspStreamInfo configured{
      config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")};
  const bool tcp = config.boolean("input.rtsp.tcp");
  std::map<std::string, sima_examples::RtspStreamInfo> by_url;
  for (const auto& url : sources) {
    if (by_url.count(url) == 0) {
      by_url[url] = sima_examples::probe_rtsp_stream_info(url, tcp, configured);
    }
  }
  std::vector<sima_examples::RtspStreamInfo> streams;
  for (const auto& url : sources) {
    streams.push_back(by_url.at(url));
  }
  return streams;
}

// The list at `key`, which holds one entry per input.source stream.
std::vector<std::string> per_stream(const sima_examples::Config& config, const std::string& key,
                                    std::size_t streams) {
  auto values = config.string_list(key);
  if (values.size() != streams) {
    throw std::runtime_error(key + " has " + std::to_string(values.size()) +
                             " entries and input.source has " + std::to_string(streams) +
                             ": give one per stream");
  }
  return values;
}

neat::TensorList sample_tensors(const neat::Sample& sample) {
  // Core candidate: Python has no binding of tensors_from_sample (GraphOptions.h), so both
  // languages pick the tensors out of the Sample here.
  if (sample.kind == neat::SampleKind::TensorSet && !sample.tensors.empty()) {
    return sample.tensors;
  }
  if (sample.kind == neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return {*sample.tensor};
  }
  throw std::runtime_error("expected a tensor result");
}

// The tensor's values, copied once. A frame with nothing found has zero rows.
template <typename T> std::vector<T> tensor_values(const neat::Tensor& tensor) {
  std::vector<T> values(tensor.dense_bytes_tight() / sizeof(T));
  if (!values.empty() &&
      !tensor.copy_dense_bytes_tight_to(reinterpret_cast<std::uint8_t*>(values.data()),
                                        values.size() * sizeof(T))) {
    throw std::runtime_error("cannot read a result tensor");
  }
  return values;
}

// The objects, segments, or poses of one results sample, in the stream's frame pixels.
std::vector<insight_metadata::Entry> metadata_entries(const neat::Sample& sample,
                                                      const Stream& stream,
                                                      const std::vector<std::string>& labels,
                                                      int max_detections) {
  const int width = stream.info.width;
  const int height = stream.info.height;
  const auto tensors = sample_tensors(sample);
  if (stream.metadata_type == kDetection) {
    const auto result =
        neat::decode_bbox_tensor(tensors.front(), width, height, max_detections, /*strict=*/true);
    return insight_metadata::objects(result.boxes, labels, width, height);
  }
  // A frame with nothing found decodes to tensors with zero rows.
  std::vector<insight_metadata::Entry> entries;
  if (stream.metadata_type == kSegmentation) {
    for (const auto& item : neat::decode_segmentation(tensors, width, height, max_detections,
                                                      /*strict=*/true)) {
      const auto segments = insight_metadata::segments(
          tensor_values<float>(item.boxes), tensor_values<std::uint8_t>(item.masks),
          static_cast<int>(item.masks.shape.at(2)), static_cast<int>(item.masks.shape.at(1)),
          labels, width, height);
      entries.insert(entries.end(), segments.begin(), segments.end());
    }
    return entries;
  }
  for (const auto& item :
       neat::decode_pose(tensors, width, height, max_detections, /*strict=*/true)) {
    const auto poses = insight_metadata::poses(tensor_values<float>(item.boxes),
                                               tensor_values<float>(item.keypoints));
    entries.insert(entries.end(), poses.begin(), poses.end());
  }
  return entries;
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Send every stream's results to its Insight channel until each stream sent `frames` (0: no limit)
// or a stop signal. A stream whose decoded frames arrive with no result for its deadline
// throws with its model.
void send_results(neat::Run& run, std::vector<Stream>& streams,
                  const std::vector<std::string>& labels, int max_detections, int frames,
                  const std::atomic<bool>& stop) {
  const auto all_sent = [&] {
    return frames > 0 && std::all_of(streams.begin(), streams.end(),
                                     [&](const Stream& s) { return s.processed >= frames; });
  };
  while (!stop && !all_sent()) {
    bool pulled = false;
    const double now = now_s();
    for (std::size_t index = 0; index < streams.size(); ++index) {
      Stream& stream = streams[index];
      if (frames > 0 && stream.processed >= frames) {
        continue;
      }
      neat::Sample sample;
      neat::PullError error;
      const std::string frames_name = frames_output(index);
      const bool frame_arrived = sima_examples::pull_status_has_sample(
          run.pull(frames_name, 0, sample, &error), frames_name, error, run.last_error());
      const std::string results_name = results_output(index);
      const bool result_arrived = sima_examples::pull_status_has_sample(
          run.pull(results_name, 0, sample, &error), results_name, error, run.last_error());
      stream.pending.observe(frame_arrived, result_arrived, now);
      if (!result_arrived) {
        continue;
      }
      pulled = true;
      const auto [data, left_out] =
          insight_metadata::data_json(kDataKeys.at(stream.metadata_type),
                                      metadata_entries(sample, stream, labels, max_detections));
      // A nonblocking send drops the message while the socket buffer is full; the drops are
      // counted in the sender's stats and reported when the run ends. Any other failure throws.
      std::string send_error;
      if (!stream.metadata_sender.send_metadata(
              stream.metadata_type, data, sample.pts_ns >= 0 ? sample.pts_ns / 1000000 : -1,
              sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "", &send_error) &&
          !send_error.empty()) {
        throw std::runtime_error("metadata send failed: " + send_error);
      }
      ++stream.processed;
      stream.left_out += left_out;
    }
    if (!pulled) {
      std::this_thread::sleep_for(kIdleSleep);
    }
  }
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  const auto sources = config.string_list("input.source");
  const auto infos = stream_info(config, sources);
  const auto model_paths = per_stream(config, "app.model_paths", sources.size());
  const auto model_ids = per_stream(config, "app.model_ids", sources.size());
  // Each stream's BoxDecode type and metadata type, read before any model loads.
  std::vector<std::pair<neat::BoxDecodeType, std::string>> model_types;
  for (const auto& model_id : model_ids) {
    model_types.push_back(sima_examples::lookup("app.model_ids", model_id, kModelIds));
  }
  const auto labels = sima_examples::load_labels(config.string("model.labels"));
  // Used by both the models and the box decode.
  const int max_detections = config.integer("inference.max_detections");
  // Results each stream sends before the app stops. 0 runs until stopped.
  const int frames = config.integer("inference.frames");
  if (frames < 0) {
    throw std::runtime_error("inference.frames must be 0 or more");
  }

  neat::Graph graph;
  std::deque<neat::Model> models;
  std::vector<Stream> streams;
  for (std::size_t index = 0; index < sources.size(); ++index) {
    const auto& url = sources[index];
    const auto& info = infos[index];
    const auto& [decode_type, metadata_type] = model_types[index];
    const fs::path model_path = fs::absolute(config_path).parent_path() / model_paths[index];
    models.emplace_back(model_path.string(),
                        model_options(config, info, decode_type, max_detections));
    const neat::Graph rtsp = groups::RtspEncodedInput(rtsp_options(config, url, info));
    const neat::Graph decoder = decoder_graph(config, info);
    // A plain link keeps every encoded frame. A dropped one corrupts the decoded frames after it
    // until the next keyframe. Core runs an RTSP input with exactly two consumers, a decoder and
    // an encoded VideoSender, as one pipeline (ExecutionGraphCompiler.cpp), so the frames output
    // taps the decoder instead.
    graph.connect(rtsp, decoder);
    graph.connect(decoder, model_graph(models.back(), index), realtime_link());
    graph.connect(rtsp, groups::VideoSender(video_options(config, index)), realtime_link());
    graph.connect(decoder, frames_graph(index), realtime_link());
    std::string error;
    neat::MetadataSender metadata_sender(metadata_options(config, index), &error);
    sima_examples::require(metadata_sender.ok(), error);
    streams.push_back(Stream{
        info, metadata_type, std::move(metadata_sender),
        pending_results::PendingResults(static_cast<int>(index), model_paths[index],
                                        kResultDeadlineS, kFirstResultDeadlineS)});
    std::cout << "[stream " << index << "] " << model_ids[index] << " model=" << model_paths[index]
              << " rtsp=" << url << " stream=" << info.width << "x" << info.height << "@"
              << info.fps << " insight=" << config.string("output.insight.host")
              << " video=" << config.integer("output.insight.video_port_base") + index
              << " metadata=" << streams.back().metadata_sender.metadata_port() << "\n";
  }

  const std::atomic<bool>& stop = sima_examples::stop_on_signals();
  // The Run's destructor closes it when send_results throws.
  neat::Run run = graph.build(run_options());
  send_results(run, streams, labels, max_detections, frames, stop);
  run.close();

  for (std::size_t index = 0; index < streams.size(); ++index) {
    const Stream& stream = streams[index];
    std::cout << "[stream " << index << "] processed=" << stream.processed
              << " left_out=" << stream.left_out << "\n";
    const auto failures = stream.metadata_sender.stats().send_failures;
    if (failures > 0) {
      std::cout << "[stream " << index << "] " << failures << " metadata messages dropped\n";
    }
  }
  std::cout << std::flush;
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
