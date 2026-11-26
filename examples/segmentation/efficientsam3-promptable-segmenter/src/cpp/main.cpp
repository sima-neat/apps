// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

// Segments what a text prompt names in one RTSP stream with EfficientSAM3 and shows it in
// Insight. The text encoder turns the prompt into text features once, at startup. The decoded
// frames go to Insight's video and to the host, which hands the newest one, with the text
// features, to the segmentation model. Insight receives every frame's instance outlines.

#include "neat.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/clip_tokenizer.h"
#include "utils/efficientsam3.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <exception>
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

// The text encoder package sits next to the segmentation model that model.path names.
constexpr const char* kTextEncoderPackage = "efficientsam3_text_encoder_mpk.tar.gz";
const std::map<std::string, groups::RtspCodec> kRtspCodecs = {
    {"h264", groups::RtspCodec::H264},
    {"h265", groups::RtspCodec::H265},
};
constexpr int kPullTimeoutMs = 100;
// The text encoder reads 16 tokens, each one-hot over the CLIP vocabulary, and returns 256
// features per token.
constexpr int kTextTokens = 16;
constexpr int kVocabSize = 49408;
constexpr int kTextDim = 256;
// Three frames on the model keep the MLA busy while the host prepares and reads frames (13 FPS;
// 11 FPS with two).
constexpr int kPipelineDepth = 3;

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

// A model input in EV74 memory, copied from the values.
neat::Tensor ev74_tensor(const std::vector<float>& values, std::vector<int64_t> shape) {
  neat::Tensor tensor =
      neat::Tensor::from_vector(values, std::move(shape), neat::TensorMemory::EV74);
  tensor.layout = neat::TensorLayout::HWC;
  return tensor;
}

// The segmentation model's text input: the encoder's features of each token, then a row that
// marks the prompt's tokens.
std::vector<float> text_features(neat::Model& text_encoder, const std::vector<int64_t>& tokens) {
  // The MLA cannot look up token ids, so the text encoder takes them one-hot.
  std::vector<float> one_hot(static_cast<std::size_t>(kTextTokens) * kVocabSize, 0.0F);
  for (int i = 0; i < kTextTokens; ++i) {
    one_hot[static_cast<std::size_t>(i) * kVocabSize + tokens[i]] = 1.0F;
  }
  const neat::Tensor input = ev74_tensor(one_hot, {1, kTextTokens, kVocabSize});
  neat::Model::Runner encoder = text_encoder.build(neat::TensorList{input});
  encoder.push(neat::TensorList{input});
  const std::vector<float> features = sima_examples::tensor_to_floats(encoder.pull().tensors.at(0));
  encoder.close();
  std::vector<float> text(static_cast<std::size_t>(kTextTokens + 1) * kTextDim, 0.0F);
  std::copy_n(features.begin(), kTextTokens * kTextDim, text.begin());
  for (int i = 0; i < kTextTokens; ++i) {
    text[kTextTokens * kTextDim + i] = tokens[i] != 0 ? 1.0F : 0.0F;
  }
  return text;
}

neat::RunOptions segmenter_run_options() {
  neat::RunOptions options;
  options.queue_depth = kPipelineDepth;
  return options;
}

// Receive the stream and decode it to NV12 frames of the stream's size.
groups::RtspDecodedInputOptions rtsp_options(const sima_examples::Config& config,
                                             const sima_examples::RtspStreamInfo& stream) {
  groups::RtspDecodedInputOptions options;
  options.url = config.string("input.source");
  options.codec = config.choice("input.codec", kRtspCodecs);
  options.tcp = config.boolean("input.rtsp.tcp");
  options.latency_ms = config.integer("input.rtsp.latency_ms");
  if (options.codec == groups::RtspCodec::H264) {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
    options.fallback_h264_fps = stream.fps;
  } else {
    options.dec_width = stream.width;
    options.dec_height = stream.height;
    options.source_fps = stream.fps;
  }
  return options;
}

// The decoded frames, encoded as H.264 for Insight.
groups::VideoSenderOptions video_options(const sima_examples::Config& config,
                                         const sima_examples::RtspStreamInfo& stream) {
  neat::SimaEncodeOptions encode;
  encode.fps = stream.fps;
  auto options = groups::VideoSenderOptions::FromRaw(encode);
  options.host = config.string("output.insight.host");
  options.video_port_base = config.integer("output.insight.video_port_base");
  return options;
}

// Hold up to 8 decoded frames for the host, dropping the oldest when the host falls behind.
neat::OutputOptions frames_options() {
  neat::OutputOptions options = neat::OutputOptions::EveryFrame(8);
  options.drop = true;
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
  // One more queued frame than the Realtime preset's 2.
  options.queue_depth = 3;
  return options;
}

efficientsam3::SegmentOptions segment_options(const sima_examples::Config& config,
                                              const sima_examples::RtspStreamInfo& stream,
                                              const std::string& prompt) {
  efficientsam3::SegmentOptions options;
  options.width = stream.width;
  options.height = stream.height;
  options.label = prompt;
  options.min_score = config.number("inference.min_score");
  options.max_detections = config.integer("inference.max_detections");
  options.mask_threshold = config.number("inference.mask_threshold");
  return options;
}

// The frame's NV12 bytes as height * 3 / 2 rows of width bytes.
cv::Mat nv12(const neat::Tensor& tensor) {
  cv::Mat frame(tensor.height() * 3 / 2, tensor.width(), CV_8UC1);
  // contiguous() drops the row padding the decoder adds to some frames.
  sima_examples::require(tensor.contiguous().copy_payload_bytes_to(frame.data, frame.total()),
                         "decoded frame is not NV12");
  return frame;
}

// The newest decoded frame, waiting up to `timeout_ms` for the first. Every frame pulled joins
// the overlay, so it is sent a result too.
std::optional<neat::Sample> newest_frame(neat::Run& run, efficientsam3::OverlayClock& overlay,
                                         int timeout_ms) {
  std::optional<neat::Sample> newest;
  while (true) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("frames", timeout_ms, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "frames", error, run.last_error())) {
      return newest;
    }
    overlay.add_frame(sample.pts_ns / 1'000'000, std::to_string(sample.frame_id));
    newest = std::move(sample);
    timeout_ms = 0;
  }
}

// Segment the newest frame whenever the model has room, and send every frame the nearest result,
// until `frames` results (0: no limit) or a stop signal; return the count.
int segment_frames(neat::Run& run, neat::Model::Runner& segmenter, const neat::Tensor& text,
                   const neat::MetadataSender& metadata_sender,
                   const efficientsam3::SegmentOptions& options, int frames,
                   const std::atomic<bool>& stop) {
  // Each frame on the model keeps its own image tensor, and one more is filled meanwhile.
  std::vector<neat::Tensor> images;
  for (int i = 0; i <= kPipelineDepth; ++i) {
    images.push_back(ev74_tensor(
        std::vector<float>(efficientsam3::kModelSize * efficientsam3::kModelSize * 3, 0.0F),
        {efficientsam3::kModelSize, efficientsam3::kModelSize, 3}));
  }
  efficientsam3::OverlayClock overlay;
  std::deque<int64_t> in_flight; // The presentation time, in ms, of each frame on the model.
  int pushed = 0;
  int processed = 0;
  while (!stop && (frames == 0 || processed < frames)) {
    if (static_cast<int>(in_flight.size()) < kPipelineDepth) {
      const std::optional<neat::Sample> sample =
          newest_frame(run, overlay, in_flight.empty() ? kPullTimeoutMs : 0);
      if (sample) {
        const neat::Tensor& image = images[pushed % images.size()];
        {
          neat::Mapping mapping = image.map_write();
          cv::Mat out(efficientsam3::kModelSize, efficientsam3::kModelSize, CV_32FC3, mapping.data);
          efficientsam3::preprocess(nv12(sample->tensors.at(0)), out);
        }
        segmenter.push(neat::TensorList{image, text});
        ++pushed;
        in_flight.push_back(sample->pts_ns / 1'000'000);
        continue;
      }
      if (in_flight.empty()) {
        continue;
      }
    }

    const neat::Sample result = segmenter.pull();
    sima_examples::require(result.tensors.size() == 2, "the segmentation model returned no result");
    const neat::Tensor& detections = result.tensors[0];
    const neat::Tensor& masks = result.tensors[1];
    const neat::Mapping detections_view = detections.view_read();
    const neat::Mapping masks_view = masks.view_read();
    const std::size_t mask_rank = masks.shape.size();
    const auto queries = static_cast<int>(masks.shape.at(mask_rank - 1));
    const cv::Mat detection_rows(queries, static_cast<int>(detections.shape.back()), CV_32F,
                                 detections_view.data);
    const cv::Mat mask_planes(static_cast<int>(masks.shape.at(mask_rank - 3)),
                              static_cast<int>(masks.shape.at(mask_rank - 2)), CV_32FC(queries),
                              masks_view.data);
    const std::string data = efficientsam3::segments_json(
        efficientsam3::segments_of(detection_rows, mask_planes, options));
    for (const auto& message : overlay.add_result(in_flight.front(), data)) {
      // A nonblocking send drops the message while the socket buffer is full; the drops are
      // counted in the sender's stats and reported when the run ends. Any other failure throws.
      std::string send_error;
      if (!metadata_sender.send_metadata("segmentation", message.data, message.timestamp_ms,
                                         message.frame_id, &send_error) &&
          !send_error.empty()) {
        throw std::runtime_error("metadata send failed: " + send_error);
      }
    }
    in_flight.pop_front();
    ++processed;
  }
  return processed;
}

int run_app(const fs::path& config_path) {
  const auto config = sima_examples::Config::load(config_path);
  // Check the prompt and the stream first: loading the two models takes about a minute.
  const std::string prompt = config.string("app.prompt");
  // The CLIP vocabulary ships in src/common/, next to the default config.
  const ClipTokenizer tokenizer(
      sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() /
      "bpe_simple_vocab_16e6.txt.gz");
  const std::vector<int64_t> tokens = prompt_tokens(tokenizer, prompt, kTextTokens);
  // The stream's size and rate come from the probe for each value the config leaves at 0.
  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  const fs::path model_path = fs::absolute(config_path).parent_path() / config.string("model.path");
  neat::Model text_encoder((model_path.parent_path() / kTextEncoderPackage).string());
  const neat::Tensor text =
      ev74_tensor(text_features(text_encoder, tokens), {1, kTextTokens + 1, kTextDim});
  const neat::Tensor image = ev74_tensor(
      std::vector<float>(efficientsam3::kModelSize * efficientsam3::kModelSize * 3, 0.0F),
      {efficientsam3::kModelSize, efficientsam3::kModelSize, 3});
  neat::Model model(model_path.string());
  neat::Model::Runner segmenter = model.build(neat::TensorList{image, text},
                                              neat::Model::RouteOptions{}, segmenter_run_options());

  auto source = groups::RtspDecodedInput(rtsp_options(config, stream));
  neat::Graph graph;
  graph.connect(source, groups::VideoSender(video_options(config, stream)));
  graph.connect(source, neat::nodes::Output("frames", frames_options()));

  std::string metadata_error;
  neat::MetadataSender metadata_sender(metadata_options(config), &metadata_error);
  sima_examples::require(metadata_sender.ok(), metadata_error);
  std::atomic<bool>& stop = sima_examples::stop_on_signals();
  neat::Run run = graph.build(run_options());
  const std::string host = config.string("output.insight.host");
  const int video_port = config.integer("output.insight.video_port_base");
  std::cout << "source=" << config.string("input.source")
            << " codec=" << config.string("input.codec") << " stream=" << stream.width << "x"
            << stream.height << "@" << stream.fps << " prompt='" << prompt << "' tokens="
            << std::count_if(tokens.begin(), tokens.end(), [](int64_t id) { return id != 0; })
            << " insight=" << host << " video=" << video_port
            << " metadata=" << config.integer("output.insight.metadata_port_base") << std::endl;
  int processed = 0;
  std::exception_ptr segment_error;
  try {
    processed = segment_frames(run, segmenter, text, metadata_sender,
                               segment_options(config, stream, prompt),
                               config.integer("inference.frames"), stop);
  } catch (...) {
    segment_error = std::current_exception();
  }
  run.close();
  segmenter.close();
  if (segment_error) {
    std::rethrow_exception(segment_error);
  }

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
