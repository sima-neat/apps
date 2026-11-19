// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @example clip-zero-shot-classifier.cpp
 * Zero-shot classification of one RTSP stream with CLIP ViT-B/16 and Insight output.
 */
#include "clip_tokenizer.h"
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <nodes/groups/VideoSender.h>
#include <nodes/io/MetadataSender.h>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kPullTimeoutMs = 20000;
constexpr int kTextTokens = 77;
constexpr int kEmbedDim = 512;
constexpr int kVocabSize = 49408;
// The checkpoint's logit_scale.exp(): CLIP scores a prompt as 100 times its cosine similarity with
// the image.
constexpr float kLogitScale = 100.0F;
constexpr int kTopPrompts = 3;
constexpr int kModelSize = 224;
// CLIP's per-channel mean and std, from the model's README. Neat computes
// (pixel / 255 - mean) / stddev.
constexpr std::array<float, 3> kMean = {0.48145466F, 0.4578275F, 0.40821073F};
constexpr std::array<float, 3> kStd = {0.26862954F, 0.26130258F, 0.27577711F};
const std::map<std::string, neat::nodes::groups::RtspCodec> kRtspCodecs = {
    {"h264", neat::nodes::groups::RtspCodec::H264},
    {"h265", neat::nodes::groups::RtspCodec::H265},
    {"mjpeg", neat::nodes::groups::RtspCodec::MJPEG}};
// The result panel at the bottom left of the video, in pixels of a 720-line frame. Other heights
// scale it.
constexpr int kPanelLines = 720;
constexpr int kPanelMargin = 24;
// The bottom margin keeps the panel above the status line the Insight viewer draws over the video.
constexpr int kPanelBottom = 56;
constexpr int kPanelWidth = 640;
constexpr int kPanelPadding = 16;
constexpr int kTextBarGap = 10;
constexpr int kRowGap = 14;
struct PanelRowStyle {
  double font_scale;
  int thickness;
  int bar_height;
};
// Font scale, line thickness and bar height of each row, best prompt first.
constexpr std::array<PanelRowStyle, kTopPrompts> kPanelRows = {
    {{0.75, 2, 14}, {0.65, 2, 8}, {0.65, 2, 8}}};
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
// The panel keeps this share of the luma and of the chroma's distance from neutral gray.
constexpr double kPanelDim = 0.3;
constexpr int kBestTextLuma = 235;
constexpr int kTextLuma = 200;
constexpr int kTrackLuma = 60;
constexpr int kBarLuma = 170;
// Amber (RGB 255, 170, 0) in limited-range BT.601 YUV.
constexpr std::array<int, 3> kAccentYuv = {167, 41, 178};
constexpr int kNeutralChroma = 128;

volatile std::sig_atomic_t g_stop_requested = 0;

void request_stop(int) {
  g_stop_requested = 1;
}

struct Config {
  std::string image_encoder;
  std::string text_encoder;
  std::string tokenizer;
  std::string token_embeddings;
  std::string prompts_path;
  std::string rtsp_url;
  std::string codec;
  bool tcp = true;
  int latency_ms = 100;
  int frames = 0;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host;
  int video_port = 9000;
  int metadata_port = 9100;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.image_encoder = raw.string_or("model.image_encoder", "");
  cfg.text_encoder = raw.string_or("model.text_encoder", "");
  cfg.tokenizer = raw.string_or("model.tokenizer", "");
  cfg.token_embeddings = raw.string_or("model.token_embeddings", "");
  cfg.prompts_path = raw.string_or("prompts.path", "");
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.codec = raw.string_or("source.codec", "h264");
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  sima_examples::require(cfg.profile_interval > 0, "runtime.profile_interval must be > 0");
  sima_examples::require(kRtspCodecs.count(cfg.codec) == 1,
                         "source.codec must be h264, h265 or mjpeg");
  return cfg;
}

std::vector<std::string> load_prompts(const std::string& path) {
  std::ifstream file(path);
  sima_examples::require(fs::is_regular_file(path) && file.is_open(),
                         "prompts.path " + path +
                             " does not exist; point it at a text file with one prompt per line");
  std::vector<std::string> prompts;
  for (std::string line; std::getline(file, line);) {
    if (const std::string prompt = sima_examples::trim_copy(line); !prompt.empty()) {
      prompts.push_back(prompt);
    }
  }
  // A softmax over one prompt always gives it 100%.
  sima_examples::require(prompts.size() >= 2,
                         path + " has " + std::to_string(prompts.size()) +
                             " prompts; write at least two, one per line, to compare them");
  return prompts;
}

// The (prompts, 77) token ids, rejecting a prompt the text encoder would not see whole.
std::vector<std::vector<int64_t>> prompt_tokens(const std::string& tokenizer_path,
                                                const std::vector<std::string>& prompts) {
  sima_examples::require(
      fs::is_regular_file(tokenizer_path),
      "model.tokenizer " + tokenizer_path +
          " does not exist; point it at text_encoder/tokenizer.json in the model "
          "package");
  const ClipTokenizer tokenizer(tokenizer_path);
  std::vector<std::vector<int64_t>> tokens;
  for (const std::string& prompt : prompts) {
    const std::size_t count = tokenizer.tokens(prompt).size();
    sima_examples::require(count <= static_cast<std::size_t>(kTextTokens - 2),
                           "prompt '" + prompt + "' is " + std::to_string(count) +
                               " tokens; the text encoder reads at most " +
                               std::to_string(kTextTokens - 2) + ", so shorten it");
    tokens.push_back(tokenizer.encode(prompt, kTextTokens));
  }
  return tokens;
}

// Where the rows of the float32 (49408, 512) token-embedding table start in its .npy file.
struct TokenEmbeddings {
  std::string path;
  std::streamoff data_offset = 0;
};

TokenEmbeddings load_token_embeddings(const std::string& path) {
  sima_examples::require(fs::is_regular_file(path),
                         "model.token_embeddings " + path +
                             " does not exist; point it at text_encoder/token_embeddings.npy in "
                             "the model package");
  // Only the header is read here: the prompts read only their own rows of the 100 MB table.
  std::ifstream file(path, std::ios::binary);
  std::string magic(8, '\0');
  file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  sima_examples::require(file && magic.compare(0, 6, "\x93NUMPY") == 0,
                         "model.token_embeddings " + path +
                             " is not a NumPy .npy file; download the model package again");
  // Version 1 stores the header length in 2 bytes, later versions in 4, little-endian.
  const int length_bytes = magic[6] == 1 ? 2 : 4;
  std::size_t header_size = 0;
  for (int i = 0; i < length_bytes; ++i) {
    header_size |= static_cast<std::size_t>(static_cast<unsigned char>(file.get())) << (8 * i);
  }
  std::string header(header_size, '\0');
  file.read(header.data(), static_cast<std::streamsize>(header.size()));
  const std::size_t shape_at = header.find("'shape': (");
  sima_examples::require(file && shape_at != std::string::npos,
                         "model.token_embeddings " + path +
                             " has no .npy shape header; download the model package again");
  const std::size_t shape_start = header.find('(', shape_at);
  const std::string shape =
      header.substr(shape_start, header.find(')', shape_start) - shape_start + 1);
  const std::string expected =
      "(" + std::to_string(kVocabSize) + ", " + std::to_string(kEmbedDim) + ")";
  sima_examples::require(shape == expected,
                         "model.token_embeddings " + path + " has shape " + shape + ", not " +
                             expected + "; point it at CLIP ViT-B/16's token-embedding table");
  sima_examples::require(header.find("'descr': '<f4'") != std::string::npos &&
                             header.find("'fortran_order': False") != std::string::npos,
                         "model.token_embeddings " + path +
                             " is not little-endian float32 in C order; save it with numpy.save");
  return {path, static_cast<std::streamoff>(8 + length_bytes + header_size)};
}

// The (77, 512) token-embedding rows of the prompt's token ids.
std::vector<float> token_embeds(const TokenEmbeddings& table, const std::vector<int64_t>& ids) {
  std::ifstream file(table.path, std::ios::binary);
  std::vector<float> rows(ids.size() * kEmbedDim);
  const auto row_bytes = static_cast<std::streamsize>(kEmbedDim * sizeof(float));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    file.seekg(table.data_offset + ids[i] * row_bytes);
    file.read(reinterpret_cast<char*>(rows.data() + i * kEmbedDim), row_bytes);
  }
  sima_examples::require(static_cast<bool>(file),
                         "could not read " + table.path + "; download the model package again");
  return rows;
}

int64_t element_count(const neat::Tensor& tensor) {
  return std::accumulate(tensor.shape.begin(), tensor.shape.end(), int64_t{1}, std::multiplies<>());
}

neat::Model::Options text_encoder_options() {
  // The input is the prompt's token embeddings, not an image.
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Tensor;
  options.preprocess.resize.enable = neat::AutoFlag::Off;
  options.preprocess.color_convert.enable = neat::AutoFlag::Off;
  options.preprocess.normalize.enable = neat::AutoFlag::Off;
  return options;
}

// The unit-length text encoder row at the prompt's first end-of-text token, the highest id in the
// vocabulary.
std::vector<float> end_of_text_row(const float* rows, const std::vector<int64_t>& ids) {
  const auto end = std::max_element(ids.begin(), ids.end()) - ids.begin();
  std::vector<float> row(rows + end * kEmbedDim, rows + (end + 1) * kEmbedDim);
  const float norm = std::sqrt(std::inner_product(row.begin(), row.end(), row.begin(), 0.0F));
  for (float& value : row) {
    value /= norm;
  }
  return row;
}

neat::Tensor ev74_tensor(const std::vector<float>& values) {
  // Neat sizes the input as height 1, width 77 and depth 512, so the tensor declares HWC axes.
  neat::Tensor tensor =
      neat::Tensor::from_vector(values, {1, kTextTokens, kEmbedDim}, neat::TensorMemory::EV74);
  tensor.layout = neat::TensorLayout::HWC;
  return tensor;
}

// The (prompts, 512) unit-length text embeddings, one text encoder run per prompt.
std::vector<std::vector<float>> encode_prompts(const Config& cfg,
                                               const std::vector<std::vector<int64_t>>& tokens,
                                               const TokenEmbeddings& table) {
  neat::Model model(cfg.text_encoder, text_encoder_options());
  neat::Model::Runner runner =
      model.build(neat::TensorList{ev74_tensor(token_embeds(table, tokens.front()))});
  std::vector<std::vector<float>> vectors;
  for (const std::vector<int64_t>& ids : tokens) {
    const neat::TensorList outputs =
        runner.run(neat::TensorList{ev74_tensor(token_embeds(table, ids))}, kPullTimeoutMs);
    const neat::Tensor& rows = outputs.at(0);
    sima_examples::require(rows.dtype == neat::TensorDType::Float32 &&
                               element_count(rows) == int64_t{kTextTokens} * kEmbedDim,
                           "model.text_encoder " + cfg.text_encoder +
                               " did not return 77 x 512 float32 values; point it at CLIP "
                               "ViT-B/16's text encoder");
    const neat::Mapping view = rows.view_read();
    vectors.push_back(end_of_text_row(static_cast<const float*>(view.data), ids));
  }
  runner.close();
  return vectors;
}

// The codec of an OpenCV FOURCC value, or "" when it is not one this application decodes.
std::string fourcc_codec(double fourcc) {
  std::string tag;
  for (int i = 0; i < 4; ++i) {
    tag += static_cast<char>(std::tolower((static_cast<int>(fourcc) >> (8 * i)) & 0xFF));
  }
  if (tag == "h264" || tag == "avc1") {
    return "h264";
  }
  if (tag == "hevc" || tag == "hev1" || tag == "hvc1" || tag == "h265") {
    return "h265";
  }
  if (tag == "mjpg" || tag == "jpeg") {
    return "mjpeg";
  }
  return "";
}

sima_examples::RtspStreamInfo probe_stream(const Config& cfg) {
  if (cfg.tcp) {
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 1);
  }
  cv::VideoCapture capture(cfg.rtsp_url, cv::CAP_FFMPEG);
  const sima_examples::RtspStreamInfo stream{
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)),
      static_cast<int>(std::nearbyint(capture.get(cv::CAP_PROP_FPS)))};
  const std::string stream_codec = fourcc_codec(capture.get(cv::CAP_PROP_FOURCC));
  capture.release();
  sima_examples::require(stream.width > 0 && stream.height > 0,
                         "could not read the frame size of " + cfg.rtsp_url +
                             "; check source.rtsp_url and that the stream is running");
  sima_examples::require(stream.fps > 0,
                         cfg.rtsp_url +
                             " does not report its frame rate; use a stream that reports its "
                             "frame rate");
  // The decoder accepts only the configured codec. Another codec would fail only after the models
  // load.
  sima_examples::require(stream_codec.empty() || stream_codec == cfg.codec,
                         cfg.rtsp_url + " is " + stream_codec + ", but source.codec is " +
                             cfg.codec + "; set source.codec to " + stream_codec);
  return stream;
}

neat::nodes::groups::RtspDecodedInputOptions
rtsp_options(const Config& cfg, const sima_examples::RtspStreamInfo& stream) {
  neat::nodes::groups::RtspDecodedInputOptions options;
  options.url = cfg.rtsp_url;
  options.tcp = cfg.tcp;
  options.latency_ms = cfg.latency_ms;
  options.codec = kRtspCodecs.at(cfg.codec);
  options.source_fps = stream.fps;
  if (cfg.codec == "h264") {
    options.fallback_h264_width = stream.width;
    options.fallback_h264_height = stream.height;
  } else {
    options.dec_width = stream.width;
    options.dec_height = stream.height;
  }
  auto& caps = options.output_caps;
  caps.enable = true;
  caps.width = stream.width;
  caps.height = stream.height;
  caps.fps = stream.fps;
  return options;
}

neat::Model::Options image_encoder_options(const sima_examples::RtspStreamInfo& stream) {
  // Neat stretches the decoded NV12 frame to the model's 224x224 input and applies CLIP's mean and
  // std on the board.
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  // Without a size Neat infers it from the model and prints a planner warning at every start.
  options.preprocess.resize.width = options.preprocess.resize.height = kModelSize;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  options.preprocess.normalize.mean = kMean;
  options.preprocess.normalize.stddev = kStd;
  options.preprocess.normalize.has_explicit_stats = true;
  return options;
}

neat::InputOptions video_input_options(const sima_examples::RtspStreamInfo& stream) {
  // The application pushes each NV12 frame after drawing the result panel into it.
  neat::InputOptions options;
  options.payload_type = neat::PayloadType::Image;
  options.format = "NV12";
  options.width = stream.width;
  options.height = stream.height;
  options.depth = 1;
  options.fps_n = stream.fps;
  options.fps_d = 1;
  return options;
}

neat::nodes::groups::VideoSenderOptions video_options(const Config& cfg, int fps) {
  neat::SimaEncodeOptions encode;
  encode.fps = fps;
  auto options = neat::nodes::groups::VideoSenderOptions::FromRaw(encode);
  options.host = cfg.insight_host;
  options.video_port_base = cfg.video_port;
  return options;
}

neat::MetadataSenderOptions metadata_options(const Config& cfg) {
  neat::MetadataSenderOptions options;
  options.host = cfg.insight_host;
  options.metadata_port_base = cfg.metadata_port;
  return options;
}

neat::RunOptions run_options() {
  neat::RunOptions options;
  options.preset = neat::RunPreset::Realtime;
  options.queue_depth = 3;
  options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  options.output_memory = neat::OutputMemory::ZeroCopy;
  return options;
}

struct Graphs {
  neat::Graph classifier;
  neat::Graph video;
};

// The classifier graph, RTSP source -> decoded frames -> {"frame", image encoder -> "embeddings"}
// joined per frame into "result", and the video graph, pushed frames -> Insight video sender.
Graphs build_graphs(const Config& cfg, const neat::Model& image_encoder,
                    const sima_examples::RtspStreamInfo& stream) {
  neat::Graph branch = neat::graphs::Branch("source", {"frame", "model"});
  neat::Graph frame("frame");
  frame.add(neat::nodes::Output("frame", neat::OutputOptions::EveryFrame(4)));
  neat::Graph model("model");
  model.connect(neat::nodes::Input("model"), image_encoder);
  neat::Graph embeddings("embeddings");
  embeddings.add(neat::nodes::Output("embeddings", neat::OutputOptions::EveryFrame(4)));
  neat::Graph result =
      neat::graphs::Combine({"frame", "embeddings"}, "result", neat::CombinePolicy::ByFrame);

  Graphs graphs;
  graphs.classifier.connect(neat::nodes::groups::RtspDecodedInput(rtsp_options(cfg, stream)),
                            branch);
  graphs.classifier.connect(branch, frame);
  graphs.classifier.connect(branch, model);
  graphs.classifier.connect(model, embeddings);
  graphs.classifier.connect(frame, result);
  graphs.classifier.connect(embeddings, result);

  graphs.video = neat::Graph("video");
  graphs.video.connect(neat::nodes::Input(video_input_options(stream)),
                       neat::nodes::groups::VideoSender(video_options(cfg, stream.fps)));
  return graphs;
}

// The image encoder's 512-value embedding, found by its size: some packages also return 1000
// ImageNet logits.
neat::Tensor image_embedding(const neat::Sample& sample) {
  std::string sizes;
  for (const neat::Tensor& tensor : neat::tensors_from_sample(sample, false)) {
    const int64_t size = element_count(tensor);
    if (size == kEmbedDim) {
      return tensor;
    }
    sizes += (sizes.empty() ? "" : ", ") + std::to_string(size);
  }
  throw std::runtime_error("model.image_encoder returned outputs of sizes [" + sizes +
                           "], none of " + std::to_string(kEmbedDim) +
                           " values; point it at CLIP ViT-B/16's image encoder");
}

// The field of a joined sample that came from the named output.
const neat::Sample& field(const neat::Sample& sample, const std::string& label) {
  for (const neat::Sample& part : sample.fields) {
    if (part.stream_label == label) {
      return part;
    }
  }
  throw std::runtime_error("the joined sample has no " + label + " field");
}

// A sample of the NV12 frame with the source frame's timestamp and id, which Insight matches the
// metadata to.
neat::Sample nv12_sample(const cv::Mat& nv12, const neat::Sample& source) {
  const int height = nv12.rows * 2 / 3;
  const int width = nv12.cols;
  neat::Tensor tensor = neat::Tensor::from_cv_mat_view(nv12, neat::ImageSpec::PixelFormat::GRAY8);
  tensor.semantic.image = neat::ImageSpec{neat::ImageSpec::PixelFormat::NV12, ""};
  tensor.shape = {height, width};
  tensor.strides_bytes.clear();
  tensor.axis_semantics = {neat::TensorAxisSemantic::H, neat::TensorAxisSemantic::W};
  tensor.planes = {{neat::PlaneRole::Y, {height, width}, {width, 1}, 0},
                   {neat::PlaneRole::UV, {height / 2, width}, {width, 1}, int64_t{height} * width}};
  neat::Sample sample = neat::make_tensor_sample("video", tensor);
  sample.pts_ns = source.pts_ns;
  sample.frame_id = source.frame_id;
  return sample;
}

struct TopPrompt {
  std::string label;
  double confidence = 0.0;
  double similarity = 0.0;
};

// Python's round(value, 4).
double round4(float value) {
  return std::nearbyint(static_cast<double>(value) * 1e4) / 1e4;
}

// The best prompts by softmax probability over all prompts, with their cosine similarity.
std::vector<TopPrompt> top_prompts(const float* image,
                                   const std::vector<std::vector<float>>& prompt_vectors,
                                   const std::vector<std::string>& prompts) {
  const float image_norm = std::sqrt(std::inner_product(image, image + kEmbedDim, image, 0.0F));
  std::vector<float> similarity;
  std::vector<float> logits;
  for (const std::vector<float>& prompt : prompt_vectors) {
    similarity.push_back(std::inner_product(prompt.begin(), prompt.end(), image, 0.0F) /
                         image_norm);
    logits.push_back(kLogitScale * similarity.back());
  }
  // topk_with_softmax breaks ties by index, so equal scores keep prompts in file order.
  std::vector<TopPrompt> top;
  for (const sima_examples::ScoredIndex& best :
       sima_examples::topk_with_softmax(logits, kTopPrompts)) {
    top.push_back({prompts[best.index], round4(best.prob), round4(similarity[best.index])});
  }
  return top;
}

// The classification metadata Insight draws.
std::string top_classes_json(const std::vector<TopPrompt>& top) {
  nlohmann::json items = nlohmann::json::array();
  for (const TopPrompt& prompt : top) {
    items.push_back({{"label", prompt.label},
                     {"confidence", prompt.confidence},
                     {"similarity", prompt.similarity}});
  }
  return nlohmann::json{{"top_classes", items}}.dump();
}

struct PanelRow {
  double font_scale = 0.0;
  int thickness = 0;
  int baseline_y = 0;
  int label_width = 0;
  // The bar at 100%.
  cv::Rect bar;
};

struct PanelLayout {
  // The darkened area.
  cv::Rect box;
  std::vector<PanelRow> rows;
};

// The nearest even pixel count: NV12 has one chroma sample per 2x2 pixels.
int even(double value) {
  return static_cast<int>(value / 2 + 0.5) * 2;
}

int text_width(const std::string& text, double font_scale, int thickness) {
  int baseline = 0;
  return cv::getTextSize(text, kFont, font_scale, thickness, &baseline).width;
}

// Where the result panel of `rows` prompts goes in a width x height frame.
PanelLayout panel_layout(int width, int height, int rows) {
  const double scale = static_cast<double>(height) / kPanelLines;
  const int margin = even(kPanelMargin * scale);
  const int padding = even(kPanelPadding * scale);
  const int panel_width = std::min(even(kPanelWidth * scale), width - 2 * margin);
  const int inner_x = margin + padding;
  const int inner_width = panel_width - 2 * padding;
  PanelLayout layout;
  int y = padding;
  for (int i = 0; i < rows; ++i) {
    PanelRow row;
    row.font_scale = kPanelRows[i].font_scale * scale;
    row.thickness = std::max(1, static_cast<int>(kPanelRows[i].thickness * scale + 0.5));
    int baseline = 0;
    const int text_height =
        cv::getTextSize("Ag", kFont, row.font_scale, row.thickness, &baseline).height;
    row.baseline_y = y + text_height;
    row.label_width = inner_width - text_width(" 100%", row.font_scale, row.thickness);
    row.bar = cv::Rect(inner_x, even(y + text_height + kTextBarGap * scale), inner_width,
                       std::max(2, even(kPanelRows[i].bar_height * scale)));
    y = row.bar.y + row.bar.height + even(kRowGap * scale);
    layout.rows.push_back(row);
  }
  const int panel_height = y - even(kRowGap * scale) + padding;
  const int top = height - even(kPanelBottom * scale) - panel_height;
  layout.box = cv::Rect(margin, top, panel_width, panel_height);
  for (PanelRow& row : layout.rows) {
    row.baseline_y += top;
    row.bar.y += top;
  }
  return layout;
}

// The text in ASCII, which is all the panel font draws, cut with "..." when it is wider than
// max_width.
std::string fit_text(const std::string& text, double font_scale, int thickness, int max_width) {
  // A UTF-8 lead byte becomes '?' and its continuation bytes are dropped.
  std::string fitted;
  for (const unsigned char c : text) {
    if (c < 0x80 || c >= 0xC0) {
      fitted += c < 0x80 ? static_cast<char>(c) : '?';
    }
  }
  if (text_width(fitted, font_scale, thickness) <= max_width) {
    return fitted;
  }
  while (!fitted.empty() && text_width(fitted + "...", font_scale, thickness) > max_width) {
    fitted.pop_back();
  }
  return sima_examples::trim_copy(fitted) + "...";
}

// Rendered texts by text, font scale, thickness, luma and maximum width: the text drawn on black,
// and the row of its baseline.
using TextImages =
    std::map<std::tuple<std::string, double, int, int, int>, std::pair<cv::Mat, int>>;

// The text, cut to max_width when that is above 0, drawn on black, and the row of its baseline.
// Each text, style and width is fitted and rendered once into `texts`: drawing antialiased text
// into the frame took up to 1.8 ms per line on the DevKit.
const std::pair<cv::Mat, int>& text_image(TextImages& texts, const std::string& text,
                                          double font_scale, int thickness, int luma,
                                          int max_width = 0) {
  const auto key = std::make_tuple(text, font_scale, thickness, luma, max_width);
  auto found = texts.find(key);
  if (found == texts.end()) {
    const std::string fitted =
        max_width > 0 ? fit_text(text, font_scale, thickness, max_width) : text;
    int descent = 0;
    const cv::Size size = cv::getTextSize(fitted, kFont, font_scale, thickness, &descent);
    cv::Mat image(size.height + descent + 2 * thickness, size.width + 2 * thickness, CV_8UC1,
                  cv::Scalar(0));
    cv::putText(image, fitted, cv::Point(thickness, thickness + size.height), kFont, font_scale,
                luma, thickness, cv::LINE_AA);
    found = texts.emplace(key, std::make_pair(image, thickness + size.height)).first;
  }
  return found->second;
}

// Draws the top prompts into an NV12 frame in place: a darkened box with each prompt's text, bar
// and percentage, best first. `texts` keeps the rendered texts from frame to frame.
void draw_result_panel(cv::Mat& nv12, const std::vector<TopPrompt>& top, const PanelLayout& layout,
                       TextImages& texts) {
  const int height = nv12.rows * 2 / 3;
  cv::Mat luma = nv12.rowRange(0, height);
  cv::Mat chroma(height / 2, nv12.cols / 2, CV_8UC2, nv12.ptr(height));
  const cv::Rect& box = layout.box;
  cv::Mat box_luma = luma(box);
  cv::Mat box_chroma = chroma(cv::Rect(box.x / 2, box.y / 2, box.width / 2, box.height / 2));
  cv::convertScaleAbs(box_luma, box_luma, kPanelDim, 0);
  cv::convertScaleAbs(box_chroma, box_chroma, kPanelDim, kNeutralChroma * (1 - kPanelDim));
  for (std::size_t i = 0; i < layout.rows.size() && i < top.size(); ++i) {
    const PanelRow& row = layout.rows[i];
    const cv::Rect& bar = row.bar;
    const int fill = even(bar.width * top[i].confidence);
    cv::rectangle(luma, bar, kTrackLuma, cv::FILLED);
    if (fill > 0) {
      cv::rectangle(luma, cv::Rect(bar.x, bar.y, fill, bar.height),
                    i == 0 ? kAccentYuv[0] : kBarLuma, cv::FILLED);
      if (i == 0) {
        cv::rectangle(chroma, cv::Rect(bar.x / 2, bar.y / 2, fill / 2, bar.height / 2),
                      cv::Scalar(kAccentYuv[1], kAccentYuv[2]), cv::FILLED);
      }
    }
    // The label starts at the bar's left end, the percentage ends at its right end.
    const std::string percent =
        std::to_string(static_cast<int>(top[i].confidence * 100 + 0.5)) + "%";
    for (const auto& [text, max_width, align_right] :
         {std::tuple{top[i].label, row.label_width, false}, std::tuple{percent, 0, true}}) {
      const auto& [image, baseline] = text_image(texts, text, row.font_scale, row.thickness,
                                                 i == 0 ? kBestTextLuma : kTextLuma, max_width);
      const int left =
          align_right ? bar.x + bar.width - image.cols + row.thickness : bar.x - row.thickness;
      cv::Mat area = luma(cv::Rect(left, row.baseline_y - baseline, image.cols, image.rows));
      // The text is brighter than the darkened box, so the brighter pixel of the two keeps its
      // antialiased edge.
      cv::max(area, image, area);
    }
  }
}

void run(const Config& cfg) {
  // Check the prompts, the table and the stream first: they fail in a second, the models take
  // longer to load.
  const std::vector<std::string> prompts = load_prompts(cfg.prompts_path);
  const std::vector<std::vector<int64_t>> tokens = prompt_tokens(cfg.tokenizer, prompts);
  const TokenEmbeddings table = load_token_embeddings(cfg.token_embeddings);
  const sima_examples::RtspStreamInfo stream = probe_stream(cfg);

  const double started_ms = sima_examples::time_ms();
  const std::vector<std::vector<float>> prompt_vectors = encode_prompts(cfg, tokens, table);
  const double encoded_ms = sima_examples::time_ms();
  const neat::Model image_encoder(cfg.image_encoder, image_encoder_options(stream));
  Graphs graphs = build_graphs(cfg, image_encoder, stream);
  // The sender opens before the Runs build, so a bad host stops the application before the stream
  // starts.
  std::string metadata_error;
  const neat::MetadataSender metadata(metadata_options(cfg), &metadata_error);
  sima_examples::require(metadata.ok(), "could not open the Insight metadata sender: " +
                                            metadata_error + "; check output.insight.host");
  neat::Run graph_run = graphs.classifier.build(run_options());
  neat::Run video_run = graphs.video.build(neat::RunOptions{});
  std::cout << "rtsp=" << cfg.rtsp_url << " stream=" << stream.width << "x" << stream.height << "@"
            << stream.fps << " codec=" << cfg.codec << " prompts=" << prompts.size()
            << cv::format(" text_encoding_s=%.1f graph_build_s=%.1f",
                          (encoded_ms - started_ms) / 1000.0,
                          (sima_examples::time_ms() - encoded_ms) / 1000.0)
            << " insight=" << cfg.insight_host << " video=" << cfg.video_port
            << " metadata=" << cfg.metadata_port << " channel=0" << std::endl;

  const PanelLayout layout = panel_layout(stream.width, stream.height,
                                          std::min(kTopPrompts, static_cast<int>(prompts.size())));
  TextImages texts;
  int processed = 0;
  int64_t skipped_frames = 0;
  int64_t window_start_skipped = 0;
  double window_start_ms = sima_examples::time_ms();
  double ranking_ms = 0.0;
  double panel_ms = 0.0;
  int64_t last_frame_id = -1;
  g_stop_requested = 0;
  const auto previous_sigint = std::signal(SIGINT, request_stop);
  const auto finish = [&] {
    std::signal(SIGINT, previous_sigint);
    graph_run.close();
    video_run.close();
    std::cout << "processed=" << processed << " skipped_frames=" << skipped_frames
              << " video_sender=" << cfg.insight_host << ":" << cfg.video_port << std::endl;
  };
  try {
    while (g_stop_requested == 0 && (cfg.frames <= 0 || processed < cfg.frames)) {
      const std::optional<neat::Sample> sample = graph_run.pull("result", kPullTimeoutMs);
      if (!sample) {
        if (g_stop_requested != 0) {
          break;
        }
        const std::string error = graph_run.last_error();
        throw std::runtime_error("no frame classified in " + std::to_string(kPullTimeoutMs) +
                                 " ms; check that " + cfg.rtsp_url + " is streaming" +
                                 (error.empty() ? "" : ": " + error));
      }
      const double ranking_start_ms = sima_examples::time_ms();
      const neat::Mapping embedding = image_embedding(field(*sample, "embeddings")).view_read();
      const std::vector<TopPrompt> top =
          top_prompts(static_cast<const float*>(embedding.data), prompt_vectors, prompts);
      ranking_ms += sima_examples::time_ms() - ranking_start_ms;
      const double panel_start_ms = sima_examples::time_ms();
      // A new frame each time: the video Run reads the pushed frame later, on its own thread.
      cv::Mat nv12(stream.height * 3 / 2, stream.width, CV_8UC1);
      // The copy leaves the decoder's buffer as it is: the decoder may still read it as a
      // reference frame.
      const neat::TensorList frame = neat::tensors_from_sample(field(*sample, "frame"), false);
      sima_examples::require(!frame.empty() &&
                                 frame.front().copy_nv12_contiguous_to(nv12.data, nv12.total()),
                             "the decoded frame is not " + std::to_string(stream.width) + "x" +
                                 std::to_string(stream.height) + " NV12");
      draw_result_panel(nv12, top, layout, texts);
      if (!video_run.push(nv12_sample(nv12, *sample))) {
        throw std::runtime_error("the Insight video sender stopped accepting frames");
      }
      panel_ms += sima_examples::time_ms() - panel_start_ms;
      const int64_t timestamp_ms = sample->pts_ns >= 0 ? sample->pts_ns / 1'000'000 : -1;
      std::string error;
      if (!metadata.send_metadata("classification", top_classes_json(top), timestamp_ms,
                                  std::to_string(sample->frame_id), &error) &&
          !error.empty()) {
        throw std::runtime_error(error);
      }

      ++processed;
      // A frame the decoder produced but the image encoder did not classify leaves a gap in the
      // frame ids.
      if (last_frame_id >= 0) {
        skipped_frames += std::max<int64_t>(0, sample->frame_id - last_frame_id - 1);
      }
      last_frame_id = sample->frame_id;
      if (cfg.profile && processed % cfg.profile_interval == 0) {
        const double elapsed_s = (sima_examples::time_ms() - window_start_ms) / 1000.0;
        std::cout << cv::format("[profile] frames=%d classified_fps=%.1f ranking_ms=%.3f "
                                "panel_ms=%.3f skipped_frames=%lld top='%s' confidence=%.2f",
                                cfg.profile_interval, cfg.profile_interval / elapsed_s,
                                ranking_ms / cfg.profile_interval, panel_ms / cfg.profile_interval,
                                static_cast<long long>(skipped_frames - window_start_skipped),
                                top[0].label.c_str(), top[0].confidence)
                  << std::endl;
        window_start_ms = sima_examples::time_ms();
        ranking_ms = 0.0;
        panel_ms = 0.0;
        window_start_skipped = skipped_frames;
      }
    }
  } catch (...) {
    finish();
    throw;
  }
  finish();
}

} // namespace

int main(int argc, char** argv) {
  try {
    fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config") {
        sima_examples::require(i + 1 < argc, "--config requires a path");
        config_path = argv[++i];
      } else if (arg == "--help" || arg == "-h") {
        std::cout << "Usage: " << argv[0] << " [--config <path>]\n"
                  << "CLIP zero-shot classification of one RTSP stream against text prompts, "
                     "video and top prompts to Insight\n";
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }
    run(load_config(config_path));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERR] " << e.what() << "\n";
    return 1;
  }
}
