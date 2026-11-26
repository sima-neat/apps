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

// Flags surface defects with PatchCore and outlines them, in an RTSP stream or a folder of images.
// The model turns each frame into patch features, and the host compares every patch with a memory
// bank of known-good patches that --calibrate builds. Insight receives the stream as received and
// each anomalous frame's defect regions as polygons; image_dir writes the outlined images instead.

#include "neat.h"
#include "sima_examples/anomaly_regions.h"
#include "sima_examples/config.h"
#include "sima_examples/example_utils.h"
#include "sima_examples/pull_status.h"
#include "sima_examples/rtsp_probe.h"
#include "sima_examples/stop_signals.h"
#include "utils/memory_bank.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

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
const std::map<std::string, std::string> kInputTypes = {{"rtsp", "rtsp"},
                                                        {"image_dir", "image_dir"}};
constexpr const char* kBackbone = "wide_resnet50_2";
constexpr const char* kTorchvisionWeights = "IMAGENET1K_V1";
constexpr int kEmbedDim = 1536;
constexpr int kModelInputSide =
    224; // The model's input, and the side of the map regions are found on.
constexpr int kPatchGrid = 28;
constexpr int kNumNeighbors =
    9; // Neighborhood size of the image score's reweighting (paper Eq. 7-8).
constexpr std::uint64_t kSeed = 0;
constexpr double kMapSigma = 4.0; // Smoothing of the upsampled score map, in map pixels.
constexpr int kMinRegionPx = 1;   // The smoothing already removes single-patch speckle.
constexpr std::size_t kThresholdEvery =
    5; // --calibrate keeps every 5th nominal image out of the bank.
constexpr int kPullTimeoutMs = 100;
const cv::Scalar kOutlineBgr(0, 0, 255);

struct CliOptions {
  fs::path config_path;
  bool calibrate = false;
};

CliOptions parse_args(int argc, char** argv) {
  const fs::path adjacent_config = fs::path(argv[0]).parent_path() / "config.yaml";
  CliOptions options;
  options.config_path = fs::exists(adjacent_config)
                            ? adjacent_config
                            : sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      options.config_path = argv[++i];
    } else if (arg == "--calibrate") {
      options.calibrate = true;
    } else {
      throw std::runtime_error("usage: " + std::string(argv[0]) +
                               " [--config <path>] [--calibrate]");
    }
  }
  return options;
}

// Host-decoded RGB images, already at the model's input size, normalized as ImageNet.
neat::Model::Options image_model_options() {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.preset = neat::NormalizePreset::ImageNet;
  return options;
}

// EV74 converts each NV12 frame, stretches it to the model input as image_input does on the host,
// and normalizes it as ImageNet, so the bank's images and the stream match.
neat::Model::Options stream_model_options(const sima_examples::RtspStreamInfo& stream) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::NV12;
  options.preprocess.input_max_width = stream.width;
  options.preprocess.input_max_height = stream.height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.preset = neat::NormalizePreset::ImageNet;
  return options;
}

neat::Tensor image_input(const cv::Mat& bgr) {
  cv::Mat resized, rgb;
  cv::resize(bgr, resized, cv::Size(kModelInputSide, kModelInputSide), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  return neat::Tensor::from_cv_mat(rgb, neat::ImageSpec::PixelFormat::RGB,
                                   neat::TensorMemory::EV74);
}

std::vector<fs::path> image_paths(const fs::path& folder) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(folder)) {
    std::string suffix = entry.path().extension().string();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (suffix == ".bmp" || suffix == ".jpeg" || suffix == ".jpg" || suffix == ".png") {
      paths.push_back(entry.path());
    }
  }
  sima_examples::require(!paths.empty(), "no images in " + folder.string());
  std::sort(paths.begin(), paths.end());
  return paths;
}

cv::Mat read_image(const fs::path& path) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  sima_examples::require(!bgr.empty(), "cannot read image " + path.string());
  return bgr;
}

patchcore::PatchEmbeddings tensor_features(const neat::Tensor& tensor) {
  const std::size_t bytes = tensor.dense_bytes_tight();
  std::vector<float> values(bytes / sizeof(float));
  sima_examples::require(
      tensor.dtype == neat::TensorDType::Float32 &&
          tensor.copy_dense_bytes_tight_to(reinterpret_cast<std::uint8_t*>(values.data()), bytes),
      "the model must return Float32 patch features");
  return patchcore::extract_hwc(tensor.shape, std::move(values), kEmbedDim);
}

patchcore::PatchEmbeddings patch_features(neat::Model& model, const cv::Mat& bgr) {
  return tensor_features(model.run(neat::TensorList{image_input(bgr)}).front());
}

// Build the memory bank from the nominal images and measure both thresholds on the images it
// keeps out, then write memory_bank.npy and bank_meta.json.
void calibrate(const sima_examples::Config& config, const fs::path& model_path) {
  const auto nominal = image_paths(config.string("app.nominal_images_dir"));
  sima_examples::require(nominal.size() >= kThresholdEvery,
                         "app.nominal_images_dir needs at least " +
                             std::to_string(kThresholdEvery) + " images");
  std::vector<fs::path> bank_images, held_out;
  for (std::size_t i = 0; i < nominal.size(); ++i) {
    (i % kThresholdEvery == kThresholdEvery - 1 ? held_out : bank_images).push_back(nominal[i]);
  }
  neat::Model model(model_path.string(), image_model_options());
  const double coreset_ratio = config.number("app.coreset_ratio");
  std::vector<patchcore::PatchEmbeddings> features;
  for (const auto& path : bank_images) {
    features.push_back(patch_features(model, read_image(path)));
  }
  const auto bank = patchcore::MemoryBank::build(features, coreset_ratio, kSeed);

  std::vector<float> image_scores, patch_scores;
  for (const auto& path : held_out) {
    const auto scored = bank.score(patch_features(model, read_image(path)), kNumNeighbors);
    image_scores.push_back(scored.image_score);
    patch_scores.insert(patch_scores.end(), scored.score_map.begin(), scored.score_map.end());
  }
  const double percentile = config.number("app.threshold_percentile");
  const auto margin = static_cast<float>(config.number("app.threshold_margin"));
  const float threshold = patchcore::percentile_threshold(image_scores, percentile) * margin;
  const float patch_threshold = patchcore::percentile_threshold(patch_scores, percentile) * margin;

  const fs::path bank_dir = config.string("app.memory_bank_dir");
  fs::create_directories(bank_dir);
  bank.save(bank_dir / "memory_bank.npy");
  patchcore::BankMeta meta;
  meta.model_sha256 = patchcore::sha256_file(model_path);
  meta.bank_sha256 = patchcore::sha256_file(bank_dir / "memory_bank.npy");
  meta.model_filename = model_path.filename().string();
  meta.backbone = kBackbone;
  meta.torchvision_weights = kTorchvisionWeights;
  meta.embed_dim = kEmbedDim;
  meta.patch_grid_h = kPatchGrid;
  meta.patch_grid_w = kPatchGrid;
  meta.coreset_ratio = coreset_ratio;
  meta.seed = kSeed;
  meta.num_nominal_images = static_cast<int>(bank_images.size());
  meta.bank_size = static_cast<int>(bank.size());
  meta.num_neighbors = kNumNeighbors;
  meta.gaussian_sigma = kMapSigma;
  meta.threshold_value = threshold;
  meta.threshold_percentile = percentile;
  meta.threshold_num_images = static_cast<int>(held_out.size());
  meta.has_patch_threshold = true;
  meta.patch_threshold_value = patch_threshold;
  meta.created_at = patchcore::current_utc_timestamp();
  patchcore::save_bank_meta(bank_dir / "bank_meta.json", meta);
  std::cout << "bank=" << bank.size() << " patches from " << bank_images.size()
            << " images threshold=" << threshold << " patch_threshold=" << patch_threshold << " ("
            << held_out.size() << " held-out images) saved=" << bank_dir.string() << std::endl;
}

struct CalibratedBank {
  patchcore::MemoryBank bank;
  patchcore::BankMeta meta;
};

// The calibrated bank and its metadata, checked against the configured model.
CalibratedBank load_bank(const sima_examples::Config& config, const fs::path& model_path) {
  const fs::path bank_dir = config.string("app.memory_bank_dir");
  const fs::path bank_path = bank_dir / "memory_bank.npy";
  const fs::path meta_path = bank_dir / "bank_meta.json";
  sima_examples::require(fs::exists(bank_path) && fs::exists(meta_path),
                         "no memory bank in " + bank_dir.string() + "; run with --calibrate first");
  auto meta = patchcore::load_bank_meta(meta_path);
  patchcore::verify_bank_matches_model(meta, model_path);
  patchcore::verify_bank_hash(meta, bank_path);
  sima_examples::require(meta.has_patch_threshold,
                         meta_path.string() +
                             " has no patch_threshold; run with --calibrate again");
  return {patchcore::MemoryBank::load(bank_path), std::move(meta)};
}

sima_examples::RegionOptions region_options(const sima_examples::Config& config,
                                            const patchcore::BankMeta& meta, int width,
                                            int height) {
  sima_examples::RegionOptions options;
  options.width = width;
  options.height = height;
  options.threshold = meta.patch_threshold_value;
  options.min_region_px = kMinRegionPx;
  options.max_regions = config.integer("inference.max_detections");
  return options;
}

struct FrameResult {
  float score = 0.0F;                         // The frame's anomaly score.
  std::vector<sima_examples::Region> regions; // Empty unless the score is above threshold.
};

FrameResult defect_regions(const patchcore::MemoryBank& bank,
                           const patchcore::PatchEmbeddings& features, double threshold,
                           const sima_examples::RegionOptions& options) {
  const auto scored = bank.score(features, kNumNeighbors);
  if (scored.image_score <= threshold) {
    return {scored.image_score, {}};
  }
  const cv::Mat map =
      patchcore::anomaly_map(scored, features.height, features.width, kModelInputSide, kMapSigma);
  return {scored.image_score, sima_examples::defect_regions(map, options)};
}

// Score every image of input.source and write it to output.dir with its regions outlined.
void score_images(const sima_examples::Config& config, const fs::path& model_path,
                  const CalibratedBank& calibrated) {
  const fs::path output_dir = config.string("output.dir");
  fs::create_directories(output_dir);
  const double threshold = calibrated.meta.threshold_value;
  neat::Model model(model_path.string(), image_model_options());
  int flagged = 0;
  const auto paths = image_paths(config.string("input.source"));
  for (const auto& path : paths) {
    cv::Mat bgr = read_image(path);
    const auto result = defect_regions(calibrated.bank, patch_features(model, bgr), threshold,
                                       region_options(config, calibrated.meta, bgr.cols, bgr.rows));
    for (const auto& region : result.regions) {
      std::vector<cv::Point> outline;
      for (const auto& point : region.polygon) {
        outline.emplace_back(point[0], point[1]);
      }
      cv::polylines(bgr, outline, true, kOutlineBgr, 2);
    }
    sima_examples::require(cv::imwrite((output_dir / path.filename()).string(), bgr),
                           "cannot write " + (output_dir / path.filename()).string());
    const bool anomalous = result.score > threshold;
    std::printf("%s: %s score=%.2f threshold=%.2f regions=%zu\n", path.filename().c_str(),
                anomalous ? "ANOMALOUS" : "normal", result.score, threshold, result.regions.size());
    flagged += anomalous ? 1 : 0;
  }
  std::cout << "processed=" << paths.size() << " flagged=" << flagged
            << " output=" << output_dir.string() << std::endl;
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

// Decoded frames -> PatchCore feature extractor -> "features" output.
neat::Graph features_graph(neat::Model& model) {
  neat::Graph graph("extractor");
  graph.add(model.graph());
  graph.add(neat::nodes::Output("features"));
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

struct RegionCounts {
  int processed = 0; // Frames processed.
  int flagged = 0;   // Anomalous frames.
};

// Send every frame's defect regions to Insight until `frames` (0: no limit) or a stop signal.
RegionCounts send_regions(neat::Run& run, neat::MetadataSender& metadata_sender,
                          const patchcore::MemoryBank& bank, double threshold,
                          const sima_examples::RegionOptions& options, int frames,
                          const std::atomic<bool>& stop) {
  RegionCounts counts;
  while (!stop && (frames == 0 || counts.processed < frames)) {
    neat::Sample sample;
    neat::PullError error;
    const auto status = run.pull("features", kPullTimeoutMs, sample, &error);
    if (!sima_examples::pull_status_has_sample(status, "features", error, run.last_error())) {
      continue;
    }
    const auto result = defect_regions(
        bank, tensor_features(neat::tensors_from_sample(sample).front()), threshold, options);
    // A normal frame sends no segments, which clears the last outline. A nonblocking send drops
    // the message while the socket buffer is full; the drops are counted in the sender's stats
    // and reported when the run ends. Any other failure throws.
    std::string send_error;
    if (!metadata_sender.send_metadata("segmentation", sima_examples::segments_json(result.regions),
                                       sample.pts_ns >= 0 ? sample.pts_ns / 1'000'000 : -1,
                                       sample.frame_id >= 0 ? std::to_string(sample.frame_id) : "",
                                       &send_error) &&
        !send_error.empty()) {
      throw std::runtime_error("metadata send failed: " + send_error);
    }
    ++counts.processed;
    counts.flagged += result.regions.empty() ? 0 : 1;
  }
  return counts;
}

int run_app(const CliOptions& cli) {
  const auto config = sima_examples::Config::load(cli.config_path);
  const fs::path model_path =
      fs::absolute(cli.config_path).parent_path() / config.string("model.path");
  if (cli.calibrate) {
    calibrate(config, model_path);
    return 0;
  }
  const CalibratedBank calibrated = load_bank(config, model_path);
  if (config.choice("input.type", kInputTypes) == "image_dir") {
    score_images(config, model_path, calibrated);
    return 0;
  }

  const auto stream = sima_examples::probe_rtsp_stream_info(
      config.string("input.source"), config.boolean("input.rtsp.tcp"),
      {config.integer("input.width"), config.integer("input.height"), config.integer("input.fps")});
  neat::Model model(model_path.string(), stream_model_options(stream));

  const neat::Graph rtsp = groups::RtspEncodedInput(rtsp_options(config, stream));
  const neat::Graph decoder = decoder_graph(config, stream);
  const neat::Graph video = groups::VideoSender(video_options(config));
  neat::Graph graph;
  graph.connect(rtsp, decoder);
  // A plain link lets Core tee the encoded stream to the decoder and the video sender.
  graph.connect(rtsp, video);
  graph.connect(decoder, features_graph(model));

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
            << stream.height << "@" << stream.fps << " bank=" << calibrated.bank.size()
            << " insight=" << host << " video=" << video_port
            << " metadata=" << config.integer("output.insight.metadata_port_base") << std::endl;
  const RegionCounts counts =
      send_regions(run, metadata_sender, calibrated.bank, calibrated.meta.threshold_value,
                   region_options(config, calibrated.meta, stream.width, stream.height),
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
