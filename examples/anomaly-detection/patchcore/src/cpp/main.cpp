/**
 * @example patchcore.cpp
 * PatchCore anomaly detection: a compiled wide_resnet50_2 patch-feature
 * extractor on the MLA, plus host-side coreset memory-bank scoring.
 *
 * Usage:
 *   patchcore --calibrate [--config <path>]
 *   patchcore [--config <path>]
 */
#include "neat.h"
#include "neat/node_groups.h"
#include "neat/nodes.h"
#include "support/anomaly_detection/patchcore_memory_bank.h"
#include "support/anomaly_detection/sha256.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using sima_examples::time_ms;

namespace {

// WideResNet-50 layer2 (512ch) + upsampled layer3 (1024ch) concatenated patch embedding,
// on a 28x28 grid for a 224x224 input -- the one qualified configuration this example ships.
constexpr int kEmbedDim = 1536;
constexpr int kPatchGridH = 28;
constexpr int kPatchGridW = 28;
constexpr const char* kBackbone = "wide_resnet50_2";
constexpr const char* kTorchvisionWeights = "IMAGENET1K_V1";

enum class SourceType { ImageDir, VideoFile, Rtsp };
enum class RtspCodec { H264, H265, Mjpeg };

struct Config {
  std::string model_path;

  SourceType source_type = SourceType::ImageDir;
  fs::path image_dir;
  fs::path video_path;
  std::string rtsp_url;
  RtspCodec rtsp_codec = RtspCodec::H264;
  bool rtsp_tcp = true;
  int rtsp_latency_ms = 200;
  int rtsp_width = 0;
  int rtsp_height = 0;

  fs::path memory_bank_path;
  fs::path bank_meta_path;

  fs::path calibration_nominal_dir;
  double coreset_ratio = 0.01;
  std::uint64_t seed = 0;
  double threshold_percentile = 99.0;
  fs::path calibration_threshold_dir; // empty = reuse calibration_nominal_dir

  int num_neighbors = 9;
  double gaussian_sigma = 4.0;

  fs::path output_dir;
  int save_every = 0; // image_dir: always saves regardless. video_file/rtsp: 0 disables local saving.
  double overlay_alpha = 0.45;

  std::string insight_host; // video_file/rtsp only
  int insight_video_port = 9000;
  int insight_channel = 0;
  int insight_bitrate_kbps = 1000;

  int timeout_ms = 5000;
  int frames = 0;
};

SourceType parse_source_type(const std::string& v) {
  if (v == "image_dir") return SourceType::ImageDir;
  if (v == "video_file") return SourceType::VideoFile;
  if (v == "rtsp") return SourceType::Rtsp;
  throw std::runtime_error("source.type must be image_dir, video_file, or rtsp");
}

RtspCodec parse_rtsp_codec(const std::string& v) {
  if (v == "h264") return RtspCodec::H264;
  if (v == "h265") return RtspCodec::H265;
  if (v == "mjpeg") return RtspCodec::Mjpeg;
  throw std::runtime_error("source.rtsp.codec must be h264, h265, or mjpeg");
}

void validate_config(const Config& cfg) {
  sima_examples::require(!cfg.model_path.empty(), "model.path must be set");
  if (cfg.source_type == SourceType::VideoFile) {
    sima_examples::require(!cfg.video_path.empty(),
                           "source.video_path must be set when source.type=video_file");
  }
  if (cfg.source_type == SourceType::Rtsp) {
    sima_examples::require(!cfg.rtsp_url.empty(),
                           "source.rtsp.url must be set when source.type=rtsp");
  }
  sima_examples::require(cfg.coreset_ratio > 0.0 && cfg.coreset_ratio <= 1.0,
                         "calibration.coreset_ratio must be in (0, 1]");
  sima_examples::require(cfg.threshold_percentile >= 0.0 && cfg.threshold_percentile <= 100.0,
                         "calibration.threshold_percentile must be between 0 and 100");
  sima_examples::require(cfg.num_neighbors >= 1, "scoring.num_neighbors must be >= 1");
  sima_examples::require(cfg.timeout_ms > 0, "runtime.timeout_ms must be > 0");
  sima_examples::require(cfg.frames >= 0, "runtime.frames must be >= 0");
  if (cfg.source_type != SourceType::ImageDir) {
    sima_examples::require(!cfg.insight_host.empty(),
                           "output.insight.host must be set for source.type=video_file/rtsp");
    sima_examples::require(cfg.insight_video_port > 0 && cfg.insight_video_port <= 65535,
                           "output.insight.video_port must be in [1, 65535]");
    sima_examples::require(cfg.insight_channel >= 0, "output.insight.channel must be >= 0");
    sima_examples::require(cfg.insight_bitrate_kbps > 0, "output.insight.bitrate_kbps must be > 0");
  }
}

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "");

  cfg.source_type = parse_source_type(raw.string_or("source.type", "image_dir"));
  cfg.image_dir = raw.string_or("source.image_dir", "assets/datasets/patchcore/images");
  cfg.video_path = raw.string_or("source.video_path", "");
  cfg.rtsp_url = raw.string_or("source.rtsp.url", "");
  cfg.rtsp_codec = parse_rtsp_codec(raw.string_or("source.rtsp.codec", "h264"));
  cfg.rtsp_tcp = raw.bool_or("source.rtsp.tcp", true);
  cfg.rtsp_latency_ms = raw.int_or("source.rtsp.latency_ms", 200);
  cfg.rtsp_width = raw.int_or("source.rtsp.width", 0);
  cfg.rtsp_height = raw.int_or("source.rtsp.height", 0);

  cfg.memory_bank_path = raw.string_or("memory_bank.path", "sandbox/patchcore/memory_bank.npy");
  cfg.bank_meta_path = raw.string_or("memory_bank.meta_path", "sandbox/patchcore/bank_meta.json");

  cfg.calibration_nominal_dir =
      raw.string_or("calibration.nominal_images_dir", "assets/datasets/patchcore/nominal");
  cfg.coreset_ratio = raw.double_or("calibration.coreset_ratio", 0.01);
  cfg.seed = static_cast<std::uint64_t>(raw.int_or("calibration.seed", 0));
  cfg.threshold_percentile = raw.double_or("calibration.threshold_percentile", 99.0);
  cfg.calibration_threshold_dir = raw.string_or("calibration.threshold_images_dir", "");

  cfg.num_neighbors = raw.int_or("scoring.num_neighbors", 9);
  cfg.gaussian_sigma = raw.double_or("scoring.gaussian_sigma", 4.0);

  cfg.output_dir = raw.string_or("output.dir", "sandbox/patchcore");
  cfg.save_every = raw.int_or("output.save_every", 0);
  cfg.overlay_alpha = raw.double_or("output.overlay_alpha", 0.45);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.insight_video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.insight_channel = raw.int_or("output.insight.channel", 0);
  cfg.insight_bitrate_kbps = raw.int_or("output.insight.bitrate_kbps", 1000);

  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 5000);
  cfg.frames = raw.int_or("runtime.frames", 0);

  validate_config(cfg);
  return cfg;
}

struct CliOptions {
  fs::path config_path;
  bool calibrate = false;
  bool validate_config_only = false;
};

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  options.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      options.config_path = argv[++i];
    } else if (arg == "--calibrate") {
      options.calibrate = true;
    } else if (arg == "--validate-config-only") {
      options.validate_config_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--config <path>] [--calibrate] [--validate-config-only]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

bool is_image(const fs::path& p) {
  auto ext = p.extension().string();
  for (auto& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> find_images(const fs::path& dir) {
  std::vector<fs::path> images;
  if (!fs::is_directory(dir)) {
    return images;
  }
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

/// Heatmap-only overlay -- score/threshold/verdict are printed to stdout, not
/// burned into the frame.
cv::Mat draw_overlay(const cv::Mat& bgr, const patchcore::AnomalyResult& result, int map_h,
                     int map_w, double sigma, double alpha) {
  cv::Mat score_mat(map_h, map_w, CV_32FC1, const_cast<float*>(result.score_map.data()));
  cv::Mat heat;
  cv::resize(score_mat, heat, bgr.size(), 0, 0, cv::INTER_LINEAR);
  if (sigma > 0.0) {
    cv::GaussianBlur(heat, heat, cv::Size(0, 0), sigma);
  }

  double lo = 0.0;
  double hi = 0.0;
  cv::minMaxLoc(heat, &lo, &hi);
  cv::Mat heat_u8;
  if (hi > lo) {
    heat.convertTo(heat_u8, CV_8U, 255.0 / (hi - lo), -255.0 * lo / (hi - lo));
  } else {
    heat_u8 = cv::Mat::zeros(bgr.size(), CV_8UC1);
  }
  cv::Mat heat_color;
  cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

  cv::Mat out;
  cv::addWeighted(bgr, 1.0 - alpha, heat_color, alpha, 0.0, out);
  return out;
}

// ---------------------------------------------------------------------------
// Model: one shared Options for every source type. Each source decodes to a
// host-side BGR cv::Mat before calling the model, so no Graph-embedded decode
// source or hand-specified resize geometry is needed here.
// ---------------------------------------------------------------------------

simaai::neat::Model::Options image_model_options() {
  simaai::neat::Model::Options opt;
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.preset = simaai::neat::NormalizePreset::ImageNet;
  return opt;
}

simaai::neat::Tensor rgb_tensor(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return simaai::neat::Tensor::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
                                           simaai::neat::TensorMemory::EV74);
}

patchcore::PatchEmbeddings extract_from_bgr(simaai::neat::Model& model, const cv::Mat& bgr,
                                            int timeout_ms) {
  auto tensors = model.run(simaai::neat::TensorList{rgb_tensor(bgr)}, timeout_ms);
  if (tensors.empty()) {
    throw std::runtime_error("model returned no output tensors");
  }
  const auto& t = tensors.front();
  const auto flat = sima_examples::tensor_to_floats(t);
  return patchcore::extract_hwc(t.shape, flat, kEmbedDim);
}

int cmd_calibrate(const Config& cfg) {
  const auto paths = find_images(cfg.calibration_nominal_dir);
  if (paths.empty()) {
    std::cerr << "[FATAL] no images found in " << cfg.calibration_nominal_dir << "\n";
    return 3;
  }

  simaai::neat::Model model(cfg.model_path, image_model_options());
  std::cout << "Extracting patch embeddings from " << paths.size() << " nominal images ...\n";

  std::vector<patchcore::PatchEmbeddings> per_image;
  per_image.reserve(paths.size());
  std::size_t total_patches = 0;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    const auto& path = paths[i];
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "[WARN] could not read image: " << path << "\n";
      continue;
    }
    auto embedding = extract_from_bgr(model, bgr, cfg.timeout_ms);
    total_patches += embedding.patch_count();
    per_image.push_back(std::move(embedding));
    if ((i + 1) % 10 == 0 || i + 1 == paths.size()) {
      std::cout << "  [" << (i + 1) << "/" << paths.size() << "] " << path.filename().string()
                << "\n";
    }
  }

  const auto bank = patchcore::MemoryBank::build(per_image, cfg.coreset_ratio, cfg.seed);
  std::cout << "Coreset: " << bank.size() << " / " << total_patches
            << " patches (ratio=" << cfg.coreset_ratio << ")\n";

  const fs::path threshold_dir =
      cfg.calibration_threshold_dir.empty() ? cfg.calibration_nominal_dir
                                            : cfg.calibration_threshold_dir;
  const auto threshold_paths = find_images(threshold_dir);
  std::vector<float> scores;
  scores.reserve(threshold_paths.size());
  for (const auto& path : threshold_paths) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      continue;
    }
    const auto embedding = extract_from_bgr(model, bgr, cfg.timeout_ms);
    const auto scored = bank.score(embedding, cfg.num_neighbors);
    scores.push_back(scored.image_score);
  }
  const int threshold_num_images = static_cast<int>(scores.size());
  const float threshold = patchcore::percentile_threshold(scores, cfg.threshold_percentile);
  std::cout << "Threshold: " << threshold << " (p" << cfg.threshold_percentile << " over "
            << threshold_num_images << " nominal images from " << threshold_dir << ")\n";

  if (!cfg.memory_bank_path.parent_path().empty()) {
    fs::create_directories(cfg.memory_bank_path.parent_path());
  }
  if (!cfg.bank_meta_path.parent_path().empty()) {
    fs::create_directories(cfg.bank_meta_path.parent_path());
  }
  bank.save(cfg.memory_bank_path);

  patchcore::BankMeta meta;
  meta.model_sha256 = patchcore::sha256_file(cfg.model_path);
  meta.bank_sha256 = patchcore::sha256_file(cfg.memory_bank_path);
  meta.model_filename = fs::path(cfg.model_path).filename().string();
  meta.backbone = kBackbone;
  meta.torchvision_weights = kTorchvisionWeights;
  meta.embed_dim = kEmbedDim;
  meta.patch_grid_h = kPatchGridH;
  meta.patch_grid_w = kPatchGridW;
  meta.coreset_ratio = cfg.coreset_ratio;
  meta.seed = cfg.seed;
  meta.num_nominal_images = static_cast<int>(per_image.size());
  meta.bank_size = static_cast<int>(bank.size());
  meta.num_neighbors = cfg.num_neighbors;
  meta.gaussian_sigma = cfg.gaussian_sigma;
  meta.threshold_value = threshold;
  meta.threshold_percentile = cfg.threshold_percentile;
  meta.threshold_num_images = threshold_num_images;
  meta.created_at = patchcore::current_utc_timestamp();
  patchcore::save_bank_meta(cfg.bank_meta_path, meta);

  std::cout << "Saved " << cfg.memory_bank_path << " (" << bank.size() << " x " << bank.embed_dim()
            << ") and " << cfg.bank_meta_path << "\n";
  return 0;
}

// ---------------------------------------------------------------------------
// Score: image_dir -- writes annotated overlays to output.dir, no live view
// (matches every folder-based example in this repo: depth-estimator,
// classification/image-classifier, etc.).
// ---------------------------------------------------------------------------

int cmd_score_image_dir(const Config& cfg, const patchcore::MemoryBank& bank, float threshold,
                        int num_neighbors) {
  const auto paths = find_images(cfg.image_dir);
  if (paths.empty()) {
    std::cerr << "[FATAL] no images found in " << cfg.image_dir << "\n";
    return 3;
  }
  fs::create_directories(cfg.output_dir);

  simaai::neat::Model model(cfg.model_path, image_model_options());

  int processed = 0;
  for (const auto& path : paths) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "[WARN] could not read image: " << path << "\n";
      continue;
    }

    const double mla_start = time_ms();
    const auto embedding = extract_from_bgr(model, bgr, cfg.timeout_ms);
    const double mla_ms = time_ms() - mla_start;

    const double host_start = time_ms();
    const auto scored = bank.score(embedding, num_neighbors);
    const cv::Mat overlay = draw_overlay(bgr, scored, embedding.height, embedding.width,
                                         cfg.gaussian_sigma, cfg.overlay_alpha);
    const double host_ms = time_ms() - host_start;

    const bool anomalous = scored.image_score >= threshold;
    std::cout << path.string() << ": score=" << scored.image_score << " threshold=" << threshold
              << " verdict=" << (anomalous ? "ANOMALOUS" : "normal") << " (mla=" << mla_ms
              << "ms host=" << host_ms << "ms)\n";
    cv::imwrite((cfg.output_dir / path.filename()).string(), overlay);
    ++processed;
  }
  std::cout << "Done: " << processed << " images processed -- overlays written to " << cfg.output_dir
            << "\n";
  return processed > 0 ? 0 : 3;
}

// ---------------------------------------------------------------------------
// Score: video_file -- cv::VideoCapture, streaming the annotated overlay live
// to Insight via a small host-pushed graph; `output.save_every > 0`
// additionally writes periodic local snapshots. Also used by rtsp.
// ---------------------------------------------------------------------------

struct VideoSender {
  simaai::neat::Graph graph{"insight"};
  simaai::neat::Run run;
  int port = 0;
};

/// A small, separate Insight push-graph: the host manually pushes each
/// annotated frame into it.
VideoSender build_video_sender(const Config& cfg, double fps, int width, int height) {
  const int output_fps = std::max(1, static_cast<int>(std::lround(fps)));
  simaai::neat::InputOptions input_options;
  input_options.payload_type = simaai::neat::PayloadType::Image;
  input_options.format = "RGB";
  input_options.width = width;
  input_options.height = height;
  input_options.depth = 3;
  input_options.fps_n = output_fps;
  input_options.fps_d = 1;
  input_options.memory_policy = simaai::neat::InputMemoryPolicy::Ev74;

  auto sender_options = simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(
      width, height, output_fps);
  sender_options.host = cfg.insight_host;
  sender_options.channel = cfg.insight_channel;
  sender_options.video_port_base = cfg.insight_video_port;
  sender_options.encoder.bitrate_kbps = cfg.insight_bitrate_kbps;

  VideoSender sender;
  sender.port = sender_options.video_port();
  sender.graph.add(simaai::neat::nodes::Input(input_options));
  sender.graph.add(simaai::neat::nodes::groups::VideoSender(sender_options));
  cv::Mat seed(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  const auto tensor = simaai::neat::Tensor::from_cv_mat(
      seed, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);
  sender.run = sender.graph.build(simaai::neat::TensorList{tensor});
  return sender;
}

void stream_frame(simaai::neat::Run& run, const cv::Mat& frame_bgr) {
  if (!run.push(simaai::neat::TensorList{rgb_tensor(frame_bgr)})) {
    throw std::runtime_error("Insight video push failed");
  }
}

int cmd_score_video_file(const Config& cfg, const patchcore::MemoryBank& bank, float threshold,
                         int num_neighbors) {
  cv::VideoCapture video(cfg.video_path.string());
  cv::Mat frame;
  if (!video.isOpened() || !video.read(frame)) {
    std::cerr << "[FATAL] failed to open video source: " << cfg.video_path << "\n";
    return 2;
  }
  const int width = frame.cols;
  const int height = frame.rows;

  const double input_fps = video.get(cv::CAP_PROP_FPS);
  const double fps = std::isfinite(input_fps) && input_fps > 0.0 ? input_fps : 30.0;

  simaai::neat::Model model(cfg.model_path, image_model_options());
  auto runner = model.build(simaai::neat::TensorList{rgb_tensor(frame)});
  auto video_sender = build_video_sender(cfg, fps, width, height);
  std::cout << "streaming to Insight: " << cfg.insight_host << ":" << video_sender.port << "\n";

  if (cfg.save_every > 0) {
    fs::create_directories(cfg.output_dir);
  }

  int processed = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    const double mla_start = time_ms();
    auto tensors = runner.run(simaai::neat::TensorList{rgb_tensor(frame)}, cfg.timeout_ms);
    if (tensors.empty()) {
      throw std::runtime_error("model returned no output tensors");
    }
    const auto flat = sima_examples::tensor_to_floats(tensors.front());
    const auto embedding = patchcore::extract_hwc(tensors.front().shape, flat, kEmbedDim);
    const double mla_ms = time_ms() - mla_start;

    const double host_start = time_ms();
    const auto scored = bank.score(embedding, num_neighbors);
    const cv::Mat overlay = draw_overlay(frame, scored, embedding.height, embedding.width,
                                         cfg.gaussian_sigma, cfg.overlay_alpha);
    const double host_ms = time_ms() - host_start;

    ++processed;
    const bool anomalous = scored.image_score >= threshold;
    std::cout << "frame=" << processed << ": score=" << scored.image_score
              << " threshold=" << threshold << " verdict=" << (anomalous ? "ANOMALOUS" : "normal")
              << " (mla=" << mla_ms << "ms host=" << host_ms << "ms)\n";

    stream_frame(video_sender.run, overlay);
    if (cfg.save_every > 0 && processed % cfg.save_every == 0) {
      cv::imwrite((cfg.output_dir / ("frame_" + std::to_string(processed) + ".jpg")).string(),
                 overlay);
    }

    if ((cfg.frames > 0 && processed >= cfg.frames) || !video.read(frame)) {
      break;
    }
  }

  runner.close();
  video_sender.run.close();
  std::cout << "Done: " << processed << " frames processed  video_sender=" << cfg.insight_host
            << ":" << video_sender.port << "\n";
  return processed > 0 ? 0 : 3;
}

// ---------------------------------------------------------------------------
// Score: rtsp -- decode-only RtspDecodedInput graph; the host pulls each raw
// decoded frame, scores it, and host-pushes the annotated heatmap overlay to
// Insight via the same VideoSender helper as video_file, rather than
// embedding the model in the live graph. A Model-object graph route hardcodes
// a small buffer pool that can't survive the model's one-time MLA warm-up
// stall against a live source; scoring host-side sidesteps that entirely.
// ---------------------------------------------------------------------------

struct SourceGeometry {
  int width = 0;
  int height = 0;
  int fps = 0;
};

SourceGeometry probe_rtsp_geometry(const Config& cfg) {
  sima_examples::RtspStreamInfo probe;
  sima_examples::RtspProbeOptions probe_options;
  probe_options.latency_ms = cfg.rtsp_latency_ms;
  probe_options.rtsp_tcp = cfg.rtsp_tcp;
  (void)sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, probe_options, probe);

  SourceGeometry geometry;
  geometry.width = cfg.rtsp_width > 0 ? cfg.rtsp_width : probe.width;
  geometry.height = cfg.rtsp_height > 0 ? cfg.rtsp_height : probe.height;
  geometry.fps = probe.fps;
  if (cfg.rtsp_codec == RtspCodec::Mjpeg && geometry.fps <= 0) {
    throw std::runtime_error(
        "MJPEG source did not provide a valid frame rate; set source.rtsp.width/height or use a "
        "source with probeable FPS metadata");
  }
  return geometry;
}

simaai::neat::Graph make_rtsp_source_fragment(const Config& cfg, const SourceGeometry& geometry) {
  simaai::neat::nodes::groups::RtspDecodedInputOptions opt;
  opt.url = cfg.rtsp_url;
  opt.latency_ms = cfg.rtsp_latency_ms;
  opt.tcp = cfg.rtsp_tcp;
  opt.insert_queue = true;
  opt.out_format = "NV12";
  opt.decoder_name = "decoder";
  opt.decoder_raw_output = true;
  opt.codec = cfg.rtsp_codec == RtspCodec::H264
                 ? simaai::neat::nodes::groups::RtspCodec::H264
                 : cfg.rtsp_codec == RtspCodec::H265
                       ? simaai::neat::nodes::groups::RtspCodec::H265
                       : simaai::neat::nodes::groups::RtspCodec::MJPEG;
  opt.source_fps = geometry.fps;
  if (cfg.rtsp_codec == RtspCodec::H264) {
    opt.auto_caps_from_stream = true;
    opt.fallback_h264_width = geometry.width;
    opt.fallback_h264_height = geometry.height;
  } else {
    // h265/mjpeg caps aren't self-describing the way h264 SPS is, so Neat
    // needs an explicit hint; h264 negotiates it from the stream itself.
    opt.dec_width = geometry.width;
    opt.dec_height = geometry.height;
  }
  return simaai::neat::nodes::groups::RtspDecodedInput(opt);
}

const simaai::neat::Sample* find_field(const simaai::neat::Sample& sample,
                                       const std::string& label) {
  if (sample.stream_label == label) {
    return &sample;
  }
  for (const auto& field : sample.fields) {
    if (const auto* found = find_field(field, label)) {
      return found;
    }
  }
  return nullptr;
}

cv::Mat frame_bgr_from_sample(const simaai::neat::Sample& sample) {
  // Falls back to the sample itself for a plain (non-joined) single-output pull.
  const auto* field = find_field(sample, "frame");
  if (field == nullptr) {
    field = &sample;
  }
  const auto tensors = simaai::neat::tensors_from_sample(*field, true);
  if (tensors.empty()) {
    throw std::runtime_error("frame field has no tensor");
  }
  cv::Mat bgr;
  std::string err;
  if (sima_examples::nv12_to_bgr(tensors.front(), bgr, err)) {
    return bgr;
  }
  return tensors.front().to_cv_mat_copy(simaai::neat::ImageSpec::PixelFormat::BGR);
}

struct RtspRuntime {
  simaai::neat::Graph graph;
  simaai::neat::Run run;
};

RtspRuntime build_rtsp_runtime(const Config& cfg, const SourceGeometry& geometry) {
  RtspRuntime rt;
  auto source = make_rtsp_source_fragment(cfg, geometry);

  rt.graph = simaai::neat::Graph("patchcore");
  rt.graph.connect(source, simaai::neat::nodes::Output("frame"));

  simaai::neat::RunOptions run_options;
  run_options.preset = simaai::neat::RunPreset::Realtime;
  run_options.queue_depth = 3;
  run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
  rt.run = rt.graph.build(run_options);
  return rt;
}

int cmd_score_rtsp(const Config& cfg, const patchcore::MemoryBank& bank, float threshold,
                   int num_neighbors) {
  const auto geometry = probe_rtsp_geometry(cfg);
  if (geometry.width <= 0 || geometry.height <= 0 || geometry.fps <= 0) {
    std::cerr << "[FATAL] failed to resolve source geometry for " << cfg.rtsp_url << "\n";
    return 2;
  }

  simaai::neat::Model model(cfg.model_path, image_model_options());
  auto rt = build_rtsp_runtime(cfg, geometry);
  auto video_sender = build_video_sender(cfg, geometry.fps, geometry.width, geometry.height);
  std::cout << "streaming to Insight: " << cfg.insight_host << ":" << video_sender.port << "\n";

  const bool save_frames = cfg.save_every > 0;
  if (save_frames) {
    fs::create_directories(cfg.output_dir);
  }

  int processed = 0;
  while (cfg.frames <= 0 || processed < cfg.frames) {
    simaai::neat::Sample sample;
    simaai::neat::PullError pull_error;
    // -1 waits forever, so a network stall doesn't end the loop.
    const auto status = rt.run.pull("frame", -1, sample, &pull_error);
    if (status == simaai::neat::PullStatus::Closed) {
      break;
    }
    if (status != simaai::neat::PullStatus::Ok) {
      throw std::runtime_error("failed to pull output: " + pull_error.message);
    }

    const cv::Mat bgr = frame_bgr_from_sample(sample);
    const double mla_start = time_ms();
    const auto embedding = extract_from_bgr(model, bgr, cfg.timeout_ms);
    const double mla_ms = time_ms() - mla_start;

    const double host_start = time_ms();
    const auto scored = bank.score(embedding, num_neighbors);
    const cv::Mat overlay = draw_overlay(bgr, scored, embedding.height, embedding.width,
                                         cfg.gaussian_sigma, cfg.overlay_alpha);
    const bool anomalous = scored.image_score >= threshold;
    ++processed;
    std::cout << "frame=" << processed << ": score=" << scored.image_score
              << " threshold=" << threshold << " verdict=" << (anomalous ? "ANOMALOUS" : "normal")
              << " (mla=" << mla_ms << "ms host=" << (time_ms() - host_start) << "ms)\n";

    stream_frame(video_sender.run, overlay);
    if (save_frames && processed % cfg.save_every == 0) {
      cv::imwrite((cfg.output_dir / ("frame_" + std::to_string(processed) + ".jpg")).string(),
                 overlay);
    }
  }

  rt.run.close();
  video_sender.run.close();
  std::cout << "Done: " << processed << " frames processed  video_sender=" << cfg.insight_host
            << ":" << video_sender.port << "\n";
  return processed > 0 ? 0 : 3;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const CliOptions cli = parse_args(argc, argv);
    if (cli.validate_config_only) {
      load_config(cli.config_path);
      std::cout << "Config validated: " << cli.config_path << "\n";
      return 0;
    }
    const Config cfg = load_config(cli.config_path);

    if (cli.calibrate) {
      return cmd_calibrate(cfg);
    }

    if (!fs::exists(cfg.memory_bank_path) || !fs::exists(cfg.bank_meta_path)) {
      std::cerr << "[FATAL] memory bank not found: " << cfg.memory_bank_path << " / "
                << cfg.bank_meta_path
                << "\n        Build it first: patchcore --calibrate --config <path>\n";
      return 2;
    }
    const auto meta = patchcore::load_bank_meta(cfg.bank_meta_path);
    std::cout << "Verifying model package...\n";
    patchcore::verify_bank_matches_model(meta, cfg.model_path);
    patchcore::verify_bank_hash(meta, cfg.memory_bank_path);
    const auto bank = patchcore::MemoryBank::load(cfg.memory_bank_path);
    const auto threshold = static_cast<float>(meta.threshold_value);
    // num_neighbors changes the neighborhood-reweighting term, which changes
    // the score distribution the threshold above was derived from. Score with
    // the value the bank was actually calibrated with (like the threshold
    // itself), not whatever the live config currently says -- otherwise a
    // config edit after calibration silently compares scores and a threshold
    // from different distributions.
    const int num_neighbors = meta.num_neighbors;
    if (num_neighbors != cfg.num_neighbors) {
      std::cerr << "[WARN] scoring.num_neighbors=" << cfg.num_neighbors
                << " in config differs from the bank's calibrated value (" << num_neighbors
                << "); using " << num_neighbors
                << " to stay consistent with the saved threshold. Recalibrate to adopt the new "
                   "value.\n";
    }

    if (cfg.source_type == SourceType::ImageDir) {
      return cmd_score_image_dir(cfg, bank, threshold, num_neighbors);
    }
    if (cfg.source_type == SourceType::VideoFile) {
      return cmd_score_video_file(cfg, bank, threshold, num_neighbors);
    }
    return cmd_score_rtsp(cfg, bank, threshold, num_neighbors);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
