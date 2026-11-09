/**
 * @example superpoint-feature-matcher.cpp
 * Match a live stream's SuperPoint keypoints to a reference frame with LightGlue or SuperGlue.
 *
 * The first frame with 512 keypoints becomes the reference. Every later frame runs SuperPoint, the
 * selected matcher matches its keypoints to the reference's, and Insight receives the reference on
 * the left and the live frame on the right, with one line per match colored by its score.
 *
 * Usage: superpoint-feature-matcher [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/pull_status.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

// Both matchers take exactly 512 keypoints per image and have no padding mask.
constexpr int kKeypoints = 512;
// SuperPoint's input size, which both matchers' keypoint normalization assumes.
const cv::Size kModelInputSize(640, 480);
constexpr int kSourceTimeoutMs = 20000;
constexpr int kModelTimeoutMs = 10000;
// Frames each model runner may hold, so that its stages overlap: on the board, SuperPoint
// alone runs at 11.6 ms per frame one at a time and 7.4 ms with two frames in flight.
constexpr int kRunnerQueueDepth = 2;
// Frames waiting between two stages; a full queue holds back the stage before it.
constexpr std::size_t kStageQueueDepth = 2;
// Frames matched and drawn at once, each on its own worker. On the board one frame takes up to
// about 50 ms of CPU: drawing 430 anti-aliased lines, or SuperGlue's Sinkhorn iterations plus
// drawing, so a 30 fps stream with that many matches needs several frames in parallel.
constexpr std::size_t kRenderWorkers = 3;
// The summary's inlier ratio samples every tenth frame: RANSAC takes about 18 ms.
constexpr int kInlierSampleInterval = 10;
// Frames the displayed rate is measured over.
constexpr std::size_t kFpsWindow = 30;
// LightGlue's learned keypoint projection (positional_encoder.projector.weight, [32, 2]) in
// ETH-CVG/lightglue_superpoint revision 5f5f626e, Apache-2.0. Each row turns a normalized
// keypoint into one rotary phase.
constexpr std::array<std::array<double, 2>, 32> kLightGlueProjection{{
    {0.07869736105203629, 0.6576962471008301},      {0.30121079087257385, -1.3067657947540283},
    {-0.17319072782993317, 0.3640412390232086},     {-1.8221570253372192, 0.07125379890203476},
    {4.831599235534668, 0.024766402319073677},      {-0.6115880608558655, -0.5418711304664612},
    {0.5497006177902222, -0.7080969214439392},      {0.9728097915649414, 0.8288111686706543},
    {-0.18647703528404236, 0.2699703872203827},     {1.3076542615890503, -1.4809051752090454},
    {-0.19871585071086884, 2.5542619228363037},     {-1.342088222503662, -1.5769871473312378},
    {-0.023420853540301323, -0.2613159120082855},   {-0.5988006591796875, -4.0408616065979},
    {0.4600417912006378, -0.07377243041992188},     {-2.12711763381958, -1.0835118293762207},
    {-0.6925885677337646, 1.3105987310409546},      {0.25808796286582947, -0.06581223011016846},
    {0.09479046612977982, -0.44248539209365845},    {-1.2112597227096558, -1.857155680656433},
    {-0.24696263670921326, -0.0016804899787530303}, {-2.5080809593200684, 3.7625091075897217},
    {-1.390500783920288, 0.7762705087661743},       {0.3077039122581482, -1.0518392324447632},
    {-1.6019573211669922, -0.010330487042665482},   {0.5500544309616089, 2.5014572143554688},
    {-1.0771361589431763, 0.19658993184566498},     {0.25206997990608215, -0.5227173566818237},
    {-0.5797532200813293, 0.07100092619657516},     {-0.36304882168769836, 1.7311886548995972},
    {0.554409921169281, 1.0818008184432983},        {0.4458959400653839, 0.6291354298591614},
}};
const std::vector<std::string> kSourceTypes{"rtsp", "http", "usb", "mipi"};
const std::vector<std::string> kMatcherTypes{"lightglue", "superglue"};
const std::vector<std::string> kRtspCodecs{"h264", "h265", "mjpeg"};
// Composed frames this large or larger need H.264 level 5.1; smaller ones fit level 4.0.
constexpr int kLevel4MaxMacroblocks = 8192;

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int) {
  stop_requested = 1;
}

struct AppConfig {
  std::string superpoint_path;
  std::string matcher_type;
  std::string matcher_path;
  double match_threshold = 0.0;
  int sinkhorn_iterations = 0;
  std::string source_type;
  // The key that names the source in errors: source.<type>.url or source.<type>.camera.
  std::string source_key;
  std::string source_url;
  std::string source_codec;
  bool tcp = true;
  int latency_ms = 0;
  bool ssl_strict = true;
  // source.usb and source.mipi: the camera and the exact mode it captures.
  std::string camera;
  int source_width = 0;
  int source_height = 0;
  // 0 probes the stream; USB and MIPI cameras take it from their mode.
  int source_fps = 0;
  int frames = 0;
  std::string insight_host;
  int video_port = 0;
  int bitrate_kbps = 0;
  std::string save_dir;
  int save_every = 0;
};

// One frame's 512 SuperPoint keypoints, in pixels of the model's input.
struct Features {
  cv::Mat keypoints;   // CV_32F [512, 2], (x, y)
  cv::Mat scores;      // CV_32F [512, 1]
  cv::Mat descriptors; // CV_32F [512, 256], unit length
};

// Keypoint pairs, as indices into the reference's and the live frame's keypoints.
struct Matches {
  std::vector<int> reference;
  std::vector<int> live;
  std::vector<float> scores;
};

// The selected matcher's CPU steps around its model.
struct Matcher {
  std::string name;
  // One image's model inputs from its features, in the model's input order.
  std::function<neat::TensorList(const Features&)> image_inputs;
  // The matches in the model's output scores.
  std::function<Matches(const std::vector<float>&)> matches;
};

// A decoded frame: its luma plane, for SuperPoint, and its BGR pixels, for the view.
struct DecodedFrame {
  cv::Mat luma;
  cv::Mat bgr;
};

// The reference frame and its features.
struct Reference {
  cv::Mat bgr;
  Features features;
};

// A frame on its way to the publisher; `features` is empty when it has fewer than 512.
struct LiveFrame {
  cv::Mat bgr;
  std::optional<Features> features;
};

// One composed frame and what it adds to the summary.
struct Rendered {
  cv::Mat canvas;
  std::optional<int> matches;
  std::optional<double> inlier_ratio;
};

// Match line endpoints in pixels of the composed frame, (x0, y0, x1, y1), and their scores.
struct MatchLines {
  std::vector<cv::Vec4i> endpoints;
  std::vector<float> scores;
};

// Frames per second over a list of send times, in seconds.
double rate(std::span<const double> times) {
  return times.size() > 1 ? static_cast<double>(times.size() - 1) / (times.back() - times.front())
                          : 0.0;
}

double now_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// What the publisher sent, for the summary line.
struct MatchStats {
  int frames = 0;
  int matched = 0;
  long long matches = 0;
  std::vector<double> inlier_ratios;
  std::vector<double> sent_at;

  std::string summary() const {
    const double average = matched ? static_cast<double>(matches) / matched : 0.0;
    const double inliers = inlier_ratios.empty()
                               ? 0.0
                               : std::accumulate(inlier_ratios.begin(), inlier_ratios.end(), 0.0) /
                                     static_cast<double>(inlier_ratios.size());
    return cv::format("frames=%d matched=%d skipped=%d average_matches=%.1f inlier_ratio=%.2f "
                      "output_fps=%.1f",
                      frames, matched, frames - matched, average, inliers, rate(sent_at));
  }
};

// One mapping of the config. Reports a bad value by its own key, as the Python app does.
class ConfigSection {
public:
  ConfigSection(const sima_examples::ScalarConfig& raw, std::string path)
      : raw_(raw), path_(std::move(path)) {}

  ConfigSection section(const std::string& key) const {
    if (raw_.string_value(full(key))) {
      throw std::invalid_argument(key + " must be a mapping");
    }
    return ConfigSection(raw_, full(key));
  }

  std::string string_or(const std::string& key, const std::string& fallback = "") const {
    return raw_.string_or(full(key), fallback);
  }

  int int_or(const std::string& key, int fallback) const {
    try {
      return raw_.int_or(full(key), fallback);
    } catch (const std::exception&) {
      throw std::invalid_argument(key + " must be an integer");
    }
  }

  double float_or(const std::string& key, double fallback) const {
    try {
      return raw_.double_or(full(key), fallback);
    } catch (const std::exception&) {
      throw std::invalid_argument(key + " must be numeric");
    }
  }

  bool bool_or(const std::string& key, bool fallback) const {
    try {
      return raw_.bool_or(full(key), fallback);
    } catch (const std::exception&) {
      throw std::invalid_argument(key + " must be true or false");
    }
  }

private:
  std::string full(const std::string& key) const {
    return path_.empty() ? key : path_ + "." + key;
  }

  const sima_examples::ScalarConfig& raw_;
  std::string path_;
};

std::string choice(std::string value, const std::vector<std::string>& choices,
                   const std::string& key) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
    return value;
  }
  std::string allowed;
  for (std::size_t i = 0; i + 1 < choices.size(); ++i) {
    allowed += (i ? ", " : "") + choices[i];
  }
  throw std::invalid_argument(key + " must be " + allowed + " or " + choices.back());
}

void validate_config(const AppConfig& cfg) {
  const auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  require(!cfg.superpoint_path.empty(), "superpoint.path must be set");
  require(!cfg.matcher_path.empty(), "matcher." + cfg.matcher_type + ".path must be set");
  require(cfg.match_threshold >= 0.0 && cfg.match_threshold < 1.0,
          "matcher." + cfg.matcher_type + ".match_threshold must be in [0, 1)");
  require(cfg.sinkhorn_iterations > 0, "matcher.superglue.sinkhorn_iterations must be > 0");
  const bool stream = cfg.source_type == "rtsp" || cfg.source_type == "http";
  const bool camera = cfg.source_type == "usb" || cfg.source_type == "mipi";
  require(!stream || !cfg.source_url.empty(), cfg.source_key + " must be set");
  require(cfg.source_type != "usb" || !cfg.camera.empty(), "source.usb.camera must be set");
  if (camera) {
    const std::string prefix = "source." + cfg.source_type;
    require(cfg.source_width > 0 && cfg.source_height > 0 && cfg.source_fps > 0,
            prefix + ".width, " + prefix + ".height and " + prefix + ".fps must be > 0");
  }
  require(cfg.latency_ms >= 0, "source.rtsp.latency_ms must be >= 0");
  require(cfg.source_fps >= 0, "source." + cfg.source_type + ".fps must be >= 0");
  require(cfg.frames >= 0, "inference.frames must be >= 0");
  require(!cfg.insight_host.empty(), "output.insight.host must be set");
  require(cfg.video_port > 0 && cfg.video_port <= 65535,
          "output.insight.video_port must be in [1, 65535]");
  require(cfg.bitrate_kbps > 0, "output.insight.bitrate_kbps must be > 0");
  require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

AppConfig load_app_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  const ConfigSection root(raw, "");
  const ConfigSection superpoint = root.section("superpoint");
  const ConfigSection matcher = root.section("matcher");
  const ConfigSection source = root.section("source");
  const ConfigSection inference = root.section("inference");
  const ConfigSection output = root.section("output");
  const ConfigSection insight = output.section("insight");

  // Only the sections that source.type and matcher.type name are read; the others may hold
  // placeholders.
  AppConfig cfg;
  cfg.matcher_type = choice(matcher.string_or("type", "lightglue"), kMatcherTypes, "matcher.type");
  const ConfigSection selected_matcher = matcher.section(cfg.matcher_type);
  cfg.source_type = choice(source.string_or("type", "rtsp"), kSourceTypes, "source.type");
  const ConfigSection selected_source = source.section(cfg.source_type);
  const bool camera_source = cfg.source_type == "usb" || cfg.source_type == "mipi";
  const std::string codec =
      cfg.source_type == "rtsp" ? selected_source.string_or("codec", "h264") : "mjpeg";

  cfg.superpoint_path = superpoint.string_or("path");
  cfg.matcher_path = selected_matcher.string_or("path");
  cfg.match_threshold =
      selected_matcher.float_or("match_threshold", cfg.matcher_type == "lightglue" ? 0.1 : 0.2);
  cfg.sinkhorn_iterations = selected_matcher.int_or("sinkhorn_iterations", 20);
  cfg.source_key = "source." + cfg.source_type + "." + (camera_source ? "camera" : "url");
  cfg.source_url = selected_source.string_or("url");
  cfg.source_codec = choice(codec, kRtspCodecs, "source.rtsp.codec");
  cfg.tcp = selected_source.bool_or("tcp", true);
  cfg.latency_ms = selected_source.int_or("latency_ms", 100);
  cfg.ssl_strict = selected_source.bool_or("ssl_strict", true);
  cfg.camera = selected_source.string_or("camera");
  cfg.source_width = selected_source.int_or("width", 0);
  cfg.source_height = selected_source.int_or("height", 0);
  cfg.source_fps = selected_source.int_or("fps", 0);
  cfg.frames = inference.int_or("frames", 0);
  cfg.insight_host = insight.string_or("host");
  cfg.video_port = insight.int_or("video_port", 9000);
  cfg.bitrate_kbps = insight.int_or("bitrate_kbps", 8000);
  cfg.save_dir = output.string_or("save_dir");
  cfg.save_every = output.int_or("save_every", 0);
  validate_config(cfg);
  return cfg;
}

fs::path parse_config_path(int argc, char** argv) {
  fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::invalid_argument("--config requires a path");
      }
      config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  return config_path;
}

// The configured frame rate, or, when it is 0, the one an RTSP or HTTP stream reports. The
// encoder needs it before the first frame.
int stream_fps(const AppConfig& cfg) {
  if (cfg.source_fps > 0) {
    return cfg.source_fps;
  }
  cv::VideoCapture capture(cfg.source_url);
  const int fps =
      capture.isOpened() ? static_cast<int>(std::nearbyint(capture.get(cv::CAP_PROP_FPS))) : 0;
  capture.release();
  if (fps <= 0) {
    throw std::runtime_error("could not read the frame rate of " + cfg.source_key +
                             "; set source." + cfg.source_type + ".fps");
  }
  return fps;
}

groups::RtspDecodedInputOptions rtsp_source_options(const AppConfig& cfg, int fps) {
  groups::RtspDecodedInputOptions options;
  options.url = cfg.source_url;
  options.tcp = cfg.tcp;
  options.latency_ms = cfg.latency_ms;
  options.source_fps = fps;
  options.codec = cfg.source_codec == "h264"   ? groups::RtspCodec::H264
                  : cfg.source_codec == "h265" ? groups::RtspCodec::H265
                                               : groups::RtspCodec::MJPEG;
  options.auto_caps_from_stream = cfg.source_codec != "mjpeg";
  return options;
}

groups::HttpMjpegDecodedInputOptions http_source_options(const AppConfig& cfg, int fps) {
  groups::HttpMjpegDecodedInputOptions options;
  options.url = cfg.source_url;
  options.ssl_strict = cfg.ssl_strict;
  options.source_fps = fps;
  return options;
}

neat::UsbCameraInputOptions usb_camera_options(const AppConfig& cfg) {
  neat::UsbCameraInputOptions options;
  options.camera_name = cfg.camera;
  options.width = static_cast<std::uint32_t>(cfg.source_width);
  options.height = static_cast<std::uint32_t>(cfg.source_height);
  options.framerate_num = static_cast<std::uint32_t>(cfg.source_fps);
  return options;
}

neat::SimaDecodeOptions usb_decode_options(const AppConfig& cfg, neat::FormatTag decoded_format) {
  neat::SimaDecodeOptions options;
  options.type = neat::SimaDecodeType::MJPEG;
  options.raw_output = true;
  options.out_format = decoded_format;
  options.dec_width = cfg.source_width;
  options.dec_height = cfg.source_height;
  options.dec_fps = cfg.source_fps;
  return options;
}

neat::CameraInputOptions mipi_camera_options(const AppConfig& cfg) {
  neat::CameraInputOptions options;
  if (!cfg.camera.empty()) {
    options.camera_name = cfg.camera;
  }
  options.width = static_cast<std::uint32_t>(cfg.source_width);
  options.height = static_cast<std::uint32_t>(cfg.source_height);
  options.framerate_num = static_cast<std::uint32_t>(cfg.source_fps);
  options.format = "NV12";
  return options;
}

struct SourceRun {
  neat::Graph graph{"source"};
  neat::Run run;
};

// The decoded frames of the selected source, newest first: a slow consumer skips frames.
SourceRun build_source_run(const AppConfig& cfg, int fps) {
  SourceRun source;
  if (cfg.source_type == "rtsp") {
    source.graph.add(groups::RtspDecodedInput(rtsp_source_options(cfg, fps)));
  } else if (cfg.source_type == "http") {
    source.graph.add(groups::HttpMjpegDecodedInput(http_source_options(cfg, fps)));
  } else if (cfg.source_type == "usb") {
    const auto camera = usb_camera_options(cfg);
    // Checks the camera and its mode once; most webcams send 4:2:2 JPEG, which decodes to NV16.
    const auto info = neat::usb_camera_info(camera);
    source.graph.add(neat::nodes::UsbCameraInput(camera, info));
    source.graph.add(neat::nodes::SimaDecode(usb_decode_options(cfg, info.decoded_format)));
  } else {
    source.graph.add(neat::nodes::CameraInput(mipi_camera_options(cfg)));
  }
  auto newest = neat::OutputOptions::Latest();
  newest.max_buffers = 2;
  source.graph.add(neat::nodes::Output("frame", newest));

  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = 2;
  run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = neat::OutputMemory::ZeroCopy;
  source.run = source.graph.build(run_options);
  return source;
}

// SuperPoint on the frame's luma: Neat stretches it to the model input and scales it to [0, 1].
// The LightGlueV1 decode (threshold 0.0005, NMS radius 4, 4-pixel border) is the one both
// matchers' keypoints were extracted with, keeping the best 512 keypoints.
neat::Model::Options superpoint_options(int width, int height) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.input_max_width = width;
  options.preprocess.input_max_height = height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.color_convert.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::GRAY8;
  options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::GRAY8;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  options.preprocess.normalize.mean = {0.0F, 0.0F, 0.0F};
  options.preprocess.normalize.stddev = {1.0F, 1.0F, 1.0F};
  options.preprocess.normalize.has_explicit_stats = true;
  options.decode_type = neat::BoxDecodeType::SuperPoint;
  options.superpoint.profile = neat::SuperPointProfile::LightGlueV1;
  options.superpoint.output_format = neat::SuperPointOutputFormat::FeaturePointsV1;
  options.superpoint.descriptor_output_dtype = neat::TensorDType::Float32;
  options.top_k = kKeypoints;
  return options;
}

// The matchers take keypoint features, not images: Neat only quantizes and tessellates them.
neat::Model::Options matcher_options() {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Tensor;
  options.preprocess.resize.enable = neat::AutoFlag::Off;
  options.preprocess.color_convert.enable = neat::AutoFlag::Off;
  options.preprocess.normalize.enable = neat::AutoFlag::Off;
  return options;
}

// The SuperPoint input size, the frame its keypoints are in (sima-neat/core#1328).
cv::Size model_input_size(const neat::Model& model) {
  const auto contract = model.resolved_preprocess_plan().mla_contract;
  if (contract.width <= 0 || contract.height <= 0) {
    throw std::runtime_error("the SuperPoint model input size is unresolved");
  }
  return {contract.width, contract.height};
}

neat::Tensor frame_tensor(const neat::Sample& sample) {
  const neat::Sample* field = &sample;
  if (!sample.fields.empty()) {
    const auto frame = std::find_if(sample.fields.begin(), sample.fields.end(),
                                    [](const auto& part) { return part.stream_label == "frame"; });
    if (frame == sample.fields.end()) {
      throw std::runtime_error("the source sample has no frame field");
    }
    field = &*frame;
  }
  return neat::tensors_from_sample(*field).front();
}

// A decoded frame's luma plane and BGR pixels. MJPEG with 4:2:2 chroma, as most webcams send,
// decodes to NV16 instead of NV12; each frame is converted by the layout it actually has.
DecodedFrame decoded_frame(const neat::Tensor& tensor) {
  DecodedFrame frame;
  std::string error;
  if (tensor.image_format() == neat::ImageSpec::PixelFormat::NV16) {
    cv::Mat yuy2;
    sima_examples::require(sima_examples::nv16_to_yuy2(tensor, yuy2, error), error);
    cv::cvtColor(yuy2, frame.bgr, cv::COLOR_YUV2BGR_YUY2);
    cv::extractChannel(yuy2, frame.luma, 0);
    return frame;
  }
  if (!tensor.is_nv12()) {
    throw std::runtime_error("unsupported decoded frame format: " + tensor.debug_string());
  }
  // Decoders pad rows and planes; the contiguous copy reads each plane at its own stride.
  std::vector<uint8_t> nv12 = tensor.copy_nv12_contiguous();
  const int width = tensor.width();
  const int height = tensor.height();
  sima_examples::require(!nv12.empty() && width > 0 && height > 0, "NV12 frame copy failed");
  const cv::Mat yuv(height + height / 2, width, CV_8UC1, nv12.data());
  cv::cvtColor(yuv, frame.bgr, cv::COLOR_YUV2BGR_NV12);
  frame.luma = yuv.rowRange(0, height).clone();
  return frame;
}

// The newest decoded frame. The decoder's buffer is released on return: the frame is a CPU copy.
DecodedFrame pull_frame(neat::Run& source_run, const std::string& source_key) {
  neat::Sample sample;
  neat::PullError error;
  const auto status = source_run.pull("frame", kSourceTimeoutMs, sample, &error);
  if (!sima_examples::pull_status_has_sample(status, "frame", error, source_run.last_error())) {
    throw std::runtime_error("no frame arrived from " + source_key + " for " +
                             std::to_string(kSourceTimeoutMs) + " ms");
  }
  return decoded_frame(frame_tensor(sample));
}

neat::Tensor luma_tensor(const cv::Mat& luma) {
  return neat::Tensor::from_cv_mat(luma, neat::ImageSpec::PixelFormat::GRAY8,
                                   neat::TensorMemory::EV74);
}

// A Float32 tensor's values as a CV_32F matrix with `cols` columns.
cv::Mat float_rows(const neat::Tensor& tensor, int cols) {
  const std::vector<float> values = sima_examples::tensor_to_floats(tensor);
  return cv::Mat(static_cast<int>(values.size()) / cols, cols, CV_32F,
                 const_cast<float*>(values.data()))
      .clone();
}

// The frame's 512 best keypoints, or nothing when SuperPoint found fewer.
std::optional<Features> superpoint_features(const neat::TensorList& outputs) {
  const auto decoded = neat::decode_superpoint(outputs);
  if (decoded.size() != 1) {
    throw std::runtime_error("SuperPoint must return one feature set per frame");
  }
  const auto& points = decoded.front();
  if (points.keypoints.shape.empty() || points.keypoints.shape[0] < kKeypoints) {
    return std::nullopt;
  }
  if (points.keypoints.shape[0] != kKeypoints) {
    throw std::runtime_error("SuperPoint returned more than 512 keypoints");
  }
  return Features{float_rows(points.keypoints, 2), float_rows(points.scores, 1),
                  float_rows(points.descriptors, static_cast<int>(points.descriptors.shape[1]))};
}

// A [512, C] matcher input as the [1, 1, 512, C] EV74 tensor the model reads.
neat::Tensor model_tensor(const cv::Mat& rows) {
  const cv::Mat values = rows.isContinuous() ? rows : rows.clone();
  return neat::Tensor::from_vector(std::vector<float>(values.begin<float>(), values.end<float>()),
                                   {1, 1, kKeypoints, values.cols}, neat::TensorMemory::EV74);
}

// LightGlue's rotary encoding [512, 128] of SuperPoint's 640x480 keypoints: each keypoint
// normalized as ((x, y) - (320, 240)) / 320, projected to 32 phases in double precision, each
// phase repeated twice, then the cosines and the sines.
cv::Mat lightglue_encoding(const cv::Mat& keypoints) {
  cv::Mat encoding(keypoints.rows, 128, CV_32F);
  for (int i = 0; i < keypoints.rows; ++i) {
    const double x = (keypoints.at<float>(i, 0) - 320.0) / 320.0;
    const double y = (keypoints.at<float>(i, 1) - 240.0) / 320.0;
    float* row = encoding.ptr<float>(i);
    for (std::size_t j = 0; j < kLightGlueProjection.size(); ++j) {
      const double phase = x * kLightGlueProjection[j][0] + y * kLightGlueProjection[j][1];
      row[2 * j] = row[2 * j + 1] = static_cast<float>(std::cos(phase));
      row[64 + 2 * j] = row[65 + 2 * j] = static_cast<float>(std::sin(phase));
    }
  }
  return encoding;
}

neat::TensorList lightglue_inputs(const Features& features) {
  return {model_tensor(lightglue_encoding(features.keypoints)), model_tensor(features.descriptors)};
}

// The mutual best pairs of a [512, 512] score matrix whose score exceeds the threshold.
Matches mutual_matches(const cv::Mat& scores, float threshold) {
  std::vector<int> best_live(scores.rows);
  std::vector<int> best_reference(scores.cols, 0);
  std::vector<float> column_best(scores.cols, -std::numeric_limits<float>::infinity());
  for (int i = 0; i < scores.rows; ++i) {
    const float* row = scores.ptr<float>(i);
    best_live[i] = static_cast<int>(std::max_element(row, row + scores.cols) - row);
    for (int j = 0; j < scores.cols; ++j) {
      if (row[j] > column_best[j]) {
        column_best[j] = row[j];
        best_reference[j] = i;
      }
    }
  }
  Matches matches;
  for (int i = 0; i < scores.rows; ++i) {
    const int live = best_live[i];
    const float best = scores.at<float>(i, live);
    if (best_reference[live] == i && best > threshold) {
      matches.reference.push_back(i);
      matches.live.push_back(live);
      matches.scores.push_back(best);
    }
  }
  return matches;
}

// The model's flat output as a square score matrix.
cv::Mat score_matrix(const std::vector<float>& scores, int side, const std::string& matcher) {
  if (scores.size() != static_cast<std::size_t>(side) * side) {
    throw std::runtime_error(matcher + " returned " + std::to_string(scores.size()) +
                             " scores, expected " + std::to_string(side) + "x" +
                             std::to_string(side));
  }
  return cv::Mat(side, side, CV_32F, const_cast<float*>(scores.data()));
}

// LightGlue's assignment [512, 512] is the probability that two keypoints match.
Matches lightglue_matches(const std::vector<float>& scores, float threshold) {
  return mutual_matches(score_matrix(scores, kKeypoints, "LightGlue"), threshold);
}

// SuperGlue's inputs: keypoints centered on the 640x480 input and divided by 0.7 times its
// longer side, SuperPoint's scores, and the descriptors.
neat::TensorList superglue_inputs(const Features& features) {
  cv::Mat keypoints(kKeypoints, 2, CV_32F);
  for (int i = 0; i < kKeypoints; ++i) {
    keypoints.at<float>(i, 0) =
        static_cast<float>((features.keypoints.at<float>(i, 0) - 320.0) / (0.7 * 640.0));
    keypoints.at<float>(i, 1) =
        static_cast<float>((features.keypoints.at<float>(i, 1) - 240.0) / (0.7 * 640.0));
  }
  return {model_tensor(keypoints), model_tensor(features.scores),
          model_tensor(features.descriptors)};
}

// SuperGlue's Sinkhorn iterations over its [513, 513] couplings (index 512 is the dustbin), in
// the linear domain as in SuperGlue's match_pairs.py evaluation. Each score is the transport mass
// of a pair times M + N.
Matches superglue_matches(const std::vector<float>& scores, float threshold, int iterations) {
  const cv::Mat couplings = score_matrix(scores, kKeypoints + 1, "SuperGlue");
  const float total = 2.0F * kKeypoints;
  cv::Mat marginal(kKeypoints + 1, 1, CV_32F, cv::Scalar(1.0F / total));
  marginal.at<float>(kKeypoints) = kKeypoints / total;
  // Clamped above float32's smallest normal number so that no row or column sums to zero; the
  // clamped entries are negligible against every row's largest.
  double largest = 0.0;
  cv::minMaxLoc(couplings, nullptr, &largest);
  cv::Mat kernel;
  cv::exp(cv::max(couplings - largest, -80.0), kernel);
  const cv::Mat kernel_t = kernel.t();
  cv::Mat u = cv::Mat::ones(kKeypoints + 1, 1, CV_32F);
  cv::Mat v = cv::Mat::ones(kKeypoints + 1, 1, CV_32F);
  cv::Mat product;
  for (int i = 0; i < iterations; ++i) {
    cv::gemm(kernel, v, 1.0, cv::noArray(), 0.0, product);
    cv::divide(marginal, product, u);
    cv::gemm(kernel_t, u, 1.0, cv::noArray(), 0.0, product);
    cv::divide(marginal, product, v);
  }
  cv::Mat transport(kKeypoints, kKeypoints, CV_32F);
  for (int i = 0; i < kKeypoints; ++i) {
    const float* kernel_row = kernel.ptr<float>(i);
    float* row = transport.ptr<float>(i);
    for (int j = 0; j < kKeypoints; ++j) {
      row[j] = kernel_row[j] * (u.at<float>(i) * v.at<float>(j)) * total;
    }
  }
  return mutual_matches(transport, threshold);
}

Matcher select_matcher(const AppConfig& cfg) {
  const auto threshold = static_cast<float>(cfg.match_threshold);
  if (cfg.matcher_type == "lightglue") {
    return {"LightGlue", lightglue_inputs, [threshold](const std::vector<float>& scores) {
              return lightglue_matches(scores, threshold);
            }};
  }
  return {"SuperGlue", superglue_inputs,
          [threshold, iterations = cfg.sinkhorn_iterations](const std::vector<float>& scores) {
            return superglue_matches(scores, threshold, iterations);
          }};
}

neat::RunOptions pipelined_run_options() {
  neat::RunOptions run_options;
  run_options.queue_depth = kRunnerQueueDepth;
  run_options.overflow_policy = neat::OverflowPolicy::Block;
  return run_options;
}

std::string encoder_level(int width, int height) {
  const int macroblocks = ((width + 15) / 16) * ((height + 15) / 16);
  return macroblocks <= kLevel4MaxMacroblocks ? "4.0" : "5.1";
}

neat::Tensor bgr_tensor(const cv::Mat& bgr) {
  return neat::Tensor::from_cv_mat(bgr, neat::ImageSpec::PixelFormat::BGR,
                                   neat::TensorMemory::EV74);
}

struct InsightRun {
  neat::Graph graph{"insight"};
  neat::Run run;
  int video_port = 0;
};

// Composed BGR frames, converted to the encoder's NV12 on the CVU, to H.264 for Insight.
InsightRun build_insight_run(const AppConfig& cfg, int width, int height, int fps) {
  neat::InputOptions input_options;
  input_options.payload_type = neat::PayloadType::Image;
  input_options.format = neat::FormatTag::BGR;
  input_options.width = width;
  input_options.height = height;
  input_options.depth = 3;
  input_options.fps_n = fps;
  input_options.fps_d = 1;
  input_options.memory_policy = neat::InputMemoryPolicy::Ev74;

  neat::ColorConvertOptions nv12;
  nv12.output_format = neat::ImageFormat::NV12;
  nv12.surface = neat::ImageSurfaceLayout::EncoderNv12;

  auto sender_options = groups::VideoSenderOptions::FromRaw({.fps = fps});
  sender_options.host = cfg.insight_host;
  sender_options.video_port_base = cfg.video_port;
  sender_options.encoder.bitrate_kbps = cfg.bitrate_kbps;
  sender_options.encoder.level = encoder_level(width, height);

  InsightRun insight;
  insight.graph.add(neat::nodes::Input(input_options));
  insight.graph.add(neat::nodes::ColorConvert(nv12));
  insight.graph.add(groups::VideoSender(sender_options));
  const cv::Mat seed = cv::Mat::zeros(height, width, CV_8UC3);
  insight.run = insight.graph.build(neat::TensorList{bgr_tensor(seed)});
  insight.video_port = sender_options.video_port();
  return insight;
}

// BGR line colors from red at the match threshold to green at score 1.
std::vector<cv::Scalar> match_colors(const std::vector<float>& scores, float threshold) {
  std::vector<cv::Scalar> colors;
  colors.reserve(scores.size());
  for (const float score : scores) {
    const float strength = std::clamp((score - threshold) / (1.0F - threshold), 0.0F, 1.0F);
    colors.emplace_back(0, static_cast<int>(255 * strength),
                        static_cast<int>(255 * (1 - strength)));
  }
  return colors;
}

// Line endpoints in pixels of the composed frame and their scores, weakest match first, so the
// strongest lines are drawn on top. Keypoints are in model-input pixels until
// sima-neat/core#1328, so `scale` maps them to the frame.
MatchLines match_lines(const Matches& matches, const Features& reference, const Features& live,
                       cv::Point2f scale, int width) {
  std::vector<std::size_t> order(matches.scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return matches.scores[a] < matches.scores[b];
  });
  MatchLines lines;
  for (const std::size_t k : order) {
    const float* start = reference.keypoints.ptr<float>(matches.reference[k]);
    const float* end = live.keypoints.ptr<float>(matches.live[k]);
    lines.endpoints.emplace_back(static_cast<int>(std::lrint(start[0] * scale.x)),
                                 static_cast<int>(std::lrint(start[1] * scale.y)),
                                 static_cast<int>(std::lrint(end[0] * scale.x + width)),
                                 static_cast<int>(std::lrint(end[1] * scale.y)));
    lines.scores.push_back(matches.scores[k]);
  }
  return lines;
}

// The share of matches consistent with one camera motion (a fundamental matrix), the match
// quality the summary reports.
double inlier_ratio(const std::vector<cv::Vec4i>& lines) {
  if (lines.size() < 8) {
    return 0.0;
  }
  std::vector<cv::Point2f> start, end;
  for (const auto& line : lines) {
    start.emplace_back(static_cast<float>(line[0]), static_cast<float>(line[1]));
    end.emplace_back(static_cast<float>(line[2]), static_cast<float>(line[3]));
  }
  cv::Mat inliers;
  cv::findFundamentalMat(start, end, cv::FM_RANSAC, 1.0, 0.999, inliers);
  return inliers.empty()
             ? 0.0
             : static_cast<double>(cv::countNonZero(inliers)) / static_cast<double>(lines.size());
}

// White text on a darkened box, readable on any frame.
void draw_label(cv::Mat& canvas, const std::string& text, cv::Point origin, double scale) {
  const int thickness = std::max(1, static_cast<int>(std::nearbyint(scale * 1.5)));
  int baseline = 0;
  const cv::Size size =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  const int pad = std::max(4, size.height / 3);
  const cv::Rect box = cv::Rect(cv::Point(origin.x - pad, origin.y - size.height - pad),
                                cv::Point(origin.x + size.width + pad, origin.y + baseline + pad)) &
                       cv::Rect(0, 0, canvas.cols, canvas.rows);
  cv::Mat region = canvas(box);
  region /= 3;
  cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),
              thickness, cv::LINE_AA);
}

// The reference on the left and the live frame on the right, one line per match.
cv::Mat compose_frame(const cv::Mat& reference_bgr, const cv::Mat& live_bgr,
                      const MatchLines& lines, const std::vector<cv::Scalar>& colors,
                      const std::string& status) {
  const int height = reference_bgr.rows;
  const int width = reference_bgr.cols;
  cv::Mat canvas;
  cv::hconcat(reference_bgr, live_bgr, canvas);
  for (std::size_t i = 0; i < lines.endpoints.size(); ++i) {
    const cv::Point start(lines.endpoints[i][0], lines.endpoints[i][1]);
    const cv::Point end(lines.endpoints[i][2], lines.endpoints[i][3]);
    cv::line(canvas, start, end, colors[i], 1, cv::LINE_AA);
    cv::circle(canvas, start, 2, colors[i], cv::FILLED, cv::LINE_AA);
    cv::circle(canvas, end, 2, colors[i], cv::FILLED, cv::LINE_AA);
  }
  const double scale = height / 1200.0;
  const int margin = static_cast<int>(std::nearbyint(30 * scale));
  const int line_height = static_cast<int>(std::nearbyint(56 * scale));
  draw_label(canvas, "Reference", {margin, margin + line_height / 2}, scale);
  draw_label(canvas, "Live", {width + margin, margin + line_height / 2}, scale);
  draw_label(canvas, status, {width + margin, margin + line_height / 2 + line_height}, scale);
  draw_label(canvas, "Line color: match score, red low, green high", {margin, height - margin},
             scale);
  return canvas;
}

// A bounded queue between two stages. A full queue holds back the stage before it. Both calls
// give up once a stage has failed, and get returns nothing once the producer closed the queue.
template <typename T> class StageQueue {
public:
  StageQueue(std::size_t capacity, const std::atomic<bool>& failed)
      : capacity_(capacity), failed_(failed) {}

  bool put(T item) {
    std::unique_lock lock(mutex_);
    while (!failed_) {
      if (changed_.wait_for(lock, kPoll, [&] { return items_.size() < capacity_; })) {
        items_.push_back(std::move(item));
        changed_.notify_all();
        return true;
      }
    }
    return false;
  }

  std::optional<T> get() {
    std::unique_lock lock(mutex_);
    while (!failed_) {
      if (changed_.wait_for(lock, kPoll, [&] { return !items_.empty() || closed_; })) {
        if (items_.empty()) {
          return std::nullopt;
        }
        T item = std::move(items_.front());
        items_.pop_front();
        changed_.notify_all();
        return item;
      }
    }
    return std::nullopt;
  }

  void close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    changed_.notify_all();
  }

private:
  static constexpr std::chrono::milliseconds kPoll{100};
  const std::size_t capacity_;
  const std::atomic<bool>& failed_;
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<T> items_;
  bool closed_ = false;
};

// What the three stages share. run() owns the runners and the reference; the queues, the failure
// flag and the first error belong to this object; only the publisher writes the stats.
struct Stages {
  const AppConfig& cfg;
  const Matcher& matcher;
  neat::Run& source_run;
  neat::Model::Runner& superpoint_runner;
  neat::Model::Runner& matcher_runner;
  neat::Run& insight_run;
  const Reference& reference;
  // The reference's inputs are made once and sent with every live frame's.
  const neat::TensorList& reference_inputs;
  cv::Point2f point_scale;

  std::atomic<bool> failed{false};
  // Set once stage 1 has pushed its last frame.
  std::atomic<bool> captured_all{false};
  StageQueue<cv::Mat> captured{kStageQueueDepth, failed};
  StageQueue<LiveFrame> matched{kStageQueueDepth, failed};
  std::mutex error_mutex;
  std::exception_ptr first_error;
  MatchStats stats;

  // Records a stage's failure and unblocks the others.
  void fail(std::exception_ptr error) {
    {
      std::lock_guard lock(error_mutex);
      if (!first_error) {
        first_error = std::move(error);
      }
    }
    failed = true;
    superpoint_runner.close_input();
    matcher_runner.close_input();
  }
};

// The first frame with 512 keypoints, or nothing when a stop was requested first.
std::optional<Reference> capture_reference(neat::Run& source_run, const std::string& source_key,
                                           neat::Model::Runner& superpoint_runner) {
  while (!stop_requested) {
    DecodedFrame frame = pull_frame(source_run, source_key);
    auto features =
        superpoint_features(superpoint_runner.run(neat::TensorList{luma_tensor(frame.luma)}));
    if (features) {
      return Reference{std::move(frame.bgr), std::move(*features)};
    }
  }
  return std::nullopt;
}

// Stage 1: each newest decoded frame's luma into SuperPoint, its BGR pixels to stage 2.
void capture_frames(Stages& stages) {
  const AppConfig& cfg = stages.cfg;
  try {
    int count = 0;
    while (!stop_requested && (cfg.frames == 0 || count < cfg.frames)) {
      DecodedFrame frame = pull_frame(stages.source_run, cfg.source_key);
      if (!stages.superpoint_runner.push(neat::TensorList{luma_tensor(frame.luma)})) {
        break;
      }
      if (!stages.captured.put(std::move(frame.bgr))) {
        break;
      }
      ++count;
    }
  } catch (...) {
    stages.captured_all = true;
    stages.superpoint_runner.close_input();
    throw;
  }
  stages.captured_all = true;
  stages.superpoint_runner.close_input();
}

// Stage 2: each frame's SuperPoint features into the matcher, beside the reference's.
void match_frames(Stages& stages) {
  try {
    while (true) {
      std::optional<Features> features;
      {
        const neat::TensorList outputs = stages.superpoint_runner.pull(kModelTimeoutMs).tensors;
        if (outputs.empty()) {
          // An empty pull is the end of input once capture has finished, and a stall before.
          if (stages.captured_all || stages.failed) {
            break;
          }
          throw std::runtime_error("SuperPoint returned no output for " +
                                   std::to_string(kModelTimeoutMs) + " ms");
        }
        features = superpoint_features(outputs);
      }
      auto bgr = stages.captured.get();
      if (!bgr) {
        break;
      }
      if (features) {
        neat::TensorList inputs = stages.reference_inputs;
        const neat::TensorList live_inputs = stages.matcher.image_inputs(*features);
        inputs.insert(inputs.end(), live_inputs.begin(), live_inputs.end());
        if (!stages.matcher_runner.push(inputs)) {
          break;
        }
      }
      if (!stages.matched.put(LiveFrame{std::move(*bgr), std::move(features)})) {
        break;
      }
    }
  } catch (...) {
    stages.matcher_runner.close_input();
    stages.matched.close();
    throw;
  }
  stages.matcher_runner.close_input();
  stages.matched.close();
}

// One frame's matches drawn beside the reference; runs on a render worker.
Rendered render(const Stages& stages, const LiveFrame& frame, const std::vector<float>& scores,
                double fps, bool sample_inliers) {
  const Matcher& matcher = stages.matcher;
  const cv::Mat& reference_bgr = stages.reference.bgr;
  if (!frame.features) {
    const std::string status =
        matcher.name + ": fewer than " + std::to_string(kKeypoints) + " keypoints, not matched";
    return {compose_frame(reference_bgr, frame.bgr, {}, {}, status), std::nullopt, std::nullopt};
  }
  const Matches matches = matcher.matches(scores);
  const MatchLines lines = match_lines(matches, stages.reference.features, *frame.features,
                                       stages.point_scale, reference_bgr.cols);
  const auto colors = match_colors(lines.scores, static_cast<float>(stages.cfg.match_threshold));
  const std::string status =
      cv::format("%s: %zu matches, %.1f fps", matcher.name.c_str(), lines.endpoints.size(), fps);
  const int count = static_cast<int>(lines.endpoints.size());
  return {compose_frame(reference_bgr, frame.bgr, lines, colors, status), count,
          sample_inliers ? std::optional(inlier_ratio(lines.endpoints)) : std::nullopt};
}

// Sends one composed frame to Insight and counts it.
void send(Stages& stages, const Rendered& rendered) {
  if (!stages.insight_run.push(neat::TensorList{bgr_tensor(rendered.canvas)})) {
    throw std::runtime_error("Insight video push failed");
  }
  MatchStats& stats = stages.stats;
  ++stats.frames;
  stats.sent_at.push_back(now_seconds());
  if (rendered.matches) {
    ++stats.matched;
    stats.matches += *rendered.matches;
  }
  if (rendered.inlier_ratio) {
    stats.inlier_ratios.push_back(*rendered.inlier_ratio);
  }
  const AppConfig& cfg = stages.cfg;
  if (!cfg.save_dir.empty() && cfg.save_every && stats.frames % cfg.save_every == 0) {
    cv::imwrite((fs::path(cfg.save_dir) / cv::format("frame_%06d.jpg", stats.frames)).string(),
                rendered.canvas);
  }
}

// Stage 3: each frame's matches drawn beside the reference and sent to Insight, in order. Frames
// are matched and drawn on worker threads; this thread pulls the matcher and sends.
void publish_frames(Stages& stages) {
  std::deque<std::future<Rendered>> rendering;
  int received = 0;
  while (auto frame = stages.matched.get()) {
    std::vector<float> scores;
    if (frame->features) {
      const neat::TensorList outputs = stages.matcher_runner.pull(kModelTimeoutMs).tensors;
      if (outputs.empty()) {
        throw std::runtime_error(stages.matcher.name + " returned no output for " +
                                 std::to_string(kModelTimeoutMs) + " ms");
      }
      // A copy (about 1 MB), so that the runner reuses its output while workers draw.
      scores = sima_examples::tensor_to_floats(outputs.front());
    }
    const auto& sent_at = stages.stats.sent_at;
    const double fps = rate(std::span(sent_at).last(std::min(kFpsWindow, sent_at.size())));
    const bool sample_inliers = received % kInlierSampleInterval == 0;
    rendering.push_back(std::async(std::launch::async, render, std::cref(stages), std::move(*frame),
                                   std::move(scores), fps, sample_inliers));
    ++received;
    if (rendering.size() == kRenderWorkers) {
      send(stages, rendering.front().get());
      rendering.pop_front();
    }
  }
  while (!rendering.empty()) {
    send(stages, rendering.front().get());
    rendering.pop_front();
  }
}

int run(const AppConfig& cfg) {
  const Matcher matcher = select_matcher(cfg);
  const int fps = stream_fps(cfg);
  SourceRun source = build_source_run(cfg, fps);

  const DecodedFrame first = pull_frame(source.run, cfg.source_key);
  const int width = first.luma.cols;
  const int height = first.luma.rows;
  neat::Model superpoint(cfg.superpoint_path, superpoint_options(width, height));
  const cv::Size model_size = model_input_size(superpoint);
  if (model_size != kModelInputSize) {
    throw std::runtime_error("SuperPoint input is " + std::to_string(model_size.width) + "x" +
                             std::to_string(model_size.height) + "; both matchers need " +
                             std::to_string(kModelInputSize.width) + "x" +
                             std::to_string(kModelInputSize.height));
  }
  const cv::Point2f point_scale(static_cast<float>(width) / static_cast<float>(model_size.width),
                                static_cast<float>(height) / static_cast<float>(model_size.height));
  auto superpoint_runner = superpoint.build(neat::TensorList{luma_tensor(first.luma)},
                                            neat::Model::RouteOptions{}, pipelined_run_options());

  const auto reference = capture_reference(source.run, cfg.source_key, superpoint_runner);
  if (!reference) {
    return 0;
  }
  const neat::TensorList reference_inputs = matcher.image_inputs(reference->features);
  neat::TensorList seed_inputs = reference_inputs;
  seed_inputs.insert(seed_inputs.end(), reference_inputs.begin(), reference_inputs.end());
  neat::Model matcher_model(cfg.matcher_path, matcher_options());
  auto matcher_runner =
      matcher_model.build(seed_inputs, neat::Model::RouteOptions{}, pipelined_run_options());
  InsightRun insight = build_insight_run(cfg, 2 * width, height, fps);
  std::cout << "source=" << cfg.source_type << " stream=" << width << "x" << height << "@" << fps
            << " matcher=" << matcher.name << " insight=" << cfg.insight_host << ":"
            << insight.video_port << std::endl;

  Stages stages{cfg,         matcher,    source.run,       superpoint_runner, matcher_runner,
                insight.run, *reference, reference_inputs, point_scale};
  const auto guarded = [&stages](void (*stage)(Stages&)) {
    return [&stages, stage] {
      try {
        stage(stages);
      } catch (...) {
        stages.fail(std::current_exception());
      }
    };
  };
  std::thread match_thread(guarded(match_frames));
  std::thread publish_thread(guarded(publish_frames));
  guarded(capture_frames)();
  match_thread.join();
  publish_thread.join();
  if (stages.first_error) {
    std::rethrow_exception(stages.first_error);
  }
  std::cout << stages.stats.summary() << " video_sender=" << cfg.insight_host << ":"
            << insight.video_port << std::endl;
  // The runs close in reverse order of construction: Insight, the matcher, SuperPoint, the source.
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  AppConfig cfg;
  try {
    cfg = load_app_config(parse_config_path(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 2;
  }
  for (const auto& path : {cfg.superpoint_path, cfg.matcher_path}) {
    if (!fs::is_regular_file(path)) {
      std::cerr << "Error: model does not exist: " << path << "\n";
      return 2;
    }
  }

  try {
    if (!cfg.save_dir.empty()) {
      fs::create_directories(cfg.save_dir);
    }
    // The app runs frames in parallel on its own threads, so OpenCV uses one thread. Its idle
    // workers otherwise spin at full CPU between calls and take every core of the board.
    cv::setNumThreads(1);
    for (const auto sig : {SIGINT, SIGTERM}) {
      std::signal(sig, request_stop);
    }
    return run(cfg);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
