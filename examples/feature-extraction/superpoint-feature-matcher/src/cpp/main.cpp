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
// SuperPoint's keypoint score threshold. LightGlue's extractor uses 0.0005, which on camera frames
// leaves over 16,000 candidates and a single-threaded decode of about 70 ms per frame. At 0.005 the
// decode takes about 14 ms and returns the same best 512 keypoints; only a frame with too little
// texture to reach 512 is skipped.
constexpr float kScoreThreshold = 0.005F;
constexpr int kSourceTimeoutMs = 20000;
constexpr int kModelTimeoutMs = 10000;
// Frames each model runner may hold, so that its stages overlap: on the board, SuperPoint
// alone runs at 11.6 ms per frame one at a time and 7.4 ms with two frames in flight.
constexpr int kRunnerQueueDepth = 2;
// Frames waiting between two stages; a full queue holds back the stage before it.
constexpr std::size_t kStageQueueDepth = 2;
// Frames matched and drawn at once, each on its own worker. On the board SuperGlue's Sinkhorn
// iterations take tens of milliseconds of CPU per frame, so a 30 fps stream needs several frames
// in parallel.
constexpr std::size_t kRenderWorkers = 3;
// Line colors, from red at the match threshold to green at score 1. Each color's lines are drawn
// in one call, without anti-aliasing: on the board this takes about 2.5 ms per 2560x720 frame
// with 320 matches, against about 17 ms for one anti-aliased call per line.
constexpr int kColorSteps = 16;
// Each match also gets a dot at both ends, drawn the same way.
constexpr int kDotDiameter = 5;
// The encoder reads NV12 rows at a pitch rounded up to 64 bytes and planes of rows rounded up to 8.
constexpr int kEncoderPitchAlignment = 64;
constexpr int kEncoderRowAlignment = 8;
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

// The reference frame, in NV12, and its features.
struct Reference {
  cv::Mat nv12;
  Features features;
};

// A frame, in NV12, on its way to the publisher; `features` is empty when it has fewer than 512.
struct LiveFrame {
  cv::Mat nv12;
  std::optional<Features> features;
};

// A color's luma and its (U, V) chroma.
struct Nv12Color {
  uint8_t y = 0;
  cv::Scalar uv;
};

// One composed NV12 frame and what it adds to the summary.
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
double now_seconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// What the publisher sent, for the summary line and the displayed rate. Holds a fixed amount of
// state, so a continuous run does not grow it.
struct MatchStats {
  int frames = 0;
  int matched_frames = 0;
  long long total_matches = 0;
  double inlier_sum = 0.0;
  int inlier_samples = 0;
  double first_sent_s = 0.0;
  double last_sent_s = 0.0;
  std::deque<double> recent_sent_s;

  void record(std::optional<int> match_count, std::optional<double> inlier_ratio) {
    const double now = now_seconds();
    if (frames == 0) {
      first_sent_s = now;
    }
    ++frames;
    last_sent_s = now;
    recent_sent_s.push_back(now);
    if (recent_sent_s.size() > kFpsWindow) {
      recent_sent_s.pop_front();
    }
    if (match_count) {
      ++matched_frames;
      total_matches += *match_count;
    }
    if (inlier_ratio) {
      inlier_sum += *inlier_ratio;
      ++inlier_samples;
    }
  }

  double recent_fps() const {
    const std::size_t n = recent_sent_s.size();
    return n > 1 ? static_cast<double>(n - 1) / (recent_sent_s.back() - recent_sent_s.front())
                 : 0.0;
  }

  std::string summary() const {
    const double average =
        matched_frames ? static_cast<double>(total_matches) / matched_frames : 0.0;
    const double inliers = inlier_samples ? inlier_sum / inlier_samples : 0.0;
    const double elapsed = last_sent_s - first_sent_s;
    const double fps = elapsed > 0.0 ? (frames - 1) / elapsed : 0.0;
    return cv::format("frames=%d matched=%d skipped=%d average_matches=%.1f inlier_ratio=%.2f "
                      "output_fps=%.1f",
                      frames, matched_frames, frames - matched_frames, average, inliers, fps);
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
// The LightGlueV1 decode (NMS radius 4, 4-pixel border) is the one both matchers' keypoints were
// extracted with, keeping the best 512 keypoints above kScoreThreshold.
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
  options.score_threshold = kScoreThreshold;
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

// A decoded frame as one NV12 matrix: its luma rows, then its interleaved chroma rows. Decoders
// pad rows and planes, so each plane is read at its own offset and stride. MJPEG with 4:2:2
// chroma, as most webcams send, decodes to NV16; every other chroma row of it makes NV12.
cv::Mat frame_nv12(const neat::Tensor& tensor) {
  const bool nv16 = tensor.image_format() == neat::ImageSpec::PixelFormat::NV16;
  if (!nv16 && !tensor.is_nv12()) {
    throw std::runtime_error("unsupported decoded frame format: " + tensor.debug_string());
  }
  const int width = tensor.width();
  const int height = tensor.height();
  const neat::Plane* luma = tensor.try_plane(neat::PlaneRole::Y);
  const neat::Plane* chroma = tensor.try_plane(neat::PlaneRole::UV);
  sima_examples::require(luma && chroma && !luma->strides_bytes.empty() &&
                             !chroma->strides_bytes.empty() && width > 0 && height > 0,
                         "decoded frame without Y and UV planes: " + tensor.debug_string());
  // The payload is the whole mapped buffer, so plane offsets index into it.
  std::vector<uint8_t> payload = tensor.copy_payload_bytes();
  const int chroma_rows = nv16 ? height : height / 2;
  const auto plane = [&](const neat::Plane& layout, int rows) {
    const auto stride = static_cast<std::size_t>(layout.strides_bytes[0]);
    const std::size_t end = static_cast<std::size_t>(layout.byte_offset) +
                            stride * static_cast<std::size_t>(rows - 1) +
                            static_cast<std::size_t>(width);
    sima_examples::require(end <= payload.size(), "decoded frame smaller than its planes");
    return cv::Mat(rows, width, CV_8UC1, payload.data() + layout.byte_offset, stride);
  };
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1);
  plane(*luma, height).copyTo(nv12.rowRange(0, height));
  const cv::Mat chroma_plane = plane(*chroma, chroma_rows);
  for (int row = 0; row < height / 2; ++row) {
    chroma_plane.row(nv16 ? 2 * row : row).copyTo(nv12.row(height + row));
  }
  return nv12;
}

int nv12_height(const cv::Mat& nv12) {
  return nv12.rows * 2 / 3;
}

cv::Mat luma_plane(const cv::Mat& nv12) {
  return nv12.rowRange(0, nv12_height(nv12));
}

// The newest decoded frame, in NV12. The decoder's buffer is released on return: the frame is a
// CPU copy.
cv::Mat pull_frame(neat::Run& source_run, const std::string& source_key) {
  neat::Sample sample;
  neat::PullError error;
  const auto status = source_run.pull("frame", kSourceTimeoutMs, sample, &error);
  if (!sima_examples::pull_status_has_sample(status, "frame", error, source_run.last_error())) {
    throw std::runtime_error("no frame arrived from " + source_key + " for " +
                             std::to_string(kSourceTimeoutMs) + " ms");
  }
  return frame_nv12(frame_tensor(sample));
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

neat::Plane image_plane(neat::PlaneRole role, int64_t offset, int rows, int width, int pitch) {
  neat::Plane plane;
  plane.role = role;
  plane.byte_offset = offset;
  plane.shape = {rows, width};
  plane.strides_bytes = {pitch, 1};
  return plane;
}

int align_up(int value, int alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// An NV12 matrix as an EV74 tensor in the encoder's layout, padded when its width or height is
// not aligned. `Tensor::from_cv_mat` does not describe NV12 planes, so they are set here.
neat::Tensor encoder_tensor(const cv::Mat& nv12) {
  const int height = nv12_height(nv12);
  const int width = nv12.cols;
  const int pitch = align_up(width, kEncoderPitchAlignment);
  const int rows = align_up(height, kEncoderRowAlignment);
  cv::Mat aligned = nv12;
  if (pitch != width || rows != height) {
    aligned = cv::Mat::zeros(rows * 3 / 2, pitch, CV_8UC1);
    luma_plane(nv12).copyTo(aligned(cv::Rect(0, 0, width, height)));
    nv12.rowRange(height, nv12.rows).copyTo(aligned(cv::Rect(0, rows, width, height / 2)));
  }
  neat::Tensor tensor =
      neat::Tensor::from_cv_mat_view(aligned, neat::ImageSpec::PixelFormat::GRAY8);
  tensor.semantic.image = neat::ImageSpec{neat::ImageSpec::PixelFormat::NV12, ""};
  tensor.shape = {height, width};
  tensor.strides_bytes = {pitch, 1};
  tensor.planes = {
      image_plane(neat::PlaneRole::Y, 0, height, width, pitch),
      image_plane(neat::PlaneRole::UV, static_cast<int64_t>(pitch) * rows, height / 2, width,
                  pitch),
  };
  return tensor.cvu();
}

struct InsightRun {
  neat::Graph graph{"insight"};
  neat::Run run;
  int video_port = 0;
};

// Composed NV12 frames, encoded to H.264 for Insight.
InsightRun build_insight_run(const AppConfig& cfg, int width, int height, int fps) {
  neat::InputOptions input_options;
  input_options.payload_type = neat::PayloadType::Image;
  input_options.format = neat::FormatTag::NV12;
  input_options.width = width;
  input_options.height = height;
  input_options.fps_n = fps;
  input_options.fps_d = 1;
  input_options.memory_policy = neat::InputMemoryPolicy::Ev74;

  auto sender_options = groups::VideoSenderOptions::FromRaw({.fps = fps});
  sender_options.host = cfg.insight_host;
  sender_options.video_port_base = cfg.video_port;
  sender_options.encoder.bitrate_kbps = cfg.bitrate_kbps;
  sender_options.encoder.level = encoder_level(width, height);

  InsightRun insight;
  insight.graph.add(neat::nodes::Input(input_options));
  insight.graph.add(groups::VideoSender(sender_options));
  const cv::Mat seed = cv::Mat::zeros(height * 3 / 2, width, CV_8UC1);
  insight.run = insight.graph.build(neat::TensorList{encoder_tensor(seed)});
  insight.video_port = sender_options.video_port();
  return insight;
}

// Each score's color step, 0 (red) at the match threshold to kColorSteps - 1 (green) at 1.
std::vector<int> color_steps(const std::vector<float>& scores, float threshold) {
  std::vector<int> steps;
  steps.reserve(scores.size());
  for (const float score : scores) {
    const float strength = std::clamp((score - threshold) / (1.0F - threshold), 0.0F, 1.0F);
    steps.push_back(static_cast<int>(std::lrint(strength * (kColorSteps - 1))));
  }
  return steps;
}

// A BGR color's luma and chroma, as OpenCV converts BGR to YUV 4:2:0.
Nv12Color nv12_color(const cv::Scalar& bgr) {
  const cv::Mat pixels(2, 2, CV_8UC3, bgr);
  cv::Mat yuv;
  cv::cvtColor(pixels, yuv, cv::COLOR_BGR2YUV_I420);
  return {yuv.at<uint8_t>(0, 0), cv::Scalar(yuv.at<uint8_t>(2, 0), yuv.at<uint8_t>(2, 1))};
}

// Each color step's color, from red at step 0 to green at kColorSteps - 1.
const std::array<Nv12Color, kColorSteps>& step_colors() {
  static const std::array<Nv12Color, kColorSteps> colors = [] {
    std::array<Nv12Color, kColorSteps> steps;
    for (int step = 0; step < kColorSteps; ++step) {
      const double strength = static_cast<double>(step) / (kColorSteps - 1);
      steps[step] =
          nv12_color({0, std::nearbyint(255 * strength), std::nearbyint(255 * (1 - strength))});
    }
    return steps;
  }();
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

// White text on a darkened box, readable on any frame. The box's chroma moves toward neutral as
// its luma darkens.
void draw_label(cv::Mat& luma, cv::Mat& chroma, const std::string& text, cv::Point origin,
                double scale) {
  static const uint8_t white = nv12_color(cv::Scalar(255, 255, 255)).y;
  const int thickness = std::max(1, static_cast<int>(std::nearbyint(scale * 1.5)));
  int baseline = 0;
  const cv::Size size =
      cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
  const int pad = std::max(4, size.height / 3);
  const cv::Rect box = cv::Rect(cv::Point(origin.x - pad, origin.y - size.height - pad),
                                cv::Point(origin.x + size.width + pad, origin.y + baseline + pad)) &
                       cv::Rect(0, 0, luma.cols, luma.rows);
  cv::Mat luma_box = luma(box);
  luma_box /= 3;
  const cv::Rect chroma_rect(box.tl() / 2, box.br() / 2);
  cv::Mat chroma_box = chroma(chroma_rect & cv::Rect(0, 0, chroma.cols, chroma.rows));
  chroma_box.convertTo(chroma_box, -1, 1.0 / 3, 128.0 * 2 / 3);
  cv::putText(luma, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(white), thickness,
              cv::LINE_AA);
}

// The reference on the left and the live frame on the right, in NV12: one line per match with a
// dot at each end. Each is drawn on the luma rows and, at half the
// coordinates, on the chroma rows. The strongest color steps are drawn last, on top.
cv::Mat compose_frame(const cv::Mat& reference_nv12, const cv::Mat& live_nv12,
                      const MatchLines& lines, const std::vector<int>& steps,
                      const std::string& status) {
  const int height = nv12_height(reference_nv12);
  const int width = reference_nv12.cols;
  cv::Mat canvas;
  cv::hconcat(reference_nv12, live_nv12, canvas);
  cv::Mat luma = luma_plane(canvas);
  cv::Mat chroma(height / 2, width, CV_8UC2, canvas.ptr(height));
  std::vector<std::vector<std::vector<cv::Point>>> segments(kColorSteps);
  for (std::size_t i = 0; i < lines.endpoints.size(); ++i) {
    const cv::Vec4i& line = lines.endpoints[i];
    segments[steps[i]].push_back({{line[0], line[1]}, {line[2], line[3]}});
  }
  for (int step = 0; step < kColorSteps; ++step) {
    if (segments[step].empty()) {
      continue;
    }
    // A zero-length segment drawn with a thick pen is a dot.
    std::vector<std::vector<cv::Point>> dots;
    for (const auto& segment : segments[step]) {
      for (const cv::Point& point : segment) {
        dots.push_back({point, point});
      }
    }
    const Nv12Color& color = step_colors()[step];
    cv::polylines(luma, segments[step], false, cv::Scalar(color.y), 1, cv::LINE_8);
    cv::polylines(luma, dots, false, cv::Scalar(color.y), kDotDiameter, cv::LINE_8);
    for (auto* shapes : {&segments[step], &dots}) {
      for (auto& shape : *shapes) {
        for (cv::Point& point : shape) {
          point /= 2;
        }
      }
    }
    cv::polylines(chroma, segments[step], false, color.uv, 1, cv::LINE_8);
    cv::polylines(chroma, dots, false, color.uv, kDotDiameter / 2 + 1, cv::LINE_8);
  }
  const double scale = height / 1200.0;
  const int margin = static_cast<int>(std::nearbyint(30 * scale));
  const int line_height = static_cast<int>(std::nearbyint(56 * scale));
  draw_label(luma, chroma, "Reference", {margin, margin + line_height / 2}, scale);
  draw_label(luma, chroma, "Live", {width + margin, margin + line_height / 2}, scale);
  draw_label(luma, chroma, status, {width + margin, margin + line_height / 2 + line_height}, scale);
  draw_label(luma, chroma, "Line color: match score, red low, green high",
             {margin, height - margin}, scale);
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

  bool empty() {
    std::lock_guard lock(mutex_);
    return items_.empty();
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
    cv::Mat frame = pull_frame(source_run, source_key);
    auto features = superpoint_features(
        superpoint_runner.run(neat::TensorList{luma_tensor(luma_plane(frame))}));
    if (features) {
      return Reference{std::move(frame), std::move(*features)};
    }
  }
  return std::nullopt;
}

// Stage 1: each newest decoded frame's luma into SuperPoint, the whole NV12 frame to stage 2.
void capture_frames(Stages& stages) {
  const AppConfig& cfg = stages.cfg;
  try {
    int count = 0;
    while (!stop_requested && (cfg.frames == 0 || count < cfg.frames)) {
      cv::Mat frame = pull_frame(stages.source_run, cfg.source_key);
      if (!stages.superpoint_runner.push(neat::TensorList{luma_tensor(luma_plane(frame))})) {
        throw std::runtime_error("SuperPoint rejected a frame");
      }
      if (!stages.captured.put(std::move(frame))) {
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
          // An empty pull ends the run only once capture has finished and every captured frame
          // has been matched; otherwise SuperPoint stalled.
          if (stages.failed || (stages.captured_all && stages.captured.empty())) {
            break;
          }
          throw std::runtime_error("SuperPoint returned no output for " +
                                   std::to_string(kModelTimeoutMs) + " ms");
        }
        features = superpoint_features(outputs);
      }
      auto frame = stages.captured.get();
      if (!frame) {
        break;
      }
      if (features) {
        neat::TensorList inputs = stages.reference_inputs;
        const neat::TensorList live_inputs = stages.matcher.image_inputs(*features);
        inputs.insert(inputs.end(), live_inputs.begin(), live_inputs.end());
        if (!stages.matcher_runner.push(inputs)) {
          throw std::runtime_error(stages.matcher.name + " rejected a frame");
        }
      }
      if (!stages.matched.put(LiveFrame{std::move(*frame), std::move(features)})) {
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
  const cv::Mat& reference = stages.reference.nv12;
  if (!frame.features) {
    const std::string status =
        matcher.name + ": fewer than " + std::to_string(kKeypoints) + " keypoints, not matched";
    return {compose_frame(reference, frame.nv12, {}, {}, status), std::nullopt, std::nullopt};
  }
  const Matches matches = matcher.matches(scores);
  const MatchLines lines = match_lines(matches, stages.reference.features, *frame.features,
                                       stages.point_scale, reference.cols);
  const auto steps = color_steps(lines.scores, static_cast<float>(stages.cfg.match_threshold));
  const std::string status =
      cv::format("%s: %zu matches, %.1f fps", matcher.name.c_str(), lines.endpoints.size(), fps);
  const int count = static_cast<int>(lines.endpoints.size());
  return {compose_frame(reference, frame.nv12, lines, steps, status), count,
          sample_inliers ? std::optional(inlier_ratio(lines.endpoints)) : std::nullopt};
}

// Sends one composed frame to Insight and counts it.
void send(Stages& stages, const Rendered& rendered) {
  if (!stages.insight_run.push(neat::TensorList{encoder_tensor(rendered.canvas)})) {
    throw std::runtime_error("Insight video push failed");
  }
  MatchStats& stats = stages.stats;
  stats.record(rendered.matches, rendered.inlier_ratio);
  const AppConfig& cfg = stages.cfg;
  if (!cfg.save_dir.empty() && cfg.save_every && stats.frames % cfg.save_every == 0) {
    const fs::path path = fs::path(cfg.save_dir) / cv::format("frame_%06d.jpg", stats.frames);
    cv::Mat bgr;
    cv::cvtColor(rendered.canvas, bgr, cv::COLOR_YUV2BGR_NV12);
    if (!cv::imwrite(path.string(), bgr)) {
      throw std::runtime_error("could not save " + path.string());
    }
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
    const double fps = stages.stats.recent_fps();
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

  const cv::Mat first = pull_frame(source.run, cfg.source_key);
  const int width = first.cols;
  const int height = nv12_height(first);
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
  auto superpoint_runner = superpoint.build(neat::TensorList{luma_tensor(luma_plane(first))},
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
