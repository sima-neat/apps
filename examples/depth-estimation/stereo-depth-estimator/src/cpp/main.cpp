/**
 * @example stereo-depth-estimator.cpp
 * Show how near each point of a scene is, from two cameras, with LightStereo-S.
 *
 * A left and a right camera feed one source graph, which forwards each camera's video to its own
 * Insight channel. Each loop takes the newest frame of both, joins them side by side for the
 * model, and sends the colored disparity to a third channel. Disparity is how far a point shifts
 * between the two views: the larger it is, the nearer the point.
 *
 * Usage: stereo-depth-estimator [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/pull_status.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
namespace groups = simaai::neat::nodes::groups;

namespace {

const std::vector<std::string> kSourceTypes{"rtsp", "usb", "mipi"};
const std::vector<std::string> kRtspCodecs{"h264", "h265", "mjpeg"};
// The source graph's two outputs, in the order the model takes the views. Insight shows them on
// the first two channels and the depth map on the third.
const std::array<std::string, 2> kViews{"left", "right"};
constexpr int kDepthChannel = static_cast<int>(kViews.size());
constexpr int kSourceTimeoutMs = 20000;
constexpr int kModelTimeoutMs = 10000;

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int) {
  stop_requested = 1;
}

struct AppConfig {
  std::string model_path;
  std::string source_type;
  // The left and right camera: RTSP URLs, USB IDs or MIPI names.
  std::array<std::string, 2> cameras;
  std::string codec;
  bool tcp = true;
  int latency_ms = 0;
  int width = 0;
  int height = 0;
  int fps = 0;
  int frames = 0;
  double max_disparity = 0.0;
  std::string save_dir;
  int save_every = 0;
  std::string insight_host;
  int video_port = 0;
  int bitrate_kbps = 0;
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
  const std::string source = "source." + cfg.source_type;
  const std::string camera_key = cfg.source_type == "rtsp" ? "url" : "camera";
  const std::string left = source + ".left_" + camera_key;
  const std::string right = source + ".right_" + camera_key;
  require(!cfg.model_path.empty(), "model.path must be set");
  require(!cfg.cameras[0].empty() && !cfg.cameras[1].empty(),
          left + " and " + right + " must be set");
  require(cfg.cameras[0] != cfg.cameras[1], left + " and " + right + " must differ");
  require(cfg.source_type == "rtsp" || std::min({cfg.width, cfg.height, cfg.fps}) > 0,
          source + ".width, " + source + ".height and " + source + ".fps must be > 0");
  require(cfg.latency_ms >= 0, "source.rtsp.latency_ms must be >= 0");
  require(cfg.fps >= 0, source + ".fps must be >= 0");
  require(cfg.frames >= 0, "inference.frames must be >= 0");
  require(cfg.max_disparity > 0.0, "output.max_disparity must be > 0");
  require(cfg.save_every >= 0, "output.save_every must be >= 0");
  require(!cfg.insight_host.empty(), "output.insight.host must be set");
  require(cfg.video_port > 0 && cfg.video_port <= 65535 - kDepthChannel,
          "output.insight.video_port must be in [1, " + std::to_string(65535 - kDepthChannel) +
              "]");
  require(cfg.bitrate_kbps > 0, "output.insight.bitrate_kbps must be > 0");
}

AppConfig load_app_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);

  // Only the section that source.type names is read; the others may hold placeholders.
  AppConfig cfg;
  cfg.source_type = choice(raw.string_or("source.type", "rtsp"), kSourceTypes, "source.type");
  const std::string source = "source." + cfg.source_type + ".";
  const std::string camera_key = cfg.source_type == "rtsp" ? "url" : "camera";
  cfg.model_path = raw.string_or("model.path", "");
  cfg.cameras = {raw.string_or(source + "left_" + camera_key, ""),
                 raw.string_or(source + "right_" + camera_key, "")};
  cfg.codec = choice(raw.string_or(source + "codec", "h264"), kRtspCodecs, "source.rtsp.codec");
  cfg.tcp = raw.bool_or(source + "tcp", true);
  cfg.latency_ms = raw.int_or(source + "latency_ms", 100);
  cfg.width = raw.int_or(source + "width", 0);
  cfg.height = raw.int_or(source + "height", 0);
  cfg.fps = raw.int_or(source + "fps", 0);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.max_disparity = raw.double_or("output.max_disparity", 96.0);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.bitrate_kbps = raw.int_or("output.insight.bitrate_kbps", 8000);
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

// The configured frame rate, or, when it is 0, the one the left RTSP stream reports. The encoder
// needs it before the first frame.
int stream_fps(const AppConfig& cfg) {
  if (cfg.fps > 0) {
    return cfg.fps;
  }
  cv::VideoCapture capture(cfg.cameras[0]);
  const int fps =
      capture.isOpened() ? static_cast<int>(std::nearbyint(capture.get(cv::CAP_PROP_FPS))) : 0;
  capture.release();
  if (fps <= 0) {
    throw std::runtime_error(
        "could not read the frame rate of source.rtsp.left_url; set source.rtsp.fps");
  }
  return fps;
}

groups::RtspCodec rtsp_codec(const AppConfig& cfg) {
  return cfg.codec == "h264"   ? groups::RtspCodec::H264
         : cfg.codec == "h265" ? groups::RtspCodec::H265
                               : groups::RtspCodec::MJPEG;
}

// Receives a camera's encoded stream without decoding it.
groups::RtspEncodedInputOptions rtsp_source_options(const AppConfig& cfg, const std::string& url,
                                                    int fps) {
  groups::RtspEncodedInputOptions options;
  options.url = url;
  options.tcp = cfg.tcp;
  options.latency_ms = cfg.latency_ms;
  options.source_fps = fps;
  options.codec = rtsp_codec(cfg);
  options.auto_caps_from_stream = cfg.codec != "mjpeg";
  return options;
}

neat::SimaDecodeOptions rtsp_decode_options(const AppConfig& cfg, int fps) {
  neat::SimaDecodeOptions options;
  options.type = cfg.codec == "h264"   ? neat::SimaDecodeType::H264
                 : cfg.codec == "h265" ? neat::SimaDecodeType::H265
                                       : neat::SimaDecodeType::MJPEG;
  options.raw_output = true;
  options.dec_fps = fps;
  return options;
}

neat::UsbCameraInputOptions usb_camera_options(const AppConfig& cfg, const std::string& camera_id) {
  neat::UsbCameraInputOptions options;
  options.camera_name = camera_id;
  options.width = static_cast<std::uint32_t>(cfg.width);
  options.height = static_cast<std::uint32_t>(cfg.height);
  options.framerate_num = static_cast<std::uint32_t>(cfg.fps);
  return options;
}

neat::SimaDecodeOptions usb_decode_options(const AppConfig& cfg, neat::FormatTag decoded_format) {
  neat::SimaDecodeOptions options;
  options.type = neat::SimaDecodeType::MJPEG;
  options.raw_output = true;
  options.out_format = decoded_format;
  options.dec_width = cfg.width;
  options.dec_height = cfg.height;
  options.dec_fps = cfg.fps;
  return options;
}

neat::CameraInputOptions mipi_camera_options(const AppConfig& cfg, const std::string& camera_name) {
  neat::CameraInputOptions options;
  options.camera_name = camera_name;
  options.width = static_cast<std::uint32_t>(cfg.width);
  options.height = static_cast<std::uint32_t>(cfg.height);
  options.framerate_num = static_cast<std::uint32_t>(cfg.fps);
  options.format = "NV12";
  return options;
}

// One camera as two graphs: the camera's video as it arrives, and the decoder that turns it into
// frames. A MIPI camera sends raw frames, so its decoder graph is empty.
struct CameraGraphs {
  neat::Graph camera;
  neat::Graph decoder;
};

CameraGraphs camera_graphs(const AppConfig& cfg, const std::string& view,
                           const std::string& camera_name, int fps) {
  CameraGraphs graphs{neat::Graph(view + "_camera"), neat::Graph(view + "_decoder")};
  if (cfg.source_type == "rtsp") {
    graphs.camera.add(groups::RtspEncodedInput(rtsp_source_options(cfg, camera_name, fps)));
    graphs.decoder.add(neat::nodes::SimaDecode(rtsp_decode_options(cfg, fps)));
  } else if (cfg.source_type == "usb") {
    const auto usb_camera = usb_camera_options(cfg, camera_name);
    // Checks the camera and its mode once; most webcams send 4:2:2 JPEG, which decodes to NV16.
    const auto info = neat::usb_camera_info(usb_camera);
    graphs.camera.add(neat::nodes::UsbCameraInput(usb_camera, info));
    graphs.decoder.add(neat::nodes::SimaDecode(usb_decode_options(cfg, info.decoded_format)));
  } else {
    graphs.camera.add(neat::nodes::CameraInput(mipi_camera_options(cfg, camera_name)));
  }
  return graphs;
}

// H.264 encoding of raw frames for Insight.
groups::VideoSenderOptions h264_encoding_sender_options(const AppConfig& cfg, int fps) {
  auto options = groups::VideoSenderOptions::FromRaw({.fps = fps});
  options.encoder.bitrate_kbps = cfg.bitrate_kbps;
  return options;
}

// Sends a camera's video to its Insight channel. RTSP and USB cameras send encoded video, which
// is forwarded as it is; raw MIPI frames are encoded first.
groups::VideoSenderOptions camera_sender_options(const AppConfig& cfg, int channel, int fps) {
  auto options =
      cfg.source_type == "rtsp"  ? groups::VideoSenderOptions::Passthrough(rtsp_codec(cfg))
      : cfg.source_type == "usb" ? groups::VideoSenderOptions::Passthrough(groups::RtspCodec::MJPEG)
                                 : h264_encoding_sender_options(cfg, fps);
  options.host = cfg.insight_host;
  options.video_port_base = cfg.video_port;
  options.channel = channel;
  return options;
}

struct SourceRun {
  neat::Graph graph{"source"};
  neat::Run run;
};

// One graph with two branches per camera:
//
//   left camera  --+-- Insight channel 0
//                  +-- decode -- Output("left")
//   right camera --+-- Insight channel 1
//                  +-- decode -- Output("right")
//
// Each output keeps only its newest frame, so a pull never returns a frame that waited.
SourceRun build_source_run(const AppConfig& cfg, int fps) {
  SourceRun source;
  // Insight's branch keeps the newest video only, so a slow network never delays the decoder.
  neat::GraphLinkOptions newest_video;
  newest_video.policy = neat::GraphLinkPolicy::RealtimeLatestByStream;
  for (std::size_t i = 0; i < kViews.size(); ++i) {
    auto graphs = camera_graphs(cfg, kViews[i], cfg.cameras[i], fps);
    graphs.decoder.add(neat::nodes::Output(kViews[i], neat::OutputOptions::Latest()));
    neat::Graph video(kViews[i] + "_video");
    video.add(groups::VideoSender(camera_sender_options(cfg, static_cast<int>(i), fps)));
    source.graph.connect(graphs.camera, graphs.decoder);
    source.graph.connect(graphs.camera, video, newest_video);
  }

  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = 1;
  run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = neat::OutputMemory::ZeroCopy;
  source.run = source.graph.build(run_options);
  return source;
}

// A decoded NV12 or NV16 frame as BGR pixels the application owns. MJPEG with 4:2:2 chroma, as
// most webcams send, decodes to NV16.
cv::Mat frame_bgr(const neat::Tensor& tensor) {
  const bool nv16 = tensor.image_format() == neat::ImageSpec::PixelFormat::NV16;
  cv::Mat bgr;
  std::string error;
  const bool converted = nv16 ? sima_examples::nv16_to_bgr(tensor, bgr, error)
                              : sima_examples::nv12_to_bgr(tensor, bgr, error);
  sima_examples::require(converted, "unsupported decoded frame: " + error);
  return bgr;
}

struct StereoPair {
  cv::Mat left;
  cv::Mat right;
};

// The newest left and right frame. The cameras share no clock, so the two can be up to about one
// frame interval apart.
StereoPair pull_pair(neat::Run& source_run) {
  std::array<neat::Sample, 2> samples;
  std::array<cv::Mat, 2> frames;
  for (std::size_t i = 0; i < kViews.size(); ++i) {
    neat::PullError error;
    const auto status = source_run.pull(kViews[i], kSourceTimeoutMs, samples[i], &error);
    if (!sima_examples::pull_status_has_sample(status, kViews[i], error, source_run.last_error())) {
      throw std::runtime_error("timed out waiting for a frame from the " + kViews[i] + " camera");
    }
  }
  for (std::size_t i = 0; i < kViews.size(); ++i) {
    const neat::Sample* field = &samples[i];
    for (const auto& part : samples[i].fields) {
      if (part.stream_label == kViews[i]) {
        field = &part;
      }
    }
    const auto tensors = neat::tensors_from_sample(*field);
    sima_examples::require(!tensors.empty(), "the " + kViews[i] + " frame has no tensor");
    frames[i] = frame_bgr(tensors.front());
  }
  if (frames[0].size() != frames[1].size()) {
    throw std::runtime_error(
        cv::format("the cameras must send the same frame size: left is %dx%d, right is %dx%d",
                   frames[0].cols, frames[0].rows, frames[1].cols, frames[1].rows));
  }
  return {frames[0], frames[1]};
}

// LightStereo-S on a side-by-side BGR frame: Neat stretches it to the model input, so each half
// lands on one view, converts it to RGB and applies the ImageNet normalization.
neat::Model::Options model_options(int width, int height) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.input_max_width = width;
  options.preprocess.input_max_height = height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.color_convert.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  options.preprocess.preset = neat::NormalizePreset::ImageNet;
  return options;
}

// The model input: the left view beside the right view, in one frame.
neat::Tensor stereo_tensor(const StereoPair& pair) {
  cv::Mat frame;
  cv::hconcat(pair.left, pair.right, frame);
  return neat::Tensor::from_cv_mat(frame, neat::ImageSpec::PixelFormat::BGR,
                                   neat::TensorMemory::EV74);
}

// One disparity per pixel of the left view, in pixels of the model's view.
cv::Mat disparity_map(neat::Model::Runner& runner, const StereoPair& pair) {
  const auto outputs = runner.run(neat::TensorList{stereo_tensor(pair)}, kModelTimeoutMs);
  sima_examples::require(!outputs.empty() && outputs.front().dtype == neat::TensorDType::Float32,
                         "the model must return one Float32 disparity map");
  // The map arrives with batch and channel dimensions of 1.
  std::vector<int> size;
  for (const auto dimension : outputs.front().shape) {
    if (dimension != 1) {
      size.push_back(static_cast<int>(dimension));
    }
  }
  const auto bytes = outputs.front().copy_dense_bytes_tight();
  sima_examples::require(size.size() == 2 && bytes.size() == static_cast<std::size_t>(size[0]) *
                                                                 size[1] * sizeof(float),
                         "the model output is not one disparity map");
  cv::Mat disparity(size[0], size[1], CV_32FC1);
  std::memcpy(disparity.data, bytes.data(), bytes.size());
  return disparity;
}

// The disparity map as a BGR image, from black at 0 to the brightest color at max_disparity and
// above. The scale is fixed, so a color means the same nearness in every frame.
cv::Mat disparity_heatmap(const cv::Mat& disparity, double max_disparity) {
  cv::Mat levels;
  cv::Mat heatmap;
  cv::convertScaleAbs(disparity, levels, 255.0 / max_disparity);
  cv::applyColorMap(levels, heatmap, cv::COLORMAP_INFERNO);
  return heatmap;
}

neat::Tensor heatmap_tensor(const cv::Mat& heatmap) {
  return neat::Tensor::from_cv_mat(heatmap, neat::ImageSpec::PixelFormat::BGR,
                                   neat::TensorMemory::CPU);
}

struct DepthRun {
  neat::Graph graph{"depth"};
  neat::Run run;
};

// Encodes the BGR depth map to H.264 and sends it to its Insight channel.
DepthRun build_depth_run(const AppConfig& cfg, int width, int height, int fps) {
  neat::InputOptions input_options;
  input_options.payload_type = neat::PayloadType::Image;
  input_options.format = "BGR";
  input_options.width = width;
  input_options.height = height;
  input_options.depth = 3;
  input_options.fps_n = fps;
  input_options.fps_d = 1;

  auto sender_options = h264_encoding_sender_options(cfg, fps);
  sender_options.host = cfg.insight_host;
  sender_options.video_port_base = cfg.video_port;
  sender_options.channel = kDepthChannel;

  DepthRun depth;
  depth.graph.add(neat::nodes::Input(input_options));
  depth.graph.add(groups::VideoSender(sender_options));
  const cv::Mat seed = cv::Mat::zeros(height, width, CV_8UC3);
  depth.run = depth.graph.build(neat::TensorList{heatmap_tensor(seed)});
  return depth;
}

int run(const AppConfig& cfg) {
  const int fps = stream_fps(cfg);
  auto source = build_source_run(cfg, fps);
  StereoPair pair = pull_pair(source.run);
  const cv::Size view = pair.left.size();
  std::cout << "source=" << cfg.source_type << " cameras=" << view.width << "x" << view.height
            << "@" << fps << std::endl;

  neat::Model model(cfg.model_path, model_options(2 * view.width, view.height));
  auto runner = model.build(neat::TensorList{stereo_tensor(pair)});
  // The first disparity map sets the size of the depth map Insight receives.
  const cv::Size map = disparity_map(runner, pair).size();
  auto depth = build_depth_run(cfg, map.width, map.height, fps);

  int frames = 0;
  const auto started = std::chrono::steady_clock::now();
  while (!stop_requested && (cfg.frames == 0 || frames < cfg.frames)) {
    pair = pull_pair(source.run);
    const cv::Mat heatmap = disparity_heatmap(disparity_map(runner, pair), cfg.max_disparity);
    if (!depth.run.push(neat::TensorList{heatmap_tensor(heatmap)})) {
      throw std::runtime_error("could not send the depth map to Insight");
    }
    if (!cfg.save_dir.empty() && cfg.save_every > 0 && frames % cfg.save_every == 0) {
      const fs::path path = fs::path(cfg.save_dir) / cv::format("depth_%06d.jpg", frames);
      sima_examples::require(cv::imwrite(path.string(), heatmap),
                             "could not save " + path.string());
    }
    ++frames;
  }

  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
  std::cout << "frames=" << frames
            << cv::format(" depth_fps=%.1f", frames ? frames / elapsed.count() : 0.0)
            << " insight=" << cfg.insight_host << ":" << cfg.video_port << "-"
            << cfg.video_port + kDepthChannel << std::endl;
  // The runs close in reverse order of construction: the depth channel, the model, the source.
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
  if (!fs::is_regular_file(cfg.model_path)) {
    std::cerr << "Error: model does not exist: " << cfg.model_path << "\n";
    return 2;
  }

  try {
    if (!cfg.save_dir.empty()) {
      fs::create_directories(cfg.save_dir);
    }
    for (const auto sig : {SIGINT, SIGTERM, SIGHUP}) {
      std::signal(sig, request_stop);
    }
    return run(cfg);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
