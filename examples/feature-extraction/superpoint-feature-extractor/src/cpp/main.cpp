/**
 * @example superpoint-feature-extractor.cpp
 * Run SuperPoint on an RTSP stream and send the feature-point overlay to Insight.
 *
 * Usage: superpoint-feature-extractor [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/pull_status.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kDescriptorDim = 256;
constexpr int kMaxPoints = 600;
volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int) { stop_requested = 1; }

struct Config {
  fs::path model;
  std::string rtsp_url;
  bool tcp = true;
  int latency_ms = 100;
  std::string insight_host;
  int video_port = 9000;
  int channel = 0;
  int bitrate_kbps = 1000;
  int frames = 0;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or("model.path", "models/superpoint_mpk.tar.gz");
  cfg.rtsp_url = raw.string_or("source.rtsp_url", "");
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.latency_ms = raw.int_or("source.latency_ms", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "127.0.0.1");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.channel = raw.int_or("output.insight.channel", 0);
  cfg.bitrate_kbps = raw.int_or("output.insight.bitrate_kbps", 1000);
  cfg.frames = raw.int_or("runtime.frames", 0);
  if (cfg.frames < 0) {
    throw std::runtime_error("runtime.frames must be >= 0");
  }
  if (cfg.rtsp_url.empty() || cfg.latency_ms < 0) {
    throw std::runtime_error("set source.rtsp_url and a non-negative source.latency_ms");
  }
  if (cfg.insight_host.empty()) {
    throw std::runtime_error("output.insight.host must be set");
  }
  if (cfg.video_port <= 0 || cfg.video_port > 65535) {
    throw std::runtime_error("output.insight.video_port must be in [1, 65535]");
  }
  if (cfg.channel < 0 || cfg.video_port + cfg.channel > 65535) {
    throw std::runtime_error("output.insight channel selects an invalid UDP port");
  }
  if (cfg.bitrate_kbps <= 0) {
    throw std::runtime_error("output.insight.bitrate_kbps must be > 0");
  }
  return cfg;
}

Config parse_config(int argc, char** argv) {
  fs::path config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      config_path = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [--config <path>]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return load_config(config_path);
}

std::vector<float> tensor_floats(const neat::Tensor& tensor) {
  if (tensor.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("SuperPoint keypoints must be Float32");
  }
  const auto bytes = tensor.copy_dense_bytes_tight();
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("invalid SuperPoint keypoint byte count");
  }
  std::vector<float> values(bytes.size() / sizeof(float));
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

struct FrameSize {
  int width = 0;
  int height = 0;
};

std::vector<float> keypoints(const neat::FeaturePointTensors& features,
                             const FrameSize& source_size) {
  if (features.keypoints.shape.size() != 2 || features.keypoints.shape[1] != 2 ||
      features.scores.shape.size() != 1 || features.descriptors.shape.size() != 2) {
    throw std::runtime_error("invalid SuperPoint output ranks");
  }

  const auto count = features.keypoints.shape[0];
  if (count < 0 || count > kMaxPoints || features.scores.shape[0] != count ||
      features.descriptors.shape[0] != count || features.descriptors.shape[1] != kDescriptorDim) {
    throw std::runtime_error("invalid SuperPoint output shapes");
  }
  if (features.scores.dtype != neat::TensorDType::Float32 ||
      features.descriptors.dtype != neat::TensorDType::Float32) {
    throw std::runtime_error("unexpected SuperPoint output dtype");
  }

  auto points = tensor_floats(features.keypoints);
  if (points.size() != static_cast<std::size_t>(count) * 2U) {
    throw std::runtime_error("SuperPoint keypoint data does not match its shape");
  }
  std::vector<float> drawable_points;
  drawable_points.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); i += 2) {
    const float x = points[i];
    const float y = points[i + 1];
    // The frame edges are exclusive coordinates. Discard isolated decoder outliers instead of
    // aborting the stream when inverse-affine rounding places a point on or beyond an edge.
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F || x >= source_size.width ||
        y >= source_size.height) {
      continue;
    }
    drawable_points.push_back(x);
    drawable_points.push_back(y);
  }
  return drawable_points;
}

void draw_points(cv::Mat& frame, const std::vector<float>& points) {
  for (std::size_t i = 0; i < points.size(); i += 2) {
    cv::circle(frame, cv::Point(cvRound(points[i]), cvRound(points[i + 1])), 2,
               cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
  }
  cv::putText(frame, "points: " + std::to_string(points.size() / 2), cv::Point(12, 26),
              cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

neat::Tensor prepare_input(const cv::Mat& frame) {
  return neat::Tensor::from_cv_mat(frame, neat::ImageSpec::PixelFormat::BGR,
                                   neat::TensorMemory::EV74);
}

struct Source {
  neat::Graph graph{"rtsp"};
  neat::Run run;
};

Source build_source(const Config& cfg, const sima_examples::RtspStreamInfo& stream) {
  neat::nodes::groups::RtspDecodedInputOptions options;
  options.url = cfg.rtsp_url;
  options.tcp = cfg.tcp;
  options.latency_ms = cfg.latency_ms;
  options.fallback_h264_width = stream.width;
  options.fallback_h264_height = stream.height;
  options.fallback_h264_fps = stream.fps;
  auto output = neat::OutputOptions::Latest();
  output.max_buffers = 4;
  Source source;
  source.graph.connect(neat::nodes::groups::RtspDecodedInput(options),
                       neat::nodes::Output("frame", output));
  neat::RunOptions run_options;
  run_options.preset = neat::RunPreset::Realtime;
  run_options.queue_depth = 4;
  run_options.overflow_policy = neat::OverflowPolicy::KeepLatest;
  run_options.output_memory = neat::OutputMemory::ZeroCopy;
  source.run = source.graph.build(run_options);
  return source;
}

cv::Mat pull_frame(neat::Run& run) {
  neat::Sample sample;
  neat::PullError error;
  const auto status = run.pull("frame", 20000, sample, &error);
  if (!sima_examples::pull_status_has_sample(status, "frame", error, run.last_error())) {
    throw std::runtime_error("timed out waiting for an RTSP frame");
  }
  const neat::Sample* field = &sample;
  for (const auto& part : sample.fields) {
    if (part.stream_label == "frame")
      field = &part;
  }
  const auto tensors = neat::tensors_from_sample(*field);
  cv::Mat frame;
  std::string message;
  sima_examples::require(!tensors.empty(), "decoded frame has no tensor");
  sima_examples::require(sima_examples::nv12_to_bgr(tensors.front(), frame, message), message);
  return frame; // The CPU copy outlives the decoder loan held by sample.
}

neat::Model::Options model_options(int input_width, int input_height) {
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.input_max_width = input_width;
  options.preprocess.input_max_height = input_height;
  options.preprocess.input_max_depth = 3;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.color_convert.enable = neat::AutoFlag::On;
  options.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  options.preprocess.color_convert.output_format = neat::PreprocessColorFormat::GRAY8;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  options.preprocess.normalize.mean = {0.0F, 0.0F, 0.0F};
  options.preprocess.normalize.stddev = {1.0F, 1.0F, 1.0F};
  options.preprocess.normalize.has_explicit_stats = true;
  options.decode_type = neat::BoxDecodeType::SuperPoint;
  options.superpoint.profile = neat::SuperPointProfile::A65V1;
  options.superpoint.output_format = neat::SuperPointOutputFormat::FeaturePointsV1;
  options.superpoint.descriptor_output_dtype = neat::TensorDType::Float32;
  options.processcvu.post_run_target = "A65";
  return options;
}

struct VideoSender {
  neat::Graph graph{"insight"};
  neat::Run run;
  int port = 0;
};

VideoSender build_video_sender(const Config& cfg, double fps, int width, int height) {
  const int output_fps = std::max(1, cvRound(fps));
  neat::InputOptions input_options;
  input_options.payload_type = neat::PayloadType::Image;
  input_options.format = "BGR";
  input_options.width = width;
  input_options.height = height;
  input_options.depth = 3;
  input_options.fps_n = output_fps;
  input_options.fps_d = 1;

  auto sender_options = neat::nodes::groups::VideoSenderOptions::FromRaw({.fps = output_fps});
  sender_options.host = cfg.insight_host;
  sender_options.channel = cfg.channel;
  sender_options.video_port_base = cfg.video_port;
  sender_options.encoder.bitrate_kbps = cfg.bitrate_kbps;

  VideoSender sender;
  sender.port = sender_options.video_port();
  sender.graph.add(neat::nodes::Input(input_options));
  sender.graph.add(neat::nodes::groups::VideoSender(sender_options));
  cv::Mat seed(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  const auto tensor =
      neat::Tensor::from_cv_mat(seed, neat::ImageSpec::PixelFormat::BGR, neat::TensorMemory::CPU);
  sender.run = sender.graph.build(neat::TensorList{tensor});
  return sender;
}

void stream_frame(neat::Run& run, const cv::Mat& frame) {
  const auto tensor =
      neat::Tensor::from_cv_mat(frame, neat::ImageSpec::PixelFormat::BGR, neat::TensorMemory::CPU);
  if (!run.push(neat::TensorList{tensor})) {
    throw std::runtime_error("Insight video push failed");
  }
}

void validate_frame(const cv::Mat& frame, int expected_width = 0, int expected_height = 0) {
  if (frame.empty() || frame.cols <= 0 || frame.rows <= 0 || frame.type() != CV_8UC3) {
    throw std::runtime_error("SuperPoint input must be a non-empty BGR video frame");
  }
  if ((expected_width > 0 && frame.cols != expected_width) ||
      (expected_height > 0 && frame.rows != expected_height)) {
    throw std::runtime_error("SuperPoint input resolution changed after pipeline construction");
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_config(argc, argv);
    if (!fs::is_regular_file(cfg.model)) {
      throw std::runtime_error("model does not exist: " + cfg.model.string());
    }
    sima_examples::RtspProbeOptions probe;
    probe.rtsp_tcp = cfg.tcp;
    probe.latency_ms = cfg.latency_ms;
    sima_examples::RtspStreamInfo stream;
    sima_examples::require(
        sima_examples::probe_rtsp_stream_info(cfg.rtsp_url, probe, stream) &&
            stream.width > 0 && stream.height > 0 && stream.fps > 0,
        "could not read the frame size and rate of " + cfg.rtsp_url);
    auto source = build_source(cfg, stream);
    cv::Mat frame = pull_frame(source.run);
    validate_frame(frame, stream.width, stream.height);
    const int input_width = stream.width, input_height = stream.height;

    neat::Model model(cfg.model.string(), model_options(input_width, input_height));
    auto input = prepare_input(frame);
    auto runner = model.build(neat::TensorList{input});
    auto video_sender = build_video_sender(cfg, stream.fps, input_width, input_height);

    for (const auto sig : {SIGINT, SIGTERM, SIGHUP})
      std::signal(sig, request_stop);

    std::size_t total_points = 0;
    int processed = 0;
    while (!stop_requested && (cfg.frames <= 0 || processed < cfg.frames)) {
      const auto decoded = neat::decode_superpoint(runner.run(neat::TensorList{input}));
      if (decoded.size() != 1) {
        throw std::runtime_error("SuperPoint must return one feature set per frame");
      }
      const auto points = keypoints(decoded.front(), FrameSize{input_width, input_height});
      total_points += points.size() / 2;
      draw_points(frame, points);
      stream_frame(video_sender.run, frame);
      ++processed;

      if (stop_requested || (cfg.frames > 0 && processed >= cfg.frames)) {
        break;
      }
      frame = pull_frame(source.run);
      validate_frame(frame, input_width, input_height);
      input = prepare_input(frame);
    }

    runner.close();
    video_sender.run.close();
    source.run.close();
    const double average = processed ? static_cast<double>(total_points) / processed : 0.0;
    std::cout << "frames=" << processed << " average_points=" << cv::format("%.1f", average)
              << " descriptor_dim=" << kDescriptorDim << " video_sender=" << cfg.insight_host << ":"
              << video_sender.port << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 2;
  }
}
