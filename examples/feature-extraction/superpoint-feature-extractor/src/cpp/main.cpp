/**
 * @example superpoint-feature-extractor.cpp
 * Run SuperPoint on a video and stream the feature-point overlay to Insight.
 *
 * Usage: superpoint-feature-extractor [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
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

namespace {

constexpr int kModelWidth = 640;
constexpr int kModelHeight = 480;
constexpr int kDescriptorDim = 256;
constexpr int kMaxPoints = 600;

struct Config {
  fs::path model;
  fs::path input;
  std::string insight_host;
  int video_port = 9000;
  int channel = 0;
  int bitrate_kbps = 1000;
  int frames = 0;
  int timeout_ms = 20000;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or("model.path", "models/superpoint_mpk.tar.gz");
  cfg.input = raw.string_or("io.input", "assets/datasets/tum-rgbd/freiburg1-desk.mp4");
  cfg.insight_host = raw.string_or("output.insight.host", "127.0.0.1");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.channel = raw.int_or("output.insight.channel", 0);
  cfg.bitrate_kbps = raw.int_or("output.insight.bitrate_kbps", 1000);
  cfg.frames = raw.int_or("runtime.frames", 0);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 20000);
  if (cfg.frames < 0) {
    throw std::runtime_error("runtime.frames must be >= 0");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
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

std::vector<float> keypoints(const neat::FeaturePointTensors& features, int output_width,
                             int output_height) {
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
  for (std::size_t i = 0; i < points.size(); i += 2) {
    const float x = points[i];
    const float y = points[i + 1];
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F || x >= kModelWidth ||
        y >= kModelHeight) {
      throw std::runtime_error("SuperPoint returned an invalid keypoint coordinate");
    }
    points[i] = x * static_cast<float>(output_width) / static_cast<float>(kModelWidth);
    points[i + 1] = y * static_cast<float>(output_height) / static_cast<float>(kModelHeight);
  }
  return points;
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
  input_options.format = "RGB";
  input_options.width = width;
  input_options.height = height;
  input_options.depth = 3;
  input_options.fps_n = output_fps;
  input_options.fps_d = 1;
  input_options.memory_policy = neat::InputMemoryPolicy::Ev74;

  auto sender_options =
      neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromRaw(width, height, output_fps);
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
      neat::Tensor::from_cv_mat(seed, neat::ImageSpec::PixelFormat::RGB, neat::TensorMemory::EV74);
  sender.run = sender.graph.build(neat::TensorList{tensor});
  return sender;
}

void stream_frame(neat::Run& run, const cv::Mat& frame) {
  cv::Mat rgb;
  cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
  const auto tensor =
      neat::Tensor::from_cv_mat(rgb, neat::ImageSpec::PixelFormat::RGB, neat::TensorMemory::EV74);
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
    if (!fs::is_regular_file(cfg.input)) {
      throw std::runtime_error("input video does not exist: " + cfg.input.string());
    }

    cv::VideoCapture video(cfg.input.string());
    cv::Mat frame;
    if (!video.isOpened() || !video.read(frame)) {
      throw std::runtime_error("failed to read input video: " + cfg.input.string());
    }
    validate_frame(frame);
    const int input_width = frame.cols;
    const int input_height = frame.rows;

    const double input_fps = video.get(cv::CAP_PROP_FPS);
    const double fps = std::isfinite(input_fps) && input_fps > 0.0 ? input_fps : 30.0;

    neat::Model model(cfg.model.string(), model_options(input_width, input_height));
    auto input = prepare_input(frame);
    auto runner = model.build(neat::TensorList{input});
    auto video_sender = build_video_sender(cfg, fps, input_width, input_height);

    std::size_t total_points = 0;
    int processed = 0;
    while (true) {
      const auto decoded =
          neat::decode_superpoint(runner.run(neat::TensorList{input}, cfg.timeout_ms));
      if (decoded.size() != 1) {
        throw std::runtime_error("SuperPoint must return one feature set per frame");
      }
      const auto points = keypoints(decoded.front(), frame.cols, frame.rows);
      total_points += points.size() / 2;
      draw_points(frame, points);
      stream_frame(video_sender.run, frame);
      ++processed;

      if ((cfg.frames > 0 && processed >= cfg.frames) || !video.read(frame)) {
        break;
      }
      validate_frame(frame, input_width, input_height);
      input = prepare_input(frame);
    }

    runner.close();
    video_sender.run.close();
    const double average = static_cast<double>(total_points) / processed;
    std::cout << "frames=" << processed << " average_points=" << cv::format("%.1f", average)
              << " descriptor_dim=" << kDescriptorDim << " video_sender=" << cfg.insight_host << ":"
              << video_sender.port << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 2;
  }
}
