/**
 * @example superpoint-feature-extractor.cpp
 * Run SuperPoint on a video and draw the extracted feature points.
 *
 * Usage: superpoint-feature-extractor [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kDescriptorDim = 256;
constexpr int kMaxPoints = 600;

struct Config {
  fs::path model;
  fs::path input;
  fs::path output;
  int frames = 0;
  int timeout_ms = 20000;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or("model.path", "models/superpoint_mpk.tar.gz");
  cfg.input = raw.string_or("io.input", "assets/datasets/tum-rgbd/freiburg1-desk.mp4");
  cfg.output = raw.string_or("io.output", "sandbox/superpoint-feature-extractor.mp4");
  cfg.frames = raw.int_or("runtime.frames", 0);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 20000);
  if (cfg.frames < 0) {
    throw std::runtime_error("runtime.frames must be >= 0");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
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

std::vector<float> keypoints(const neat::FeaturePointTensors& features) {
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
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0F || y < 0.0F || x >= kWidth ||
        y >= kHeight) {
      throw std::runtime_error("SuperPoint returned an invalid keypoint coordinate");
    }
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

std::uint16_t to_bfloat16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

neat::Tensor prepare_input(const cv::Mat& frame, neat::TensorDType dtype) {
  cv::Mat gray;
  cv::Mat input;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  gray.convertTo(input, CV_32FC1, 1.0 / 255.0);
  const float* begin = input.ptr<float>();

  if (dtype == neat::TensorDType::Float32) {
    auto tensor = neat::Tensor::from_vector(std::vector<float>(begin, begin + input.total()),
                                            {kHeight, kWidth, 1}, neat::TensorMemory::EV74);
    tensor.layout = neat::TensorLayout::HWC;
    return tensor;
  }
  if (dtype == neat::TensorDType::BFloat16) {
    std::vector<std::uint16_t> values(input.total());
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = to_bfloat16(begin[i]);
    }
    auto tensor = neat::Tensor::from_vector(values, {kHeight, kWidth, 1}, neat::TensorMemory::EV74);
    tensor.dtype = neat::TensorDType::BFloat16;
    tensor.layout = neat::TensorLayout::HWC;
    return tensor;
  }
  throw std::runtime_error("SuperPoint model input must be Float32 or BFloat16");
}

neat::TensorDType select_input_dtype(const neat::TensorConstraint& spec) {
  for (const auto dtype : spec.dtypes) {
    if (dtype == neat::TensorDType::Float32 || dtype == neat::TensorDType::BFloat16) {
      return dtype;
    }
  }
  throw std::runtime_error("SuperPoint model input must support Float32 or BFloat16");
}

neat::Model::Options model_options() {
  neat::Model::Options options;
  options.preprocess.enable = neat::AutoFlag::Off;
  options.decode_type = neat::BoxDecodeType::SuperPoint;
  options.superpoint.profile = neat::SuperPointProfile::A65V1;
  options.superpoint.output_format = neat::SuperPointOutputFormat::FeaturePointsV1;
  options.superpoint.descriptor_output_dtype = neat::TensorDType::Float32;
  options.boxdecode_original_width = kWidth;
  options.boxdecode_original_height = kHeight;
  options.boxdecode_resize_mode = neat::ResizeMode::Stretch;
  options.processcvu.post_run_target = "A65";
  return options;
}

cv::VideoWriter open_writer(const fs::path& output, double fps) {
  if (output.extension() != ".mp4") {
    throw std::runtime_error("io.output must use the .mp4 extension");
  }
  cv::VideoWriter writer;
  writer.open(output.string(), cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
              cv::Size(kWidth, kHeight));
  if (!writer.isOpened()) {
    throw std::runtime_error("failed to open H.264 output video: " + output.string());
  }
  return writer;
}

void validate_frame(const cv::Mat& frame) {
  if (frame.cols != kWidth || frame.rows != kHeight || frame.type() != CV_8UC3) {
    throw std::runtime_error("SuperPoint input must be 640x480 BGR video");
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
    if (fs::absolute(cfg.input).lexically_normal() == fs::absolute(cfg.output).lexically_normal()) {
      throw std::runtime_error("input and output paths must differ");
    }

    cv::VideoCapture video(cfg.input.string());
    cv::Mat frame;
    if (!video.isOpened() || !video.read(frame)) {
      throw std::runtime_error("failed to read input video: " + cfg.input.string());
    }
    validate_frame(frame);

    if (!cfg.output.parent_path().empty()) {
      fs::create_directories(cfg.output.parent_path());
    }
    const double input_fps = video.get(cv::CAP_PROP_FPS);
    const double fps = std::isfinite(input_fps) && input_fps > 0.0 ? input_fps : 30.0;
    auto writer = open_writer(cfg.output, fps);

    neat::Model model(cfg.model.string(), model_options());
    const auto input_specs = model.input_specs();
    if (input_specs.size() != 1) {
      throw std::runtime_error("SuperPoint model must expose exactly one input");
    }
    const auto input_dtype = select_input_dtype(input_specs.front());
    auto input = prepare_input(frame, input_dtype);
    auto runner = model.build(neat::TensorList{input});

    std::size_t total_points = 0;
    int processed = 0;
    while (true) {
      const auto decoded =
          neat::decode_superpoint(runner.run(neat::TensorList{input}, cfg.timeout_ms));
      if (decoded.size() != 1) {
        throw std::runtime_error("SuperPoint must return one feature set per frame");
      }
      const auto points = keypoints(decoded.front());
      total_points += points.size() / 2;
      draw_points(frame, points);
      writer.write(frame);
      ++processed;

      if ((cfg.frames > 0 && processed >= cfg.frames) || !video.read(frame)) {
        break;
      }
      validate_frame(frame);
      input = prepare_input(frame, input_dtype);
    }

    runner.close();
    writer.release();
    const double average = static_cast<double>(total_points) / processed;
    std::cout << "frames=" << processed << " average_points=" << cv::format("%.1f", average)
              << " descriptor_dim=" << kDescriptorDim << " output=" << cfg.output << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 2;
  }
}
