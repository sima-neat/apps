// Copyright 2026 SiMa Technologies, Inc.
// SPDX-License-Identifier: Apache-2.0

/**
 * @example edgetam-video-object-segmenter.cpp
 * Follow one object through a video file with EdgeTAM and save its mask in every frame.
 */
#include "neat.h"
#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"
#include "utils/tracker.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;

namespace {

constexpr int kRunTimeoutMs = 10000;
// The mask's tint and outline, and the ring around the prompt point, in BGR.
const cv::Scalar kMaskColor(255, 170, 0);
const cv::Scalar kPromptColor(255, 255, 255);
constexpr double kMaskOpacity = 0.45;

struct Config {
  std::string image_encoder;
  std::string memory_attention;
  std::string mask_decoder;
  std::string memory_encoder;
  std::string video_path;
  double prompt_x = 0.0;
  double prompt_y = 0.0;
  int frames = 0;
  fs::path output_dir;
  int save_every = 0;
};

struct Runners {
  neat::Model::Runner image_encoder;
  neat::Model::Runner memory_attention;
  neat::Model::Runner mask_decoder;
  neat::Model::Runner memory_encoder;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.image_encoder = raw.string_or("model.image_encoder", "");
  cfg.memory_attention = raw.string_or("model.memory_attention", "");
  cfg.mask_decoder = raw.string_or("model.mask_decoder", "");
  cfg.memory_encoder = raw.string_or("model.memory_encoder", "");
  cfg.video_path = raw.string_or("source.video_path", "");
  cfg.prompt_x = raw.double_or("prompt.x", -1.0);
  cfg.prompt_y = raw.double_or("prompt.y", -1.0);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.output_dir = raw.string_or("output.dir", "sandbox/edgetam-video-object-segmenter");
  cfg.save_every = raw.int_or("output.save_every", 0);
  for (const auto& [name, model_path] :
       {std::pair<std::string, std::string>{"image_encoder", cfg.image_encoder},
        {"memory_attention", cfg.memory_attention},
        {"mask_decoder", cfg.mask_decoder},
        {"memory_encoder", cfg.memory_encoder}}) {
    sima_examples::require(fs::is_regular_file(model_path),
                           "model." + name + " " + model_path +
                               " does not exist; download the EdgeTAM model and point it at that "
                               "package");
  }
  sima_examples::require(cfg.prompt_x >= 0 && cfg.prompt_x <= 1 && cfg.prompt_y >= 0 &&
                             cfg.prompt_y <= 1,
                         cv::format("prompt.x %g and prompt.y %g must be fractions of the frame, "
                                    "from 0 to 1",
                                    cfg.prompt_x, cfg.prompt_y));
  sima_examples::require(cfg.frames >= 0, "inference.frames must be 0 or more");
  sima_examples::require(cfg.save_every >= 0, "output.save_every must be 0 or more");
  return cfg;
}

cv::VideoCapture open_video(const std::string& path) {
  cv::VideoCapture video(path);
  sima_examples::require(video.isOpened(),
                         "could not open source.video_path " + path + "; point it at a video file");
  return video;
}

neat::Model::Options image_encoder_options(int width, int height) {
  // Neat stretches the RGB frame bicubically to the model's 1024x1024 input and applies ImageNet's
  // mean and std on the board.
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Image;
  options.preprocess.enable = neat::AutoFlag::On;
  options.preprocess.input_max_width = width;
  options.preprocess.input_max_height = height;
  options.preprocess.resize.enable = neat::AutoFlag::On;
  options.preprocess.resize.mode = neat::ResizeMode::Stretch;
  options.preprocess.resize.width = options.preprocess.resize.height = edgetam::kImageSize;
  options.preprocess.resize.interpolation = neat::ResizeInterpolation::Bicubic;
  options.preprocess.normalize.enable = neat::AutoFlag::On;
  options.preprocess.preset = neat::NormalizePreset::ImageNet;
  // The output stage runs on the EV74 and writes memory the other models' input stages read, so
  // the embeddings and features pass on without a copy.
  options.processcvu.post_run_target = "EV74";
  return options;
}

neat::Model::Options tensor_options() {
  // The inputs are features, memories and prompts, not images.
  neat::Model::Options options;
  options.preprocess.kind = neat::InputKind::Tensor;
  return options;
}

neat::Model::Options mask_decoder_options() {
  // The input stage runs on the CPU, which reads the memory attention's output where its CPU
  // output stage wrote it and the image encoder's features where the EV74 wrote them.
  neat::Model::Options options = tensor_options();
  options.processcvu.pre_run_target = "A65";
  return options;
}

neat::Tensor ev74_tensor(const std::vector<float>& values, const edgetam::Shape& shape) {
  return neat::Tensor::from_vector(values, shape, neat::TensorMemory::EV74);
}

neat::Tensor zero_tensor(const edgetam::Shape& shape) {
  std::size_t size = 1;
  for (const int64_t dim : shape) {
    size *= static_cast<std::size_t>(dim);
  }
  return ev74_tensor(std::vector<float>(size, 0.0F), shape);
}

neat::Tensor frame_tensor(const cv::Mat& frame_bgr) {
  cv::Mat frame_rgb;
  cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
  return neat::Tensor::from_cv_mat(frame_rgb, neat::ImageSpec::PixelFormat::RGB,
                                   neat::TensorMemory::EV74);
}

neat::TensorList memory_tensors(const edgetam::MemoryArrays& arrays) {
  return {ev74_tensor(arrays.memories, edgetam::kMemoriesShape),
          ev74_tensor(arrays.used_memories, edgetam::kUsedMemoriesShape),
          ev74_tensor(arrays.pointers, edgetam::kPointersShape),
          ev74_tensor(arrays.used_pointers, edgetam::kUsedPointersShape)};
}

neat::TensorList prompt_tensors(const edgetam::PromptArrays& prompt) {
  return {ev74_tensor(prompt.coords, edgetam::kPointCoordsShape),
          ev74_tensor(prompt.types, edgetam::kPointTypesShape)};
}

// Loads the four models and builds one runner for each, from inputs of the shapes it takes.
Runners build_runners(const Config& cfg, int width, int height) {
  const neat::Tensor frame = frame_tensor(cv::Mat::zeros(height, width, CV_8UC3));
  const neat::Tensor embeddings = zero_tensor(edgetam::kEmbeddingShape);
  const neat::Tensor high_res_0 = zero_tensor(edgetam::kHighResShapes[0]);
  const neat::Tensor high_res_1 = zero_tensor(edgetam::kHighResShapes[1]);
  edgetam::MemoryBank bank;
  bank.add(std::vector<float>(edgetam::kMemoryTokens * edgetam::kMemoryDim, 0.0F),
           std::vector<float>(edgetam::kPointerDim, 0.0F));
  const neat::TensorList memories = memory_tensors(bank.arrays());
  const neat::TensorList prompt = prompt_tensors(edgetam::prompt_arrays(std::nullopt));
  const neat::Tensor mask = zero_tensor(edgetam::kMemoryMaskShape);

  neat::Model image_encoder(cfg.image_encoder, image_encoder_options(width, height));
  neat::Model memory_attention(cfg.memory_attention, tensor_options());
  neat::Model mask_decoder(cfg.mask_decoder, mask_decoder_options());
  neat::Model memory_encoder(cfg.memory_encoder, tensor_options());
  return {image_encoder.build(neat::TensorList{frame}),
          memory_attention.build({embeddings, memories[0], memories[1], memories[2], memories[3]}),
          mask_decoder.build({embeddings, high_res_0, high_res_1, prompt[0], prompt[1]}),
          memory_encoder.build({embeddings, mask})};
}

// The model's outputs, failing when it returned another number of them or none in time.
neat::TensorList run_model(neat::Model::Runner& runner, const neat::TensorList& inputs,
                           std::size_t outputs, const std::string& model) {
  neat::TensorList tensors = runner.run(inputs, kRunTimeoutMs);
  sima_examples::require(tensors.size() == outputs,
                         "model." + model + " returned " + std::to_string(tensors.size()) +
                             " outputs in " + std::to_string(kRunTimeoutMs) + " ms, not " +
                             std::to_string(outputs) + "; point it at EdgeTAM's package");
  return tensors;
}

// Segments the object in one frame and adds the frame to the memory bank. The first frame takes
// the prompt point; a later frame takes none and finds the object from the bank. Returns the
// [256, 256] mask logits.
cv::Mat track_frame(Runners& runners, edgetam::MemoryBank& bank, const cv::Mat& frame_bgr,
                    const std::optional<cv::Point2d>& prompt) {
  const neat::TensorList encoded =
      run_model(runners.image_encoder, {frame_tensor(frame_bgr)}, 3, "image_encoder");
  const neat::Tensor& embeddings = encoded[0];
  neat::Tensor conditioned = embeddings;
  if (!prompt) {
    const neat::TensorList memories = memory_tensors(bank.arrays());
    conditioned = run_model(runners.memory_attention,
                            {embeddings, memories[0], memories[1], memories[2], memories[3]}, 1,
                            "memory_attention")[0];
  }
  const neat::TensorList points = prompt_tensors(edgetam::prompt_arrays(prompt));
  const neat::TensorList decoded =
      run_model(runners.mask_decoder, {conditioned, encoded[1], encoded[2], points[0], points[1]},
                4, "mask_decoder");
  edgetam::ChosenMask chosen = edgetam::choose_mask(
      sima_examples::tensor_to_floats(decoded[0]), sima_examples::tensor_to_floats(decoded[1]),
      sima_examples::tensor_to_floats(decoded[2]), sima_examples::tensor_to_floats(decoded[3]));
  const neat::Tensor remembered =
      ev74_tensor(edgetam::memory_mask(chosen.mask, prompt.has_value()), edgetam::kMemoryMaskShape);
  const neat::TensorList memory =
      run_model(runners.memory_encoder, {embeddings, remembered}, 1, "memory_encoder");
  bank.add(sima_examples::tensor_to_floats(memory[0]), std::move(chosen.pointer));
  return chosen.mask;
}

// Tints and outlines the object in the frame, in place, and rings the prompt point on the
// prompted frame.
void draw_mask(cv::Mat& frame_bgr, const cv::Mat& mask, const std::optional<cv::Point2d>& prompt) {
  cv::Mat logits;
  cv::resize(mask, logits, frame_bgr.size(), 0, 0, cv::INTER_LINEAR);
  const cv::Mat covered = logits > 0;
  cv::Mat tinted;
  cv::addWeighted(frame_bgr, 1 - kMaskOpacity, cv::Mat(frame_bgr.size(), CV_8UC3, kMaskColor),
                  kMaskOpacity, 0, tinted);
  tinted.copyTo(frame_bgr, covered);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(covered, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(frame_bgr, contours, -1, kMaskColor, 2);
  if (prompt) {
    cv::circle(frame_bgr,
               cv::Point(static_cast<int>(prompt->x * frame_bgr.cols),
                         static_cast<int>(prompt->y * frame_bgr.rows)),
               6, kPromptColor, 2);
  }
}

nlohmann::ordered_json frame_result(int index, const edgetam::ObjectLocation& location) {
  nlohmann::ordered_json result = {
      {"frame", index}, {"visible", location.visible}, {"area", location.area}};
  result["center_x"] = location.center_x ? nlohmann::ordered_json(*location.center_x) : nullptr;
  result["center_y"] = location.center_y ? nlohmann::ordered_json(*location.center_y) : nullptr;
  return result;
}

void run(const Config& cfg) {
  cv::VideoCapture video = open_video(cfg.video_path);
  const int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
  fs::create_directories(cfg.output_dir);
  Runners runners = build_runners(cfg, width, height);
  std::cout << cv::format("video=%s frame=%dx%d prompt=(%g, %g)", cfg.video_path.c_str(), width,
                          height, cfg.prompt_x, cfg.prompt_y)
            << std::endl;

  edgetam::MemoryBank bank;
  nlohmann::ordered_json results = nlohmann::ordered_json::array();
  int visible = 0;
  const double started_ms = sima_examples::time_ms();
  const auto close_runners = [&] {
    runners.image_encoder.close();
    runners.memory_attention.close();
    runners.mask_decoder.close();
    runners.memory_encoder.close();
  };
  try {
    cv::Mat frame_bgr;
    while ((cfg.frames == 0 || static_cast<int>(results.size()) < cfg.frames) &&
           video.read(frame_bgr)) {
      const int index = static_cast<int>(results.size());
      const std::optional<cv::Point2d> prompt =
          index == 0 ? std::optional<cv::Point2d>({cfg.prompt_x, cfg.prompt_y}) : std::nullopt;
      const cv::Mat mask = track_frame(runners, bank, frame_bgr, prompt);
      const edgetam::ObjectLocation location = edgetam::locate_object(mask);
      results.push_back(frame_result(index, location));
      visible += location.visible ? 1 : 0;
      if (cfg.save_every > 0 && index % cfg.save_every == 0) {
        draw_mask(frame_bgr, mask, prompt);
        const fs::path image_path = cfg.output_dir / cv::format("frame_%06d.jpg", index);
        sima_examples::require(cv::imwrite(image_path.string(), frame_bgr),
                               "could not write " + image_path.string() +
                                   "; check that output.dir is writable");
        const std::string where =
            location.visible
                ? cv::format("object at (%.2f, %.2f), %.1f%% of the frame", *location.center_x,
                             *location.center_y, location.area * 100)
                : "no object";
        std::cout << "frame " << index << ": " << where << " -> " << image_path.string()
                  << std::endl;
      }
    }
  } catch (...) {
    close_runners();
    throw;
  }
  close_runners();
  sima_examples::require(!results.empty(), cfg.video_path + " has no frame to read");

  const double seconds = (sima_examples::time_ms() - started_ms) / 1000.0;
  const fs::path results_path = cfg.output_dir / "results.json";
  std::ofstream(results_path) << nlohmann::ordered_json{{"frames", results}}.dump(2) << "\n";
  std::cout << cv::format("tracked %d frames in %.1f s (%.1f frames/s), object visible in %d; "
                          "results in %s",
                          static_cast<int>(results.size()), seconds, results.size() / seconds,
                          visible, results_path.c_str())
            << std::endl;
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
                  << "Follow one object through a video file with EdgeTAM, from its position on "
                     "the first frame, and save its mask in every frame\n";
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
