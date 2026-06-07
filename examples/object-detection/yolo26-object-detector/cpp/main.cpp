/**
 * @example yolo26-object-detector.cpp
 * Minimal yolo26m sync pipeline: infer detections for every image in a folder.
 *
 * Usage: yolo26-object-detector [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kDefaultMinScore = 0.25f;
constexpr float kDefaultNmsIou = 0.45f;
constexpr int kMaxDet = 100;
constexpr int kTimeoutMs = 20000;

struct Config {
  std::string model_path;
  std::string labels_path;
  std::string input_dir;
  std::string output_dir;
  float min_score = kDefaultMinScore;
  float nms_iou = kDefaultNmsIou;
  int max_detections = kMaxDet;
  int timeout_ms = kTimeoutMs;
  bool profile = false;
  bool overlay = true;
  int num_runs = 1;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "assets/models/yolo26m-det-bf16-mla_tess-b1.tar.gz");
  cfg.labels_path = raw.string_or(
      "model.labels", "examples/object-detection/yolo26-object-detector/common/coco_label.txt");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/test_images");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/yolo26-object-detector");
  cfg.min_score = static_cast<float>(raw.double_or("decode.score_threshold", kDefaultMinScore));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", kDefaultNmsIou));
  cfg.max_detections = raw.int_or("decode.max_detections", kMaxDet);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kTimeoutMs);
  cfg.num_runs = raw.int_or("runtime.num_runs", 1);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.overlay = raw.bool_or("output.overlay", true);
  if (cfg.min_score < 0.0f || cfg.min_score > 1.0f) {
    throw std::runtime_error("decode.score_threshold must be in [0.0, 1.0]");
  }
  if (cfg.nms_iou < 0.0f || cfg.nms_iou > 1.0f) {
    throw std::runtime_error("decode.nms_iou must be in [0.0, 1.0]");
  }
  if (cfg.max_detections < 1) {
    throw std::runtime_error("decode.max_detections must be >= 1");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
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

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

int clear_output_images(const fs::path& output_dir, const fs::path& input_dir) {
  if (fs::weakly_canonical(output_dir) == fs::weakly_canonical(input_dir)) {
    std::cerr << "Skipping output cleanup because output_dir matches input_dir: " << output_dir
              << "\n";
    return 0;
  }

  int removed = 0;
  for (const auto& entry : fs::directory_iterator(output_dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      fs::remove(entry.path());
      ++removed;
    }
  }
  return removed;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream in(labels_path);
  if (!in.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      labels.push_back(line);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

std::vector<simaai::neat::Box> decode_detections(const simaai::neat::TensorList& outputs,
                                                 int image_width, int image_height,
                                                 int max_detections) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no detection tensors");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("expected one BBOX tensor from model-managed BoxDecode, got " +
                             std::to_string(outputs.size()));
  }

  return simaai::neat::decode_bbox_tensor(outputs.front(), image_width, image_height,
                                          max_detections, /*strict=*/false)
      .boxes;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0)};
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

void draw_boxes(cv::Mat& frame, const std::vector<simaai::neat::Box>& boxes,
                const std::vector<std::string>& labels) {
  for (const auto& b : boxes) {
    const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x1))));
    const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y1))));
    const int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(b.x2))));
    const int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(b.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color = class_color(b.class_id);
    const std::string text = class_name(labels, b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    int baseline = 0;
    const cv::Size tsz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int y_text_top = std::max(0, y1 - tsz.height - 4);
    const int y_text_bottom = std::max(0, y1);
    cv::rectangle(frame, cv::Point(x1, y_text_top), cv::Point(x1 + tsz.width, y_text_bottom), color,
                  cv::FILLED);
    cv::putText(frame, text, cv::Point(x1, std::max(0, y1 - 2)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  Config cfg;
  try {
    cfg = parse_config(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  const fs::path input_dir = cfg.input_dir;
  const fs::path output_dir = cfg.output_dir;
  const fs::path labels_path = cfg.labels_path;

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  fs::create_directories(output_dir);
  const int removed_outputs = clear_output_images(output_dir, input_dir);
  if (removed_outputs > 0) {
    std::cout << "Cleared " << removed_outputs << " stale output images\n";
  }

  std::vector<std::string> labels;
  try {
    labels = load_labels(labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());

  if (images.empty()) {
    std::cerr << "No images found in " << input_dir << "\n";
    return 3;
  }
  std::cout << "Found " << images.size() << " images\n";

  try {
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
    model_opt.score_threshold = cfg.min_score;
    model_opt.nms_iou_threshold = cfg.nms_iou;
    model_opt.top_k = cfg.max_detections;
    model_opt.processcvu.post_run_target = "A65";

    simaai::neat::Model model(cfg.model_path, model_opt);

    cv::Mat seed_bgr = cv::imread(images.front().string(), cv::IMREAD_COLOR);
    if (seed_bgr.empty()) {
      throw std::runtime_error("failed to read build seed image: " + images.front().string());
    }
    std::cout << "[BUILD] Building pipeline...\n";
    auto runner = model.build(std::vector<cv::Mat>{seed_bgr}, simaai::neat::Model::RouteOptions{});
    std::cout << "[BUILD] Pipeline built\n";

    // Warmup inference to stabilize timing before profiling.
    runner.run(std::vector<cv::Mat>{seed_bgr}, cfg.timeout_ms);
    std::cout << "[WARMUP] done\n";

    const int total_images = static_cast<int>(images.size()) * cfg.num_runs;
    if (cfg.num_runs > 1) {
      std::cout << "Looping " << cfg.num_runs << "x over " << images.size() << " images ("
                << total_images << " total)\n";
    }

    const auto pipeline_start = std::chrono::steady_clock::now();
    int processed = 0;

    for (int run_idx = 0; run_idx < cfg.num_runs; ++run_idx) {
      for (const auto& image_path : images) {
        const auto img_start = std::chrono::steady_clock::now();

        cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
          std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
          continue;
        }

        const auto infer_start = std::chrono::steady_clock::now();
        simaai::neat::TensorList out = runner.run(std::vector<cv::Mat>{bgr}, cfg.timeout_ms);
        const auto infer_end = std::chrono::steady_clock::now();

        const std::vector<simaai::neat::Box> boxes =
            decode_detections(out, bgr.cols, bgr.rows, cfg.max_detections);
        const auto decode_end = std::chrono::steady_clock::now();

        if (cfg.overlay) {
          draw_boxes(bgr, boxes, labels);
          const fs::path out_path = output_dir / (image_path.stem().string() + ".png");
          if (!cv::imwrite(out_path.string(), bgr)) {
            std::cerr << "Failed to write: " << out_path << "\n";
            continue;
          }
        }
        const auto img_end = std::chrono::steady_clock::now();

        ++processed;
        if (cfg.overlay) {
          std::cout << "[" << processed << "/" << total_images << "] " << image_path.filename()
                    << " -> " << image_path.stem().string() << ".png"
                    << " (" << boxes.size() << " detections)\n";
        } else {
          std::cout << "[" << processed << "/" << total_images << "] " << image_path.filename()
                    << " (" << boxes.size() << " detections)\n";
        }

        if (cfg.profile) {
          using ms = std::chrono::duration<double, std::milli>;
          const auto pre_ms = ms(infer_start - img_start).count();
          const auto inf_ms = ms(infer_end - infer_start).count();
          const auto dec_ms = ms(decode_end - infer_end).count();
          const auto post_ms = ms(img_end - decode_end).count();
          const auto tot_ms = ms(img_end - img_start).count();
          std::cout << "[PROFILE] " << image_path.filename().string()
                    << ": preprocess=" << cv::format("%.1f", pre_ms)
                    << "ms inference=" << cv::format("%.1f", inf_ms)
                    << "ms decode=" << cv::format("%.1f", dec_ms)
                    << "ms overlay+save=" << cv::format("%.1f", post_ms)
                    << "ms total=" << cv::format("%.1f", tot_ms) << "ms\n";
        }
      }
    }

    runner.close();

    if (cfg.profile && processed > 0) {
      const auto pipeline_end = std::chrono::steady_clock::now();
      const auto total_s = std::chrono::duration<double>(pipeline_end - pipeline_start).count();
      const auto avg_ms = (total_s * 1000.0) / static_cast<double>(processed);
      const auto fps = static_cast<double>(processed) / total_s;
      std::cout << "[PROFILE] Total: " << processed << " images in " << cv::format("%.1f", total_s)
                << "s (avg " << cv::format("%.1f", avg_ms) << "ms/image, "
                << cv::format("%.1f", fps) << " FPS)\n";
    }

    std::cout << "Done: " << processed << " images processed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
