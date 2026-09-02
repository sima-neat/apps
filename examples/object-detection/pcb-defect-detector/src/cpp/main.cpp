/**
 * @example pcb-defect-detector.cpp
 * PCB defect detection: run every image in a folder through a custom YOLO26n pack.
 *
 * Images of any resolution are letterboxed to the model input (640x640) before
 * inference. The compiled model pack owns color conversion, normalization, MLA
 * inference, and the on-device YOLO26 box decode. Detections come back in
 * letterboxed coordinates, are mapped onto the original frame, and are drawn on
 * the original image, one annotated image per input image.
 *
 * Usage: pcb-defect-detector [--config <path>] [--score <value>] [--nms <value>]
 *                            [--validate-config-only]
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
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kDefaultScoreThreshold = 0.25f;
constexpr float kDefaultNmsIou = 0.45f;
constexpr int kDefaultMaxDetections = 300;
constexpr int kDefaultTimeoutMs = 8000;
constexpr int kDefaultNumRuns = 1;
constexpr int kDefaultQueueDepth = 8;
constexpr int kDefaultInputSize = 640;
// Grey pad value used by the YOLO letterbox convention.
constexpr int kPadValue = 114;

struct Config {
  std::string model_path;
  std::string labels_path;
  int input_size = kDefaultInputSize;
  std::string input_dir;
  std::string output_dir;
  float score_threshold = kDefaultScoreThreshold;
  float nms_iou = kDefaultNmsIou;
  int max_detections = kDefaultMaxDetections;
  int timeout_ms = kDefaultTimeoutMs;
  int num_runs = kDefaultNumRuns;
  int queue_depth = kDefaultQueueDepth;
  bool profile = false;
  bool overlay = true;
};

struct CliOptions {
  fs::path config_path;
  std::optional<float> score;
  std::optional<float> nms;
  bool validate_only = false;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--config <path>] [--score <value>] [--nms <value>] [--validate-config-only]\n"
            << "  --config <path>          Path to YAML configuration\n"
            << "  --score <value>          Override decode.score_threshold\n"
            << "  --nms <value>            Override decode.nms_iou\n"
            << "  --validate-config-only   Validate the configuration and exit\n";
}

float parse_float_option(const std::string& flag, const std::string& value) {
  try {
    return std::stof(value);
  } catch (const std::exception&) {
    throw std::runtime_error(flag + " requires a number, got: " + value);
  }
}

CliOptions parse_cli(int argc, char** argv) {
  CliOptions cli;
  cli.config_path = sima_examples::default_config_path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR);
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config requires a path");
      }
      cli.config_path = argv[++i];
    } else if (arg == "--score") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--score requires a value");
      }
      cli.score = parse_float_option(arg, argv[++i]);
    } else if (arg == "--nms") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--nms requires a value");
      }
      cli.nms = parse_float_option(arg, argv[++i]);
    } else if (arg == "--validate-config-only") {
      cli.validate_only = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return cli;
}

Config load_config(const CliOptions& cli) {
  const auto raw = sima_examples::ScalarConfig::load(cli.config_path);

  Config cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or(
      "model.labels", "examples/object-detection/pcb-defect-detector/src/common/pcb_label.txt");
  cfg.input_size = raw.int_or("model.input_size", kDefaultInputSize);
  cfg.input_dir = raw.string_or("io.input_dir", "assets/datasets/pcb");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/pcb-defect-detector");
  cfg.score_threshold =
      static_cast<float>(raw.double_or("decode.score_threshold", kDefaultScoreThreshold));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", kDefaultNmsIou));
  cfg.max_detections = raw.int_or("decode.max_detections", kDefaultMaxDetections);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kDefaultTimeoutMs);
  cfg.num_runs = raw.int_or("runtime.num_runs", kDefaultNumRuns);
  cfg.queue_depth = raw.int_or("runtime.queue_depth", kDefaultQueueDepth);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.overlay = raw.bool_or("output.overlay", true);

  if (cli.score.has_value()) {
    cfg.score_threshold = *cli.score;
  }
  if (cli.nms.has_value()) {
    cfg.nms_iou = *cli.nms;
  }

  if (cfg.model_path.empty()) {
    throw std::runtime_error("model.path must be set to a compiled model package");
  }
  if (cfg.labels_path.empty()) {
    throw std::runtime_error("model.labels must point to a labels file");
  }
  if (cfg.input_size < 1) {
    throw std::runtime_error("model.input_size must be >= 1");
  }
  if (cfg.score_threshold < 0.0f || cfg.score_threshold > 1.0f) {
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
  if (cfg.queue_depth < 1) {
    throw std::runtime_error("runtime.queue_depth must be >= 1");
  }
  return cfg;
}

std::vector<std::string> load_labels(const fs::path& labels_path) {
  std::ifstream input(labels_path);
  if (!input.good()) {
    throw std::runtime_error("labels file does not exist: " + labels_path.string());
  }

  std::vector<std::string> labels;
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = sima_examples::trim_copy(line);
    if (!trimmed.empty()) {
      labels.push_back(trimmed);
    }
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + labels_path.string());
  }
  return labels;
}

bool is_image(const fs::path& path) {
  std::string ext = path.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> discover_images(const fs::path& input_dir) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    return labels[static_cast<size_t>(class_id)];
  }
  return "class_" + std::to_string(class_id);
}

// BGR colors, index-aligned with pcb_label.txt.
cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 6> kColors = {
      cv::Scalar(56, 56, 255), cv::Scalar(29, 178, 255), cv::Scalar(10, 249, 72),
      cv::Scalar(255, 194, 0), cv::Scalar(255, 0, 200),  cv::Scalar(49, 210, 207)};
  const size_t index = static_cast<size_t>(class_id > 0 ? class_id : 0) % kColors.size();
  return kColors[index];
}

// How a source frame was fitted into the square model input.
struct Letterbox {
  cv::Mat image;
  double scale = 1.0; // double, not float: the Python twin computes this in double.
  int pad_x = 0;
  int pad_y = 0;
};

// Aspect-preserving resize into a size x size canvas, centered on a grey pad.
Letterbox letterbox(const cv::Mat& src, int size) {
  Letterbox out;
  if (src.cols == size && src.rows == size) {
    out.image = src; // Already model-sized: no resample, so pixels stay exact.
    return out;
  }

  out.scale = std::min(static_cast<double>(size) / static_cast<double>(src.cols),
                       static_cast<double>(size) / static_cast<double>(src.rows));
  const int scaled_w = std::max(1, static_cast<int>(std::lround(src.cols * out.scale)));
  const int scaled_h = std::max(1, static_cast<int>(std::lround(src.rows * out.scale)));
  out.pad_x = (size - scaled_w) / 2;
  out.pad_y = (size - scaled_h) / 2;

  cv::Mat scaled;
  cv::resize(src, scaled, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
  out.image = cv::Mat(size, size, src.type(), cv::Scalar(kPadValue, kPadValue, kPadValue));
  scaled.copyTo(out.image(cv::Rect(out.pad_x, out.pad_y, scaled_w, scaled_h)));
  return out;
}

// Undo the letterbox so boxes land on the original frame.
std::vector<simaai::neat::Box> to_source_coordinates(const std::vector<simaai::neat::Box>& boxes,
                                                     const Letterbox& lb, int width, int height) {
  const auto unpad = [&](float value, int pad, int limit) {
    const double mapped = (static_cast<double>(value) - pad) / lb.scale;
    return static_cast<float>(std::clamp(mapped, 0.0, static_cast<double>(limit)));
  };

  std::vector<simaai::neat::Box> mapped;
  mapped.reserve(boxes.size());
  for (simaai::neat::Box box : boxes) {
    box.x1 = unpad(box.x1, lb.pad_x, width);
    box.x2 = unpad(box.x2, lb.pad_x, width);
    box.y1 = unpad(box.y1, lb.pad_y, height);
    box.y2 = unpad(box.y2, lb.pad_y, height);
    if (box.x2 > box.x1 && box.y2 > box.y1) {
      mapped.push_back(box);
    }
  }
  return mapped;
}

fs::path output_path_for(const fs::path& image_path, const fs::path& output_dir) {
  return output_dir / (image_path.stem().string() + ".png");
}

// Remove stale annotated images so a rerun cannot leave orphaned results.
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

void draw_boxes(cv::Mat& frame, const std::vector<simaai::neat::Box>& boxes,
                     const std::vector<std::string>& labels) {
  for (const auto& box : boxes) {
    const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(box.x1))));
    const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(box.y1))));
    const int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(box.x2))));
    const int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(box.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color = class_color(box.class_id);
    const std::string text = class_name(labels, box.class_id) + " " + cv::format("%.2f", box.score);
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_top = std::max(0, y1 - text_size.height - 4);
    cv::rectangle(frame, cv::Point(x1, label_top), cv::Point(x1 + text_size.width + 2, y1), color,
                  cv::FILLED);
    cv::putText(frame, text, cv::Point(x1 + 1, std::max(10, y1 - 3)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
  }
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

std::string format_counts(const std::map<std::string, int>& counts) {
  std::string out = "{";
  bool first = true;
  for (const auto& [name, count] : counts) {
    if (!first) {
      out += ", ";
    }
    first = false;
    out += name + ": " + std::to_string(count);
  }
  return out + "}";
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  CliOptions cli;
  try {
    cli = parse_cli(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  if (!fs::exists(cli.config_path)) {
    std::cerr << "Error: config file not found: " << cli.config_path << "\n";
    return 2;
  }

  Config cfg;
  std::vector<std::string> labels;
  try {
    cfg = load_config(cli);
    labels = load_labels(cfg.labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  if (cli.validate_only) {
    std::cout << "[validate] model=" << cfg.model_path << " classes=" << labels.size()
              << " input_size=" << cfg.input_size
              << " score_threshold=" << cv::format("%.2f", cfg.score_threshold)
              << " nms_iou=" << cv::format("%.2f", cfg.nms_iou)
              << " max_detections=" << cfg.max_detections << " timeout_ms=" << cfg.timeout_ms
              << " num_runs=" << cfg.num_runs << " queue_depth=" << cfg.queue_depth << "\n";
    std::cout << "[validate] configuration OK\n";
    return 0;
  }

  const fs::path input_dir = cfg.input_dir;
  const fs::path output_dir = cfg.output_dir;
  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }

  const std::vector<fs::path> images = discover_images(input_dir);
  if (images.empty()) {
    std::cerr << "No images found in " << input_dir << "\n";
    return 3;
  }

  fs::create_directories(output_dir);
  const int removed_outputs = clear_output_images(output_dir, input_dir);
  if (removed_outputs > 0) {
    std::cout << "Cleared " << removed_outputs << " stale output images\n";
  }
  std::cout << "Model: " << cfg.model_path << "\n";
  std::cout << "Found " << images.size() << " images in " << input_dir << "\n";

  try {
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26;
    model_opt.score_threshold = cfg.score_threshold;
    model_opt.nms_iou_threshold = cfg.nms_iou;
    model_opt.top_k = cfg.max_detections;
    model_opt.num_classes = static_cast<int>(labels.size());

    simaai::neat::Model model(cfg.model_path, model_opt);

    cv::Mat seed_bgr = cv::imread(images.front().string(), cv::IMREAD_COLOR);
    if (seed_bgr.empty()) {
      throw std::runtime_error("failed to read build seed image: " + images.front().string());
    }
    // Every frame is letterboxed to this shape, so the graph ingress caps never change.
    const Letterbox seed_lb = letterbox(seed_bgr, cfg.input_size);

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = cfg.queue_depth;
    run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;
    run_opt.preset = simaai::neat::RunPreset::Balanced;

    std::cout << "[BUILD] Building pipeline...\n";
    auto runner = model.build(std::vector<cv::Mat>{seed_lb.image},
                              simaai::neat::Model::RouteOptions{}, run_opt);
    std::cout << "[BUILD] Pipeline built\n";

    runner.run(std::vector<cv::Mat>{seed_lb.image}, cfg.timeout_ms);
    std::cout << "[WARMUP] done\n";

    std::vector<fs::path> all_images;
    all_images.reserve(images.size() * static_cast<size_t>(cfg.num_runs));
    for (int run = 0; run < cfg.num_runs; ++run) {
      all_images.insert(all_images.end(), images.begin(), images.end());
    }
    if (cfg.num_runs > 1) {
      std::cout << "Looping " << cfg.num_runs << "x over " << images.size() << " images ("
                << all_images.size() << " total)\n";
    }

    const auto pipeline_start = std::chrono::steady_clock::now();
    int processed = 0;
    int images_with_defects = 0;
    int total_defects = 0;
    std::map<std::string, int> per_class;

    for (const auto& image_path : all_images) {
      const auto image_start = std::chrono::steady_clock::now();

      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        std::cerr << "Skipping unreadable: " << image_path.filename().string() << "\n";
        continue;
      }
      const Letterbox lb = letterbox(bgr, cfg.input_size);

      const auto infer_start = std::chrono::steady_clock::now();
      simaai::neat::TensorList out = runner.run(std::vector<cv::Mat>{lb.image}, cfg.timeout_ms);
      const auto infer_end = std::chrono::steady_clock::now();

      // Detections arrive in letterboxed coordinates; draw them on the original frame.
      const std::vector<simaai::neat::Box> boxes = to_source_coordinates(
          decode_detections(out, cfg.input_size, cfg.input_size, cfg.max_detections), lb, bgr.cols,
          bgr.rows);

      std::map<std::string, int> counts;
      for (const auto& box : boxes) {
        ++counts[class_name(labels, box.class_id)];
      }
      for (const auto& [name, count] : counts) {
        per_class[name] += count;
      }

      const fs::path out_path = output_path_for(image_path, output_dir);
      if (cfg.overlay) {
        draw_boxes(bgr, boxes, labels);
        if (!cv::imwrite(out_path.string(), bgr)) {
          std::cerr << "Failed to write: " << out_path << "\n";
          continue;
        }
      }
      const auto image_end = std::chrono::steady_clock::now();

      ++processed;
      total_defects += static_cast<int>(boxes.size());
      if (!boxes.empty()) {
        ++images_with_defects;
      }

      std::cout << "[" << processed << "/" << all_images.size() << "] "
                << image_path.filename().string();
      if (cfg.overlay) {
        std::cout << " -> " << out_path.filename().string();
      }
      std::cout << " (" << boxes.size() << " defects) " << format_counts(counts) << "\n";

      if (cfg.profile) {
        using ms = std::chrono::duration<double, std::milli>;
        std::cout << "[PROFILE] " << image_path.filename().string()
                  << ": inference=" << cv::format("%.1f", ms(infer_end - infer_start).count())
                  << "ms overlay+save=" << cv::format("%.1f", ms(image_end - infer_end).count())
                  << "ms total=" << cv::format("%.1f", ms(image_end - image_start).count())
                  << "ms\n";
      }
    }

    runner.close();

    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - pipeline_start).count();
    std::cout << "Done: " << processed << "/" << all_images.size() << " images in "
              << cv::format("%.2f", elapsed) << "s | images_with_defects=" << images_with_defects
              << " total_defects=" << total_defects << "\n";
    if (!per_class.empty()) {
      std::cout << "Per-class totals: " << format_counts(per_class) << "\n";
    }
    return processed > 0 ? 0 : 4;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
