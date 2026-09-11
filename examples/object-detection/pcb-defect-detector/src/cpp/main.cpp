/**
 * @example pcb-defect-detector.cpp
 * PCB defect detection: run every image in a folder through a custom YOLO26n pack.
 *
 * Images of any resolution go straight to the model: Core letterboxes them to the
 * packaged input size on device, and decode_bbox_tensor returns boxes already in
 * source-image coordinates, so the application performs no geometry of its own.
 * One annotated image is written per input image.
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
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kDefaultScoreThreshold = 0.25f;
constexpr float kDefaultNmsIou = 0.45f;
constexpr int kDefaultMaxDetections = 300;
constexpr int kDefaultTimeoutMs = 8000;
constexpr int kDefaultNumRuns = 1;
constexpr int kDefaultQueueDepth = 8;
// Floor for the one-off priming run of the graph seed.
constexpr int kWarmupTimeoutMs = 30000;
// Appended to the stem of every annotated image.
constexpr const char* kOutputTag = "_pcb";
// Ingress capacity of the graph; boards up to this size share one graph.
constexpr int kDefaultInputMaxWidth = 3840;
constexpr int kDefaultInputMaxHeight = 2160;

struct Config {
  std::string model_path;
  std::string labels_path;
  int input_max_width = kDefaultInputMaxWidth;
  int input_max_height = kDefaultInputMaxHeight;
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
  std::size_t consumed = 0;
  float parsed = 0.0F;
  try {
    parsed = std::stof(value, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error(flag + " requires a number, got: " + value);
  }
  if (consumed != value.size() || !std::isfinite(parsed)) {
    throw std::runtime_error(flag + " requires a number, got: " + value);
  }
  return parsed;
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
  cfg.input_max_width = raw.int_or("model.input_max_width", kDefaultInputMaxWidth);
  cfg.input_max_height = raw.int_or("model.input_max_height", kDefaultInputMaxHeight);
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
  if (cfg.input_max_width < 1) {
    throw std::runtime_error("model.input_max_width must be >= 1");
  }
  if (cfg.input_max_height < 1) {
    throw std::runtime_error("model.input_max_height must be >= 1");
  }
  if (!std::isfinite(cfg.score_threshold) || cfg.score_threshold < 0.0f ||
      cfg.score_threshold > 1.0f) {
    throw std::runtime_error("decode.score_threshold must be in [0.0, 1.0]");
  }
  if (!std::isfinite(cfg.nms_iou) || cfg.nms_iou < 0.0f || cfg.nms_iou > 1.0f) {
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
  // Fold by absolute value, matching the Python twin. Widened first so that
  // negating INT_MIN is not signed overflow.
  const long long folded = class_id < 0 ? -static_cast<long long>(class_id) : class_id;
  const size_t index = static_cast<size_t>(folded) % kColors.size();
  return kColors[index];
}

// Annotated-output path for one input image.
fs::path output_path_for(const fs::path& image_path, const fs::path& output_dir) {
  return output_dir /
         (image_path.stem().string() + kOutputTag + image_path.extension().string());
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

// Decode the model's BBOX payload into source-image coordinates.
//
// Throws if the response is not a usable detection result; returns an empty
// vector when a valid response found nothing.
std::vector<simaai::neat::Box> decode_detections(const simaai::neat::TensorList& outputs,
                                                 int image_width, int image_height,
                                                 int max_detections, float score_threshold) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no detection tensors");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("expected one BBOX tensor from model-managed BoxDecode, got " +
                             std::to_string(outputs.size()));
  }

  // An untagged tensor would otherwise parse permissively into zero boxes and
  // read as a clean board, so require the BBOX tag explicitly.
  std::string detection_format;
  try {
    detection_format = simaai::neat::read_detection_format(outputs.front());
  } catch (const std::exception&) {  // a non-detection tensor has no format tag
    detection_format.clear();
  }
  if (detection_format.empty() || !simaai::neat::detection_format_is_bbox(detection_format)) {
    throw std::runtime_error(
        "model returned no BBOX detection tensor (format: '" +
        (detection_format.empty() ? std::string("<none>") : detection_format) +
        "'); the model package must expose YOLO26 BoxDecode output");
  }

  // strict=true rejects a truncated or over-long payload instead of decoding it
  // as zero boxes. expected_topk is 0 because strict also throws on it; the cap
  // is applied below.
  simaai::neat::BoxDecodeResult decoded;
  try {
    decoded = simaai::neat::decode_bbox_tensor(outputs.front(), image_width, image_height,
                                               /*expected_topk=*/0, /*strict=*/true);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("malformed BBOX payload: ") + error.what());
  }

  // Same post-decode filtering as the Python twin, so both report an identical
  // defect count for the same image.
  std::vector<simaai::neat::Box> boxes;
  boxes.reserve(decoded.boxes.size());
  for (const auto& box : decoded.boxes) {
    if (box.score < score_threshold) {
      continue;
    }
    if (box.x2 <= box.x1 || box.y2 <= box.y1) {
      continue;
    }
    boxes.push_back(box);
    if (static_cast<int>(boxes.size()) >= max_detections) {
      break;
    }
  }
  return boxes;
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
    return 2;
  }

  std::error_code config_error;
  if (!fs::is_regular_file(cli.config_path, config_error) || config_error) {
    std::cerr << "Error: config file not found: " << cli.config_path.string() << "\n";
    return 2;
  }

  Config cfg;
  std::vector<std::string> labels;
  try {
    cfg = load_config(cli);
    labels = load_labels(cfg.labels_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }

  if (cli.validate_only) {
    std::cout << "[validate] model=" << cfg.model_path << " classes=" << labels.size()
              << " score_threshold=" << cv::format("%.2f", cfg.score_threshold)
              << " nms_iou=" << cv::format("%.2f", cfg.nms_iou)
              << " max_detections=" << cfg.max_detections << " timeout_ms=" << cfg.timeout_ms
              << " num_runs=" << cfg.num_runs << " queue_depth=" << cfg.queue_depth << "\n";
    std::cout << "[validate] configuration OK\n";
    return 0;
  }

  const fs::path input_dir = cfg.input_dir;
  const fs::path output_dir = cfg.output_dir;
  std::error_code input_error;
  if (!fs::is_directory(input_dir, input_error) || input_error) {
    std::cerr << "Input directory does not exist: " << input_dir.string() << "\n";
    return 2;
  }

  // Sharing one directory corrupts the run: an annotated image written for one
  // board is re-read as a source for a later one, and a rerun discovers the
  // previous outputs as new inputs.
  std::error_code same_error;
  if (fs::weakly_canonical(input_dir, same_error) ==
      fs::weakly_canonical(output_dir, same_error)) {
    std::cerr << "Error: io.output_dir must differ from io.input_dir; annotated images written "
                 "beside their sources are re-read as inputs.\n";
    return 2;
  }

  std::vector<fs::path> images;
  try {
    images = discover_images(input_dir);
  } catch (const std::exception& e) {
    std::cerr << "Error: cannot read " << input_dir.string() << ": " << e.what() << "\n";
    return 2;
  }
  if (images.empty()) {
    std::cerr << "No images found in " << input_dir.string() << "\n";
    return 3;
  }

  std::error_code create_error;
  fs::create_directories(output_dir, create_error);
  if (create_error) {
    std::cerr << "Error: cannot create " << output_dir.string() << ": " << create_error.message()
              << "\n";
    return 2;
  }

  // Declared outside the try so the catch below can still report what was
  // completed and what had already failed when a systemic error aborts the run.
  std::vector<std::string> failed_images;
  std::map<std::string, int> per_class;
  std::size_t all_images_total = 0;
  int processed = 0;
  int images_with_defects = 0;
  int total_defects = 0;
  // The image in flight when a systemic failure aborts the batch, so the summary
  // names it rather than reporting failed=0.
  std::string aborted_on;
  std::cout << "Model: " << cfg.model_path << "\n";
  std::cout << "Found " << images.size() << " images in " << input_dir.string() << "\n";

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
    // Resize stays at the model package default: letterbox, grey padding.
    model_opt.preprocess.input_max_width = cfg.input_max_width;
    model_opt.preprocess.input_max_height = cfg.input_max_height;

    simaai::neat::Model model(cfg.model_path, model_opt);

    simaai::neat::RunOptions run_opt;
    run_opt.queue_depth = cfg.queue_depth;
    // Realtime applies new input caps on the first frame. The other presets wait
    // for a second frame at the same size, which a folder of images never sends.
    run_opt.preset = simaai::neat::RunPreset::Realtime;

    // The build seed fixes the graph's input caps, so it is a frame at the
    // configured capacity rather than any particular board.
    const cv::Mat graph_seed(cfg.input_max_height, cfg.input_max_width, CV_8UC3,
                             cv::Scalar(114, 114, 114));

    simaai::neat::Model::Runner runner =
        model.build(std::vector<cv::Mat>{graph_seed}, simaai::neat::Model::RouteOptions{}, run_opt);
    // One-off graph settling and first-touch allocation, which runtime.timeout_ms
    // does not cover.
    runner.run(std::vector<cv::Mat>{graph_seed}, std::max(cfg.timeout_ms, kWarmupTimeoutMs));

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

    all_images_total = all_images.size();

    for (const auto& image_path : all_images) {
      aborted_on = image_path.filename().string();
      const auto image_start = std::chrono::steady_clock::now();

      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        // An input we were asked to inspect and could not: record it so the
        // summary and the exit code stay honest.
        std::cerr << "Failed to read: " << image_path.filename().string() << "\n";
        failed_images.push_back(image_path.filename().string());
        continue;
      }
      if (bgr.cols > cfg.input_max_width || bgr.rows > cfg.input_max_height) {
        std::cerr << "Image " << bgr.cols << "x" << bgr.rows
                  << " exceeds model.input_max_width/height (" << cfg.input_max_width << "x"
                  << cfg.input_max_height << "): " << image_path.filename().string() << "\n";
        failed_images.push_back(image_path.filename().string());
        continue;
      }

      const auto infer_start = std::chrono::steady_clock::now();
      const simaai::neat::TensorList out =
          runner.run(std::vector<cv::Mat>{bgr}, cfg.timeout_ms);
      const auto infer_end = std::chrono::steady_clock::now();

      // Throws rather than reporting a clean board when the model produced no
      // usable detection output; a valid empty result yields an empty vector.
      const std::vector<simaai::neat::Box> boxes =
          decode_detections(out, bgr.cols, bgr.rows, cfg.max_detections, cfg.score_threshold);

      std::map<std::string, int> counts;
      for (const auto& box : boxes) {
        ++counts[class_name(labels, box.class_id)];
      }

      const fs::path out_path = output_path_for(image_path, output_dir);
      if (cfg.overlay) {
        draw_boxes(bgr, boxes, labels);
        bool written = false;
        try {
          written = cv::imwrite(out_path.string(), bgr);
        } catch (const cv::Exception&) {  // encoder failure, not a batch-wide problem
          written = false;
        }
        if (!written) {
          std::cerr << "Failed to write: " << out_path.string() << "\n";
          failed_images.push_back(image_path.filename().string());
          continue;
        }
      }
      const auto image_end = std::chrono::steady_clock::now();

      // Merged only once the image is fully done, so "Per-class totals" can never
      // describe an image that processed and total_defects exclude.
      for (const auto& [name, count] : counts) {
        per_class[name] += count;
      }

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
      aborted_on.clear();
    }

    runner.close();

    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - pipeline_start).count();
    std::cout << "Done: " << processed << "/" << all_images.size() << " images in "
              << cv::format("%.2f", elapsed) << "s | images_with_defects=" << images_with_defects
              << " total_defects=" << total_defects << " failed=" << failed_images.size() << "\n";
    if (!per_class.empty()) {
      std::cout << "Per-class totals: " << format_counts(per_class) << "\n";
    }

    // A batch that skipped inputs is not a success.
    if (!failed_images.empty()) {
      std::cerr << "Error: " << failed_images.size() << " of " << all_images.size()
                << " image(s) could not be processed or saved:";
      for (std::size_t i = 0; i < failed_images.size() && i < 5; ++i) {
        std::cerr << (i ? ", " : " ") << failed_images[i];
      }
      std::cerr << (failed_images.size() > 5 ? ", ...\n" : "\n");
      return 4;
    }
    if (processed != static_cast<int>(all_images.size())) {  // defensive
      std::cerr << "Error: processed " << processed << " of " << all_images.size() << " images\n";
      return 4;
    }
    return 0;
  } catch (const std::exception& e) {
    // A systemic failure repeats on every image, so stop the batch but still
    // report what completed and what failed.
    std::cerr << "Error: " << e.what() << "\n";
    if (!aborted_on.empty()) {
      failed_images.push_back(aborted_on);
    }
    std::cerr << "Aborted after " << processed << "/"
              << (all_images_total > 0 ? all_images_total : images.size())
              << " images | images_with_defects=" << images_with_defects
              << " total_defects=" << total_defects
              << " failed=" << failed_images.size() << "\n";
    if (!failed_images.empty()) {
      std::cerr << "Previously failed:";
      for (std::size_t i = 0; i < failed_images.size() && i < 5; ++i) {
        std::cerr << (i ? ", " : " ") << failed_images[i];
      }
      std::cerr << (failed_images.size() > 5 ? ", ...\n" : "\n");
    }
    return 4;
  }
}
