/**
 * @example ssd-mobilenet-object-detector.cpp
 * SSD (TF COCO) folder detection via the model-managed BoxDecodeType::Ssd pipeline. Runs any
 * of the four supported SSD models: SSD300 and SSD-MobileNet v1/v2 (300x300) or v3 (320x320,
 * set model.frame=320). Defaults to SSD-MobileNetV2.
 *
 * Usage: ssd-mobilenet-object-detector [--config <path>]
 */
#include "neat.h"
#include "aggregate_suppression.h"
#include "output_paths.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/config_utils.h"

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace neat = simaai::neat;
using ssd_mobilenet::clear_output_images;
using ssd_mobilenet::output_stem;
using ssd_mobilenet::suppress_aggregate_boxes;

namespace {

// Default model frame. SSD300 and SSD-MobileNet v1/v2 are 300x300; v3 is 320x320.
// Override via `model.frame` in the config to match the model pack.
constexpr int kDefaultModelSize = 300;
constexpr int kNumClasses = 91; // index 0 = background, 1..90 = COCO ids.
constexpr int kDefaultTimeoutMs = 20000;
constexpr const char* kDefaultLabelsPath =
    "examples/object-detection/ssd-mobilenet-object-detector/src/common/"
    "coco_labels.txt";

struct Config {
  fs::path config_file;
  std::string model = "models/ssd_mobilenet_v2_heads_mpk.tar.gz";
  fs::path labels;
  fs::path input_dir;
  fs::path output_dir;
  fs::path detections_json;
  int model_frame = kDefaultModelSize; // 300 for SSD300/v1/v2, 320 for v3.
  float score_threshold = 0.55f;
  float nms_iou = 0.60f;
  int max_detections = 100;
  ssd_mobilenet::AggregateSuppressionOptions aggregate_suppression;
  int timeout_ms = kDefaultTimeoutMs;
  int num_runs = 1;
  bool profile = false;
  bool overlay = true;
};

struct Stats {
  double mean = 0.0;
  double min = 0.0;
  double max = 0.0;
  double sum = 0.0;
};

Stats compute_stats(const std::vector<double>& values) {
  Stats s;
  if (values.empty()) {
    return s;
  }
  s.min = values.front();
  s.max = values.front();
  for (double x : values) {
    s.sum += x;
    s.min = std::min(s.min, x);
    s.max = std::max(s.max, x);
  }
  s.mean = s.sum / static_cast<double>(values.size());
  return s;
}

bool is_image_file(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

std::vector<fs::path> image_paths_in_dir(const fs::path& input_dir) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image_file(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

// Directory holding the running executable, or empty if it cannot be determined.
fs::path executable_dir() {
  std::error_code ec;
  const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
  if (ec || exe.empty()) {
    return {};
  }
  return exe.parent_path();
}

// Resolve a labels asset: configured path if present, else the shipped src/common copy. All
// fallbacks are cwd-independent (executable location, then the compiled absolute source dir).
fs::path resolve_asset(const std::string& configured, const char* default_path,
                       const char* default_name) {
  if (!configured.empty() && fs::exists(configured)) {
    return configured;
  }
  // Substitute the packaged copy only for the empty/default reference; a missing custom
  // path is returned as-is so the caller fails instead of loading different labels.
  if (!configured.empty() &&
      fs::path(configured).lexically_normal() != fs::path(default_path).lexically_normal()) {
    return configured;
  }
  std::vector<fs::path> candidates;
  const fs::path exe_dir = executable_dir();
  if (!exe_dir.empty()) {
    candidates.push_back(exe_dir / ".." / ".." / "common" / default_name); // packaged pre-built/
    candidates.push_back(exe_dir / ".." / "common" / default_name);        // same-dir binary
  }
#ifdef SIMANEAT_APPS_EXAMPLE_SOURCE_ABS_DIR
  // Absolute build-time source dir: works for the top-level build tree run from any cwd.
  candidates.push_back(fs::path(SIMANEAT_APPS_EXAMPLE_SOURCE_ABS_DIR) / ".." / "common" /
                       default_name);
#endif
  for (const fs::path& candidate : candidates) {
    if (fs::exists(candidate)) {
      return candidate;
    }
  }
  return configured;
}

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.config_file = path;
  cfg.model = raw.string_or("model.path", "models/ssd_mobilenet_v2_heads_mpk.tar.gz");
  cfg.model_frame = raw.int_or("model.frame", kDefaultModelSize);
  cfg.labels =
      resolve_asset(raw.string_or("model.labels", ""), kDefaultLabelsPath, "coco_labels.txt");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/datasets/coco");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/ssd-mobilenet-object-detector");
  cfg.detections_json = raw.string_or("io.detections_json", "");
  cfg.score_threshold = static_cast<float>(raw.double_or("decode.score_threshold", 0.55));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", 0.60));
  cfg.max_detections = raw.int_or("decode.max_detections", 100);
  cfg.aggregate_suppression.enabled = raw.bool_or("postprocess.aggregate_suppression", true);
  cfg.aggregate_suppression.min_parent_area_fraction =
      static_cast<float>(raw.double_or("postprocess.min_parent_area_fraction", 0.20));
  cfg.aggregate_suppression.min_child_containment =
      static_cast<float>(raw.double_or("postprocess.min_child_containment", 0.90));
  cfg.aggregate_suppression.max_child_area_ratio =
      static_cast<float>(raw.double_or("postprocess.max_child_area_ratio", 0.25));
  cfg.aggregate_suppression.min_children = raw.int_or("postprocess.min_children", 2);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kDefaultTimeoutMs);
  cfg.num_runs = raw.int_or("runtime.num_runs", 1);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.overlay = raw.bool_or("output.overlay", true);
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
  if (!std::isfinite(cfg.aggregate_suppression.min_parent_area_fraction) ||
      cfg.aggregate_suppression.min_parent_area_fraction < 0.0f ||
      cfg.aggregate_suppression.min_parent_area_fraction > 1.0f) {
    throw std::runtime_error("postprocess.min_parent_area_fraction must be in [0.0, 1.0]");
  }
  if (!std::isfinite(cfg.aggregate_suppression.min_child_containment) ||
      cfg.aggregate_suppression.min_child_containment <= 0.0f ||
      cfg.aggregate_suppression.min_child_containment > 1.0f) {
    throw std::runtime_error("postprocess.min_child_containment must be in (0.0, 1.0]");
  }
  if (!std::isfinite(cfg.aggregate_suppression.max_child_area_ratio) ||
      cfg.aggregate_suppression.max_child_area_ratio <= 0.0f ||
      cfg.aggregate_suppression.max_child_area_ratio > 1.0f) {
    throw std::runtime_error("postprocess.max_child_area_ratio must be in (0.0, 1.0]");
  }
  if (cfg.aggregate_suppression.min_children < 2) {
    throw std::runtime_error("postprocess.min_children must be >= 2");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  if (cfg.model_frame != 300 && cfg.model_frame != 320) {
    throw std::runtime_error("model.frame must be 300 (SSD300/MobileNet v1/v2) or 320 (v3), got " +
                             std::to_string(cfg.model_frame));
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

std::vector<std::string> load_labels(const fs::path& path) {
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error("labels file does not exist: " + path.string());
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open labels file: " + path.string());
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    labels.push_back(line);
  }
  if (labels.empty()) {
    throw std::runtime_error("labels file is empty: " + path.string());
  }
  return labels;
}

std::string class_name(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size()) {
    const std::string& name = labels[static_cast<size_t>(class_id)];
    if (!name.empty() && name != "N/A") {
      return name;
    }
  }
  return "class_" + std::to_string(class_id);
}

cv::Scalar class_color(int class_id) {
  static const std::array<cv::Scalar, 8> kColors = {
      cv::Scalar(0, 255, 0),   cv::Scalar(255, 0, 0),   cv::Scalar(0, 0, 255),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
      cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
  };
  const size_t idx = static_cast<size_t>(class_id >= 0 ? class_id : -class_id) % kColors.size();
  return kColors[idx];
}

// Model-managed SSD decode: stretch to the model frame, normalize to [-1, 1]; BoxDecode
// owns the rest. Frame is 300 for SSD300/v1/v2, 320 for v3 (set via model.frame).
neat::Model::Options make_model_options(const Config& cfg) {
  neat::Model::Options opt;
  opt.preprocess.kind = neat::InputKind::Image;
  opt.preprocess.enable = neat::AutoFlag::On;
  // STRETCH, not the default Letterbox: these TF models train on a direct square resize.
  opt.preprocess.resize.enable = neat::AutoFlag::On;
  opt.preprocess.resize.mode = neat::ResizeMode::Stretch;
  opt.preprocess.resize.width = cfg.model_frame;
  opt.preprocess.resize.height = cfg.model_frame;
  // Model input range is [-1, 1] = (pixel / 127.5 - 1); the CVU computes (pixel/255 - mean)/stddev.
  opt.preprocess.normalize.enable = neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.5f, 0.5f, 0.5f};
  opt.preprocess.normalize.stddev = {0.5f, 0.5f, 0.5f};
  opt.preprocess.normalize.has_explicit_stats = true;
  opt.preprocess.color_convert.input_format = neat::PreprocessColorFormat::BGR;
  opt.preprocess.color_convert.output_format = neat::PreprocessColorFormat::RGB;
  opt.decode_type = neat::BoxDecodeType::Ssd;
  opt.num_classes = kNumClasses;
  opt.score_threshold = cfg.score_threshold;
  opt.nms_iou_threshold = cfg.nms_iou;
  opt.top_k = cfg.max_detections;
  return opt;
}

std::vector<neat::Box> parse_detections(const neat::TensorList& outputs, int image_width,
                                        int image_height, const Config& cfg) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no detection tensors");
  }
  if (outputs.size() != 1) {
    throw std::runtime_error("expected one BBOX tensor from model-managed BoxDecode, got " +
                             std::to_string(outputs.size()));
  }
  const std::vector<neat::Box> boxes =
      neat::decode_bbox_tensor(outputs.front(), image_width, image_height, cfg.max_detections,
                               /*strict=*/false)
          .boxes;
  return suppress_aggregate_boxes(boxes, image_width, image_height, cfg.aggregate_suppression);
}

void draw_detections(cv::Mat& bgr, const std::vector<neat::Box>& boxes,
                     const std::vector<std::string>& labels) {
  for (const neat::Box& b : boxes) {
    const int x1 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(b.x1))));
    const int y1 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(b.y1))));
    const int x2 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(b.x2))));
    const int y2 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(b.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }
    const cv::Scalar color = class_color(b.class_id);
    const std::string text = class_name(labels, b.class_id) + " " + cv::format("%.2f", b.score);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    cv::putText(bgr, text, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                2, cv::LINE_AA);
  }
}

// Machine-readable detection record, written when io.detections_json is set.
nlohmann::json detections_record(const fs::path& image_path, const cv::Mat& bgr,
                                 const std::vector<neat::Box>& boxes,
                                 const std::vector<std::string>& labels) {
  nlohmann::json entry;
  entry["image"] = image_path.filename().string();
  entry["width"] = bgr.cols;
  entry["height"] = bgr.rows;
  entry["detections"] = nlohmann::json::array();
  for (const neat::Box& b : boxes) {
    entry["detections"].push_back({{"class_id", b.class_id},
                                   {"label", class_name(labels, b.class_id)},
                                   {"score", b.score},
                                   {"box", {b.x1, b.y1, b.x2, b.y2}}});
  }
  return entry;
}

void write_detections_json(const fs::path& path, const nlohmann::json& images) {
  if (path.has_parent_path()) {
    fs::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write detections json: " + path.string());
  }
  out << nlohmann::json{{"images", images}}.dump(2) << "\n";
  // Catch a failed write/flush (e.g. full disk) before reporting success on a truncated file.
  out.flush();
  if (!out) {
    throw std::runtime_error("failed to write detections json: " + path.string());
  }
}

bool paths_alias(const fs::path& lhs, const fs::path& rhs) {
  if (fs::weakly_canonical(lhs) == fs::weakly_canonical(rhs)) {
    return true;
  }
  std::error_code error;
  const bool equivalent = fs::equivalent(lhs, rhs, error);
  return !error && equivalent;
}

void validate_detections_report_path(const Config& cfg, const std::vector<fs::path>& image_paths) {
  if (cfg.detections_json.empty()) {
    return;
  }
  const fs::path report_path = fs::weakly_canonical(cfg.detections_json);
  const std::array<fs::path, 3> consumed_files = {cfg.config_file, fs::path(cfg.model), cfg.labels};
  for (const fs::path& consumed : consumed_files) {
    if (paths_alias(consumed, report_path)) {
      throw std::runtime_error("io.detections_json must not overwrite a consumed input: " +
                               consumed.string());
    }
  }
  for (const fs::path& image_path : image_paths) {
    if (paths_alias(image_path, report_path)) {
      throw std::runtime_error("io.detections_json must not overwrite an input image: " +
                               image_path.string());
    }
    if (cfg.overlay &&
        paths_alias(cfg.output_dir / (output_stem(image_path) + ".png"), report_path)) {
      throw std::runtime_error("io.detections_json must not overwrite a generated overlay: " +
                               report_path.string());
    }
  }
  if (is_image_file(report_path) && paths_alias(report_path.parent_path(), cfg.input_dir)) {
    throw std::runtime_error(
        "io.detections_json must not use an image filename inside io.input_dir: " +
        report_path.string());
  }
}

void validate_overlay_paths(const Config& cfg, const std::vector<fs::path>& image_paths) {
  if (!cfg.overlay) {
    return;
  }
  const std::array<fs::path, 3> fixed_inputs = {cfg.config_file, fs::path(cfg.model), cfg.labels};
  for (const fs::path& image_path : image_paths) {
    const fs::path overlay_path =
        fs::weakly_canonical(cfg.output_dir / (output_stem(image_path) + ".png"));
    for (const fs::path& consumed : fixed_inputs) {
      if (paths_alias(consumed, overlay_path)) {
        throw std::runtime_error("generated overlay must not overwrite a consumed input: " +
                                 overlay_path.string());
      }
    }
    for (const fs::path& consumed : image_paths) {
      if (paths_alias(consumed, overlay_path)) {
        throw std::runtime_error("generated overlay must not overwrite a consumed input: " +
                                 overlay_path.string());
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config cfg = parse_config(argc, argv);
    if (!fs::is_directory(cfg.input_dir)) {
      throw std::runtime_error("input_dir does not exist: " + cfg.input_dir.string());
    }
    // Only overlay runs write into output_dir, so only they must not alias input_dir.
    if (!cfg.profile && cfg.overlay && paths_alias(cfg.output_dir, cfg.input_dir)) {
      throw std::runtime_error("io.output_dir must differ from io.input_dir");
    }
    if (!fs::exists(cfg.model)) {
      throw std::runtime_error("model does not exist: " + cfg.model);
    }

    const std::vector<std::string> labels = load_labels(cfg.labels);

    const std::vector<fs::path> image_paths = image_paths_in_dir(cfg.input_dir);
    if (image_paths.empty()) {
      throw std::runtime_error("no images found in: " + cfg.input_dir.string());
    }
    if (!cfg.profile) {
      validate_overlay_paths(cfg, image_paths);
      validate_detections_report_path(cfg, image_paths);
    }

    cv::Mat seed_bgr = cv::imread(image_paths.front().string(), cv::IMREAD_COLOR);
    if (seed_bgr.empty()) {
      throw std::runtime_error("failed to read build seed image: " + image_paths.front().string());
    }

    // Model load and graph build resolve the SSD recipe and validate the model frame; a
    // recipe/frame mismatch surfaces here. Add a hint pointing at the two config knobs.
    neat::Model model(cfg.model, make_model_options(cfg));
    neat::Model::Runner run = [&] {
      try {
        return model.build(std::vector<cv::Mat>{seed_bgr}, neat::Model::RouteOptions{});
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) + "\n  hint: check model.path (" +
                                 cfg.model + ") and model.frame (" +
                                 std::to_string(cfg.model_frame) +
                                 "); use 300 for SSD300/MobileNet v1/v2, 320 for v3.");
      }
    }();
    run.run(std::vector<cv::Mat>{seed_bgr}, cfg.timeout_ms);

    if (cfg.profile) {
      const fs::path& image_path = image_paths.front();
      cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr_u8.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }

      const int runs = cfg.num_runs; // validated >= 1 in load_config
      std::vector<double> infer_times;
      std::vector<double> parse_times;
      std::vector<neat::Box> last_boxes;

      for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        neat::TensorList out = run.run(std::vector<cv::Mat>{bgr_u8}, cfg.timeout_ms);
        const auto t1 = std::chrono::steady_clock::now();
        last_boxes = parse_detections(out, bgr_u8.cols, bgr_u8.rows, cfg);
        const auto t2 = std::chrono::steady_clock::now();
        infer_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        parse_times.push_back(std::chrono::duration<double>(t2 - t1).count());
      }

      const Stats infer = compute_stats(infer_times);
      const Stats parse = compute_stats(parse_times);
      const double runs_d = static_cast<double>(infer_times.size());
      std::cout << "Profiling over " << infer_times.size() << " runs (image='"
                << image_path.string() << "'):\n";
      std::cout << "  Pipeline run (preprocess+infer+decode): mean=" << infer.mean
                << "s, min=" << infer.min << "s, max=" << infer.max
                << "s, FPS=" << (runs_d / infer.sum) << "\n";
      std::cout << "  Output parsing: mean=" << parse.mean << "s, min=" << parse.min
                << "s, max=" << parse.max << "s\n";
      std::cout << "Last run detections: " << last_boxes.size() << "\n";
      for (size_t i = 0; i < std::min<size_t>(last_boxes.size(), 20); ++i) {
        const auto& b = last_boxes[i];
        std::cout << "  [" << i << "] class=" << class_name(labels, b.class_id) << "(" << b.class_id
                  << ") score=" << b.score << " box=[" << b.x1 << "," << b.y1 << "," << b.x2 << ","
                  << b.y2 << "]\n";
      }
      run.close();
      return 0;
    }

    // Only overlay runs touch output_dir, so a JSON-only run leaves it alone.
    if (cfg.overlay) {
      fs::create_directories(cfg.output_dir);
      const int removed_outputs = clear_output_images(cfg.output_dir, image_paths);
      if (removed_outputs > 0) {
        std::cout << "Cleared " << removed_outputs << " stale output images\n";
      }
    }

    nlohmann::json records = nlohmann::json::array();
    int processed = 0;
    for (const fs::path& image_path : image_paths) {
      cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr_u8.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }

      neat::TensorList out = run.run(std::vector<cv::Mat>{bgr_u8}, cfg.timeout_ms);
      const std::vector<neat::Box> boxes = parse_detections(out, bgr_u8.cols, bgr_u8.rows, cfg);

      std::string output_name;
      if (cfg.overlay) {
        cv::Mat annotated = bgr_u8.clone();
        draw_detections(annotated, boxes, labels);
        const fs::path output_path = cfg.output_dir / (output_stem(image_path) + ".png");
        // A failed overlay write is a run failure, not a per-image skip.
        if (!cv::imwrite(output_path.string(), annotated)) {
          throw std::runtime_error("failed to write: " + output_path.string());
        }
        output_name = output_path.filename().string();
      }

      if (!cfg.detections_json.empty()) {
        records.push_back(detections_record(image_path, bgr_u8, boxes, labels));
      }
      ++processed;
      if (cfg.overlay) {
        std::cout << "[" << processed << "/" << image_paths.size() << "] "
                  << image_path.filename().string() << " -> " << output_name << " (" << boxes.size()
                  << " detections)\n";
      } else {
        std::cout << "[" << processed << "/" << image_paths.size() << "] "
                  << image_path.filename().string() << " (" << boxes.size() << " detections)\n";
      }
    }
    if (!cfg.detections_json.empty()) {
      write_detections_json(cfg.detections_json, records);
      std::cout << "Wrote detections: " << cfg.detections_json.string() << "\n";
    }
    std::cout << "Done: " << processed << " images processed\n";

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
