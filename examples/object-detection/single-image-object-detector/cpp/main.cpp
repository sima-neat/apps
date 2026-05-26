/**
 * @example single-image-object-detector.cpp
 * Minimal DETR single-image detection pipeline using tensor input and raw tensor decode.
 *
 * Usage: single-image-object-detector [--config <path>]
 */
#include "neat.h"
#include "support/runtime/example_utils.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kModelW = 1333;
constexpr int kModelH = 800;
constexpr int kDefaultTimeoutMs = 5000;
constexpr int kPersonClassId = 1;

constexpr std::array<std::string_view, 91> kDetrCocoLabels = {
    "N/A",          "person",         "bicycle",
    "car",          "motorcycle",     "airplane",
    "bus",          "train",          "truck",
    "boat",         "traffic light",  "fire hydrant",
    "N/A",          "stop sign",      "parking meter",
    "bench",        "bird",           "cat",
    "dog",          "horse",          "sheep",
    "cow",          "elephant",       "bear",
    "zebra",        "giraffe",        "N/A",
    "backpack",     "umbrella",       "N/A",
    "N/A",          "handbag",        "tie",
    "suitcase",     "frisbee",        "skis",
    "snowboard",    "sports ball",    "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard",    "tennis racket",  "bottle",
    "N/A",          "wine glass",     "cup",
    "fork",         "knife",          "spoon",
    "bowl",         "banana",         "apple",
    "sandwich",     "orange",         "broccoli",
    "carrot",       "hot dog",        "pizza",
    "donut",        "cake",           "chair",
    "couch",        "potted plant",   "bed",
    "N/A",          "dining table",   "N/A",
    "N/A",          "toilet",         "N/A",
    "tv",           "laptop",         "mouse",
    "remote",       "keyboard",       "cell phone",
    "microwave",    "oven",           "toaster",
    "sink",         "refrigerator",   "N/A",
    "book",         "clock",          "vase",
    "scissors",     "teddy bear",     "hair drier",
    "toothbrush",
};

struct PreprocMeta {
  int orig_h = 0;
  int orig_w = 0;
  int resized_h = 0;
  int resized_w = 0;
  int pad_top = 0;
  int pad_left = 0;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
};

struct Detection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = -1;
};

struct Config {
  fs::path image;
  std::string model = "assets/models/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz";
  fs::path output;
  float conf = 0.5f;
  int max_draw = 50;
  bool person_only = false;
  bool profile = false;
  int num_runs = 100;
  int timeout_ms = kDefaultTimeoutMs;
};

struct Tensor2D {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;
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

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or(
      "model.path", "assets/models/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz");
  cfg.image = raw.string_or("io.image", "assets/test_images/input.jpg");
  cfg.output = raw.string_or("io.output", "sandbox/single-image-object-detector/output.png");
  cfg.conf = static_cast<float>(raw.double_or("decode.confidence_threshold", 0.5));
  cfg.max_draw = raw.int_or("decode.max_draw", 50);
  cfg.person_only = raw.bool_or("decode.person_only", false);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.num_runs = raw.int_or("runtime.num_runs", 100);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kDefaultTimeoutMs);
  if (cfg.conf < 0.0f || cfg.conf > 1.0f) {
    throw std::runtime_error("decode.confidence_threshold must be in [0.0, 1.0]");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
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

std::pair<cv::Mat, PreprocMeta> preprocess_to_tensor_input(const cv::Mat& bgr_u8) {
  if (bgr_u8.empty()) {
    throw std::runtime_error("preprocess_to_tensor_input: empty image");
  }

  PreprocMeta meta;
  meta.orig_h = bgr_u8.rows;
  meta.orig_w = bgr_u8.cols;

  const float scale = std::min(static_cast<float>(kModelW) / static_cast<float>(meta.orig_w),
                               static_cast<float>(kModelH) / static_cast<float>(meta.orig_h));
  meta.resized_w =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(meta.orig_w) * scale)));
  meta.resized_h =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(meta.orig_h) * scale)));
  meta.scale_x = static_cast<float>(meta.resized_w) / static_cast<float>(meta.orig_w);
  meta.scale_y = static_cast<float>(meta.resized_h) / static_cast<float>(meta.orig_h);
  meta.pad_left = (kModelW - meta.resized_w) / 2;
  meta.pad_top = (kModelH - meta.resized_h) / 2;

  cv::Mat resized;
  cv::resize(bgr_u8, resized, cv::Size(meta.resized_w, meta.resized_h), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat canvas_bgr(kModelH, kModelW, CV_8UC3, cv::Scalar(0, 0, 0));
  resized.copyTo(canvas_bgr(cv::Rect(meta.pad_left, meta.pad_top, meta.resized_w, meta.resized_h)));

  cv::Mat canvas_rgb;
  cv::cvtColor(canvas_bgr, canvas_rgb, cv::COLOR_BGR2RGB);

  cv::Mat rgb_f32;
  canvas_rgb.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);
  const cv::Scalar mean(0.485, 0.456, 0.406);
  const cv::Scalar stdv(0.229, 0.224, 0.225);

  cv::Mat normalized(kModelH, kModelW, CV_32FC3);
  for (int y = 0; y < rgb_f32.rows; ++y) {
    const cv::Vec3f* src = rgb_f32.ptr<cv::Vec3f>(y);
    cv::Vec3f* dst = normalized.ptr<cv::Vec3f>(y);
    for (int x = 0; x < rgb_f32.cols; ++x) {
      dst[x][0] = (src[x][0] - static_cast<float>(mean[0])) / static_cast<float>(stdv[0]);
      dst[x][1] = (src[x][1] - static_cast<float>(mean[1])) / static_cast<float>(stdv[1]);
      dst[x][2] = (src[x][2] - static_cast<float>(mean[2])) / static_cast<float>(stdv[2]);
    }
  }

  return {normalized, meta};
}

simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32) {
  if (hwc_f32.empty()) {
    throw std::runtime_error("tensor_from_hwc_f32: empty mat");
  }
  if (hwc_f32.type() != CV_32FC3) {
    throw std::runtime_error("tensor_from_hwc_f32: expected CV_32FC3");
  }

  const int h = hwc_f32.rows;
  const int w = hwc_f32.cols;
  const int c = 3;
  const size_t elems = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  const size_t bytes = elems * sizeof(float);

  std::vector<float> data(elems);
  std::memcpy(data.data(), hwc_f32.ptr<float>(), bytes);

  simaai::neat::Tensor t =
      simaai::neat::Tensor::from_vector(data, {h, w, c}, simaai::neat::TensorMemory::EV74);
  t.layout = simaai::neat::TensorLayout::HWC;
  t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                     static_cast<int64_t>(c * sizeof(float)), static_cast<int64_t>(sizeof(float))};
  return t;
}

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }
  if (sample.kind == simaai::neat::SampleKind::TensorSet) {
    return sample.tensors;
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    for (const auto& field : sample.fields) {
      auto child = collect_tensors(field);
      out.insert(out.end(), child.begin(), child.end());
    }
    return out;
  }
  throw std::runtime_error("unexpected sample kind");
}

std::vector<float> tensor_to_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.size() % sizeof(float) != 0) {
    throw std::runtime_error("tensor raw byte size is not float-aligned");
  }
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

Tensor2D tensor_to_2d_rows(const simaai::neat::Tensor& t, const char* name) {
  if (t.shape.empty()) {
    throw std::runtime_error(std::string(name) + ": empty tensor shape");
  }
  if (t.shape.size() < 2) {
    throw std::runtime_error(std::string(name) + ": expected rank >= 2 tensor");
  }

  int cols = static_cast<int>(t.shape.back());
  if (cols <= 0) {
    throw std::runtime_error(std::string(name) + ": invalid trailing dimension");
  }

  int64_t rows64 = 1;
  for (size_t i = 0; i + 1 < t.shape.size(); ++i) {
    rows64 *= t.shape[i];
  }
  if (rows64 <= 0) {
    throw std::runtime_error(std::string(name) + ": invalid row count");
  }

  std::vector<float> data = tensor_to_f32(t);
  const int64_t expected = rows64 * static_cast<int64_t>(cols);
  if (static_cast<int64_t>(data.size()) != expected) {
    throw std::runtime_error(std::string(name) + ": unexpected dense tensor size");
  }

  return Tensor2D{static_cast<int>(rows64), cols, std::move(data)};
}

std::pair<Tensor2D, Tensor2D>
extract_logits_and_boxes(const std::vector<simaai::neat::Tensor>& tensors) {
  std::optional<Tensor2D> logits;
  std::optional<Tensor2D> boxes;

  for (size_t i = 0; i < tensors.size(); ++i) {
    Tensor2D flat = tensor_to_2d_rows(tensors[i], ("tensor_" + std::to_string(i)).c_str());
    if (flat.cols == 4) {
      boxes = std::move(flat);
    } else if (flat.cols > 4) {
      logits = std::move(flat);
    }
  }

  if (!logits.has_value() || !boxes.has_value()) {
    throw std::runtime_error("expected DETR logits and box tensors in model output");
  }
  if (logits->rows != boxes->rows) {
    throw std::runtime_error("DETR logits and boxes row counts do not match");
  }
  if (logits->cols < 2) {
    throw std::runtime_error("DETR logits tensor must include at least one class and background");
  }

  return {*logits, *boxes};
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

std::string class_name(int class_id) {
  if (class_id < 0 || static_cast<size_t>(class_id) >= kDetrCocoLabels.size()) {
    return "class_" + std::to_string(class_id);
  }
  const std::string_view label = kDetrCocoLabels[static_cast<size_t>(class_id)];
  if (label == "N/A") {
    return "class_" + std::to_string(class_id);
  }
  return std::string(label);
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

std::vector<Detection> decode_detr_outputs(const Tensor2D& logits, const Tensor2D& boxes,
                                           const PreprocMeta& meta, float conf_threshold,
                                           bool person_only) {
  const int foreground_classes =
      std::min(logits.cols - 1, static_cast<int>(kDetrCocoLabels.size()));
  std::vector<Detection> dets;
  dets.reserve(static_cast<size_t>(logits.rows));

  for (int row = 0; row < logits.rows; ++row) {
    const float* logit_row =
        logits.data.data() + static_cast<size_t>(row) * static_cast<size_t>(logits.cols);
    const float* box_row = boxes.data.data() + static_cast<size_t>(row) * 4U;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < logits.cols; ++c) {
      max_logit = std::max(max_logit, logit_row[c]);
    }

    float denom = 0.0f;
    for (int c = 0; c < logits.cols; ++c) {
      denom += std::exp(logit_row[c] - max_logit);
    }
    if (denom <= 0.0f) {
      continue;
    }

    int best_class = -1;
    float best_score = 0.0f;
    for (int c = 0; c < foreground_classes; ++c) {
      const float prob = std::exp(logit_row[c] - max_logit) / denom;
      if (prob > best_score) {
        best_score = prob;
        best_class = c;
      }
    }
    if (best_score <= conf_threshold || best_class < 0) {
      continue;
    }
    if (person_only && best_class != kPersonClassId) {
      continue;
    }

    const float x_c = sigmoid(box_row[0]);
    const float y_c = sigmoid(box_row[1]);
    const float w = sigmoid(box_row[2]);
    const float h = sigmoid(box_row[3]);

    float x1 = (x_c - 0.5f * w) * static_cast<float>(kModelW);
    float y1 = (y_c - 0.5f * h) * static_cast<float>(kModelH);
    float x2 = (x_c + 0.5f * w) * static_cast<float>(kModelW);
    float y2 = (y_c + 0.5f * h) * static_cast<float>(kModelH);

    x1 = (x1 - static_cast<float>(meta.pad_left)) / meta.scale_x;
    x2 = (x2 - static_cast<float>(meta.pad_left)) / meta.scale_x;
    y1 = (y1 - static_cast<float>(meta.pad_top)) / meta.scale_y;
    y2 = (y2 - static_cast<float>(meta.pad_top)) / meta.scale_y;

    x1 = std::clamp(x1, 0.0f, static_cast<float>(meta.orig_w));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(meta.orig_w));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(meta.orig_h));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(meta.orig_h));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    dets.push_back(Detection{x1, y1, x2, y2, best_score, best_class});
  }

  std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  return dets;
}

void draw_detections(cv::Mat& bgr, const std::vector<Detection>& dets, int max_draw) {
  const size_t limit =
      max_draw > 0 ? std::min(dets.size(), static_cast<size_t>(max_draw)) : dets.size();
  for (size_t i = 0; i < limit; ++i) {
    const auto& d = dets[i];
    const int x1 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(d.x1))));
    const int y1 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(d.y1))));
    const int x2 = std::max(0, std::min(bgr.cols - 1, static_cast<int>(std::lround(d.x2))));
    const int y2 = std::max(0, std::min(bgr.rows - 1, static_cast<int>(std::lround(d.y2))));
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    const cv::Scalar color = class_color(d.class_id);
    const std::string text = class_name(d.class_id) + " " + cv::format("%.2f", d.score);
    cv::rectangle(bgr, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
    cv::putText(bgr, text, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                2, cv::LINE_AA);
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config args = parse_config(argc, argv);
    if (!fs::exists(args.image)) {
      throw std::runtime_error("image does not exist: " + args.image.string());
    }
    if (!fs::exists(args.model)) {
      throw std::runtime_error("model does not exist: " + args.model);
    }

    cv::Mat bgr_u8 = cv::imread(args.image.string(), cv::IMREAD_COLOR);
    if (bgr_u8.empty()) {
      throw std::runtime_error("failed to read image: " + args.image.string());
    }

    auto [input_f32, meta] = preprocess_to_tensor_input(bgr_u8);

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Tensor;
    model_opt.preprocess.input_max_width = kModelW;
    model_opt.preprocess.input_max_height = kModelH;
    model_opt.preprocess.input_max_depth = 3;

    simaai::neat::Model model(args.model, model_opt);

    simaai::neat::Session session;
    session.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    session.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
    session.add(simaai::neat::nodes::groups::MLA(model));
    session.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
    session.add(simaai::neat::nodes::Output());

    cv::Mat dummy(kModelH, kModelW, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    auto run = session.build(simaai::neat::TensorList{tensor_from_hwc_f32(dummy)},
                             simaai::neat::RunMode::Async);
    simaai::neat::Tensor input_t = tensor_from_hwc_f32(input_f32);

    if (args.profile) {
      const int runs = std::max(1, args.num_runs);
      std::vector<double> session_times;
      std::vector<double> post_times;
      std::vector<Detection> last_dets;
      session_times.reserve(runs);
      post_times.reserve(runs);

      for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run.push(simaai::neat::TensorList{input_t})) {
          throw std::runtime_error("run.push failed during profiling");
        }
        auto out_sample = run.pull(args.timeout_ms);
        const auto t1 = std::chrono::steady_clock::now();
        if (!out_sample.has_value()) {
          throw std::runtime_error("run.pull returned no sample during profiling");
        }

        const std::vector<simaai::neat::Tensor> tensors = collect_tensors(*out_sample);
        const auto [logits, boxes] = extract_logits_and_boxes(tensors);
        const auto t2 = std::chrono::steady_clock::now();
        last_dets = decode_detr_outputs(logits, boxes, meta, args.conf, args.person_only);
        const auto t3 = std::chrono::steady_clock::now();

        session_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        post_times.push_back(std::chrono::duration<double>(t3 - t2).count());
      }

      const Stats sess = compute_stats(session_times);
      const Stats post = compute_stats(post_times);
      const double runs_d = static_cast<double>(session_times.size());
      const double total_sum = sess.sum + post.sum;

      std::cout << "Profiling over " << session_times.size() << " runs (image='"
                << args.image.string() << "'):\n";
      std::cout << "  Session (push+pull): mean=" << sess.mean << "s, min=" << sess.min
                << "s, max=" << sess.max << "s, FPS=" << (runs_d / sess.sum) << "\n";
      std::cout << "  Postprocessing (decode+NMS): mean=" << post.mean << "s, min=" << post.min
                << "s, max=" << post.max << "s, FPS=" << (runs_d / post.sum) << "\n";
      std::cout << "  Overall (session + post): mean=" << (total_sum / runs_d)
                << "s, min=" << (sess.min + post.min) << "s, max=" << (sess.max + post.max)
                << "s, FPS=" << (runs_d / total_sum) << "\n";
      std::cout << "Last run detections: " << last_dets.size() << "\n";
      for (size_t i = 0; i < std::min<size_t>(last_dets.size(), 20); ++i) {
        const auto& d = last_dets[i];
        std::cout << "  [" << i << "] class=" << class_name(d.class_id) << "(" << d.class_id
                  << ") score=" << d.score << " box=[" << d.x1 << "," << d.y1 << "," << d.x2 << ","
                  << d.y2 << "]\n";
      }
      run.close();
      return 0;
    }

    const auto t0_total = std::chrono::steady_clock::now();
    const auto t0_infer = std::chrono::steady_clock::now();
    if (!run.push(simaai::neat::TensorList{input_t})) {
      throw std::runtime_error("run.push failed");
    }

    auto out_sample = run.pull(args.timeout_ms);
    const auto t1_infer = std::chrono::steady_clock::now();
    if (!out_sample.has_value()) {
      throw std::runtime_error("run.pull returned no sample");
    }

    const std::vector<simaai::neat::Tensor> tensors = collect_tensors(*out_sample);
    std::cout << "Model produced " << tensors.size() << " tensor(s)\n";
    for (size_t i = 0; i < tensors.size(); ++i) {
      std::cout << "  [" << i << "] shape=[";
      for (size_t d = 0; d < tensors[i].shape.size(); ++d) {
        std::cout << tensors[i].shape[d] << (d + 1 < tensors[i].shape.size() ? "," : "");
      }
      std::cout << "] dtype=" << static_cast<int>(tensors[i].dtype) << "\n";
    }

    const auto [logits, boxes] = extract_logits_and_boxes(tensors);
    const std::vector<Detection> dets =
        decode_detr_outputs(logits, boxes, meta, args.conf, args.person_only);

    std::cout << "Detections: " << dets.size() << "\n";
    for (size_t i = 0; i < std::min<size_t>(dets.size(), 20); ++i) {
      const auto& d = dets[i];
      std::cout << "  [" << i << "] class=" << class_name(d.class_id) << "(" << d.class_id
                << ") score=" << d.score << " box=[" << d.x1 << "," << d.y1 << "," << d.x2 << ","
                << d.y2 << "]\n";
    }

    if (!args.output.empty()) {
      fs::path parent = args.output.parent_path();
      if (!parent.empty()) {
        fs::create_directories(parent);
      }
      cv::Mat overlay = bgr_u8.clone();
      draw_detections(overlay, dets, args.max_draw);
      if (!cv::imwrite(args.output.string(), overlay)) {
        throw std::runtime_error("failed to write: " + args.output.string());
      }
      std::cout << "Wrote annotated image: " << args.output << "\n";
    }

    const auto t1_total = std::chrono::steady_clock::now();
    if (args.profile) {
      const std::chrono::duration<double, std::milli> total_ms = t1_total - t0_total;
      const std::chrono::duration<double, std::milli> infer_ms = t1_infer - t0_infer;
      std::cout << "Timing:\n";
      std::cout << "  End-to-end: " << total_ms.count() << " ms\n";
      std::cout << "  Model inference: " << infer_ms.count() << " ms\n";
    }

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
