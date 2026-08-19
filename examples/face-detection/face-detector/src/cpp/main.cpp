/**
 * @example face-detector.cpp
 * Minimal RetinaFace pipeline (tensor input): run inference on a folder and decode
 * boxes/landmarks.
 *
 * Usage: face-detector [--config <path>]
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
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInferW = 640;
constexpr int kInferH = 640;

struct PreprocMeta {
  int orig_h = 0;
  int orig_w = 0;
  int padded_h = 0;
  int padded_w = 0;
  int pad_top = 0;
  int pad_left = 0;
};

struct Detection {
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  float score = 0;
  std::optional<std::array<float, 10>> landmarks;
};

// cfg_mnet essentials from apps/backbone_cfg.py
constexpr std::array<std::array<int, 2>, 3> kMinSizes = {{{16, 32}, {64, 128}, {256, 512}}};
constexpr std::array<int, 3> kSteps = {8, 16, 32};
constexpr std::array<float, 2> kVariance = {0.1f, 0.2f};
static void bgr_mean_subtract_inplace(cv::Mat& bgr_f32) {
  CV_Assert(bgr_f32.type() == CV_32FC3);
  const cv::Scalar mean(104.0, 117.0, 123.0);
  bgr_f32 -= mean;
}

static std::pair<cv::Mat, PreprocMeta> pad_to_aspect(const cv::Mat& img, int target_w,
                                                     int target_h) {
  if (img.empty())
    throw std::runtime_error("pad_to_aspect: empty image");
  const int orig_h = img.rows;
  const int orig_w = img.cols;
  const float aspect = static_cast<float>(orig_w) / static_cast<float>(orig_h);
  const float target = static_cast<float>(target_w) / static_cast<float>(target_h);

  PreprocMeta meta;
  meta.orig_h = orig_h;
  meta.orig_w = orig_w;

  cv::Mat padded;
  if (aspect > target) {
    const int new_h = static_cast<int>(orig_w / target);
    const int pad_top = (new_h - orig_h) / 2;
    const int pad_bottom = new_h - orig_h - pad_top;
    cv::copyMakeBorder(img, padded, pad_top, pad_bottom, 0, 0, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));
    meta.pad_top = pad_top;
    meta.pad_left = 0;
  } else {
    const int new_w = static_cast<int>(orig_h * target);
    const int pad_left = (new_w - orig_w) / 2;
    const int pad_right = new_w - orig_w - pad_left;
    cv::copyMakeBorder(img, padded, 0, 0, pad_left, pad_right, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));
    meta.pad_top = 0;
    meta.pad_left = pad_left;
  }

  meta.padded_h = padded.rows;
  meta.padded_w = padded.cols;
  return {padded, meta};
}

static simaai::neat::Tensor tensor_from_hwc_f32(const cv::Mat& hwc_f32) {
  if (hwc_f32.empty())
    throw std::runtime_error("tensor_from_hwc_f32: empty mat");
  if (hwc_f32.type() != CV_32FC3)
    throw std::runtime_error("tensor_from_hwc_f32: expected CV_32FC3");

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
  t.shape = {h, w, c};
  t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                     static_cast<int64_t>(c * sizeof(float)), static_cast<int64_t>(sizeof(float))};
  return t;
}

static std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value())
      throw std::runtime_error("tensor sample missing payload");
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

static std::vector<float> tensor_to_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32)
    throw std::runtime_error("expected Float32 tensor");
  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.size() % sizeof(float) != 0)
    throw std::runtime_error("tensor raw byte size is not float-aligned");
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

struct Prior {
  float cx, cy, sx, sy;
};

static std::vector<Prior> make_priors(int image_h, int image_w) {
  std::vector<Prior> priors;
  priors.reserve(16800);
  for (size_t k = 0; k < kSteps.size(); ++k) {
    const int step = kSteps[k];
    const int fh = static_cast<int>(std::ceil(static_cast<float>(image_h) / step));
    const int fw = static_cast<int>(std::ceil(static_cast<float>(image_w) / step));
    // Match RetinaFaceSpy.py: iterate H-major then W-major.
    for (int i = 0; i < fh; ++i) {
      for (int j = 0; j < fw; ++j) {
        for (int ms : kMinSizes[k]) {
          const float s_kx = static_cast<float>(ms) / static_cast<float>(image_w);
          const float s_ky = static_cast<float>(ms) / static_cast<float>(image_h);
          const float cx = (static_cast<float>(j) + 0.5f) * static_cast<float>(step) /
                           static_cast<float>(image_w);
          const float cy = (static_cast<float>(i) + 0.5f) * static_cast<float>(step) /
                           static_cast<float>(image_h);
          priors.push_back(Prior{cx, cy, s_kx, s_ky});
        }
      }
    }
  }
  return priors;
}

struct TensorDims4 {
  int64_t batch = 0;
  int64_t d1 = 0;
  int64_t d2 = 0;
  int64_t d3 = 0;
};

static TensorDims4 get_tensor_dims_4d_batch1(const simaai::neat::Tensor& t,
                                             const char* tensor_name) {
  if (t.shape.size() != 4) {
    throw std::runtime_error(std::string(tensor_name) + ": expected rank-4 tensor");
  }
  TensorDims4 dims;
  dims.batch = t.shape[0];
  dims.d1 = t.shape[1];
  dims.d2 = t.shape[2];
  dims.d3 = t.shape[3];
  if (dims.batch != 1) {
    throw std::runtime_error(std::string(tensor_name) + ": expected batch size 1");
  }
  if (dims.d1 <= 0 || dims.d2 <= 0 || dims.d3 <= 0) {
    throw std::runtime_error(std::string(tensor_name) + ": invalid tensor shape");
  }
  return dims;
}

static std::vector<float> transpose_0312_drop_batch(const std::vector<float>& in,
                                                    const TensorDims4& dims,
                                                    const char* tensor_name) {
  const size_t elems =
      static_cast<size_t>(dims.d1) * static_cast<size_t>(dims.d2) * static_cast<size_t>(dims.d3);
  if (in.size() != elems) {
    throw std::runtime_error(std::string(tensor_name) + ": unexpected raw tensor size");
  }
  std::vector<float> out(elems);
  for (int64_t a = 0; a < dims.d3; ++a) {
    for (int64_t b = 0; b < dims.d1; ++b) {
      for (int64_t c = 0; c < dims.d2; ++c) {
        const size_t src =
            (static_cast<size_t>(b) * static_cast<size_t>(dims.d2) + static_cast<size_t>(c)) *
                static_cast<size_t>(dims.d3) +
            static_cast<size_t>(a);
        const size_t dst =
            (static_cast<size_t>(a) * static_cast<size_t>(dims.d1) + static_cast<size_t>(b)) *
                static_cast<size_t>(dims.d2) +
            static_cast<size_t>(c);
        out[dst] = in[src];
      }
    }
  }
  return out;
}

static std::vector<float> reshape_rows_grouped(const std::vector<float>& transposed,
                                               int64_t channels, int group_size,
                                               const char* tensor_name) {
  if (channels <= 0 || channels % group_size != 0) {
    throw std::runtime_error(std::string(tensor_name) + ": channels not divisible by group size");
  }
  if (transposed.size() % static_cast<size_t>(channels) != 0) {
    throw std::runtime_error(std::string(tensor_name) + ": transposed size is not channel-aligned");
  }

  const size_t cells = transposed.size() / static_cast<size_t>(channels);
  std::vector<float> rows;
  rows.reserve((transposed.size() / static_cast<size_t>(group_size)) *
               static_cast<size_t>(group_size));

  for (size_t cell = 0; cell < cells; ++cell) {
    const size_t base = cell * static_cast<size_t>(channels);
    for (int64_t off = 0; off < channels; off += group_size) {
      for (int g = 0; g < group_size; ++g) {
        rows.push_back(transposed[base + static_cast<size_t>(off) + static_cast<size_t>(g)]);
      }
    }
  }
  return rows;
}

static void append_rows_python_style(const simaai::neat::Tensor& tensor, int group_size,
                                     const char* tensor_name, std::vector<float>& out_rows) {
  const TensorDims4 dims = get_tensor_dims_4d_batch1(tensor, tensor_name);
  const std::vector<float> raw = tensor_to_f32(tensor);
  const std::vector<float> transposed = transpose_0312_drop_batch(raw, dims, tensor_name);
  const std::vector<float> rows =
      reshape_rows_grouped(transposed, dims.d2, group_size, tensor_name);
  out_rows.insert(out_rows.end(), rows.begin(), rows.end());
}

static void decode_outputs(const std::vector<simaai::neat::Tensor>& tensors,
                           std::vector<Detection>& out, const PreprocMeta& meta, float conf_thr,
                           float nms_iou, int top_k, int keep_top_k, bool with_landmarks) {
  if (tensors.size() != 9) {
    throw std::runtime_error("expected exactly 9 tensors for RetinaFace (got " +
                             std::to_string(tensors.size()) + ")");
  }

  // Tensors arrive as [N,H,C,W]; transpose to [N,W,H,C], then group anchors in C.
  // Order is [land2, land1, land0, box2, box1, box0, cls2, cls1, cls0].
  // Concatenate as [level0(80), level1(40), level2(20)] before RetinaFace decode.
  std::vector<float> land_rows; // flattened rows of 10
  std::vector<float> box_rows;  // flattened rows of 4
  std::vector<float> cls_rows;  // flattened rows of 2 (logits)

  append_rows_python_style(tensors[2], /*group_size=*/10, "land0", land_rows);
  append_rows_python_style(tensors[1], /*group_size=*/10, "land1", land_rows);
  append_rows_python_style(tensors[0], /*group_size=*/10, "land2", land_rows);

  append_rows_python_style(tensors[5], /*group_size=*/4, "box0", box_rows);
  append_rows_python_style(tensors[4], /*group_size=*/4, "box1", box_rows);
  append_rows_python_style(tensors[3], /*group_size=*/4, "box2", box_rows);

  append_rows_python_style(tensors[8], /*group_size=*/2, "cls0", cls_rows);
  append_rows_python_style(tensors[7], /*group_size=*/2, "cls1", cls_rows);
  append_rows_python_style(tensors[6], /*group_size=*/2, "cls2", cls_rows);

  const std::vector<Prior> priors = make_priors(kInferH, kInferW);
  if (priors.size() != 16800) {
    throw std::runtime_error("unexpected priors count: " + std::to_string(priors.size()));
  }
  const size_t num_rows = priors.size();
  if (box_rows.size() != num_rows * 4) {
    throw std::runtime_error("box rows size mismatch vs priors");
  }
  if (cls_rows.size() != num_rows * 2) {
    throw std::runtime_error("class rows size mismatch vs priors");
  }
  if (land_rows.size() != num_rows * 10) {
    throw std::runtime_error("landmark rows size mismatch vs priors");
  }

  struct Cand {
    float x1, y1, x2, y2, score;
    std::optional<std::array<float, 10>> landm;
  };

  std::vector<Cand> cands;
  cands.reserve(2048);
  for (size_t i = 0; i < num_rows; ++i) {
    const float bg = cls_rows[i * 2 + 0];
    const float fg = cls_rows[i * 2 + 1];
    const float m = std::max(bg, fg);
    const float e0 = std::exp(bg - m);
    const float e1 = std::exp(fg - m);
    const float prob = e1 / (e0 + e1);
    if (!(prob > conf_thr)) {
      continue;
    }

    const Prior& p = priors[i];
    const float dx = box_rows[i * 4 + 0];
    const float dy = box_rows[i * 4 + 1];
    const float dw = box_rows[i * 4 + 2];
    const float dh = box_rows[i * 4 + 3];

    const float cx = p.cx + dx * kVariance[0] * p.sx;
    const float cy = p.cy + dy * kVariance[0] * p.sy;
    const float w = p.sx * std::exp(dw * kVariance[1]);
    const float h = p.sy * std::exp(dh * kVariance[1]);

    Cand cand;
    cand.x1 = cx - w / 2.0f;
    cand.y1 = cy - h / 2.0f;
    cand.x2 = cx + w / 2.0f;
    cand.y2 = cy + h / 2.0f;
    cand.score = prob;

    if (with_landmarks) {
      std::array<float, 10> lm{};
      for (int j = 0; j < 10; j += 2) {
        const float lx = land_rows[i * 10 + static_cast<size_t>(j)];
        const float ly = land_rows[i * 10 + static_cast<size_t>(j + 1)];
        lm[static_cast<size_t>(j)] = p.cx + lx * kVariance[0] * p.sx;
        lm[static_cast<size_t>(j + 1)] = p.cy + ly * kVariance[0] * p.sy;
      }
      cand.landm = lm;
    }
    cands.push_back(cand);
  }

  // Sort by score desc and keep top-k before NMS (like RetinaFaceSpy).
  std::sort(cands.begin(), cands.end(),
            [](const Cand& a, const Cand& b) { return a.score > b.score; });
  if (top_k > 0 && static_cast<size_t>(top_k) < cands.size()) {
    cands.resize(static_cast<size_t>(top_k));
  }

  // Match Python py_cpu_nms (+1 convention for area/intersection).
  std::vector<size_t> order(cands.size());
  for (size_t i = 0; i < cands.size(); ++i) {
    order[i] = i;
  }
  std::vector<size_t> keep_idx;
  keep_idx.reserve(cands.size());
  while (!order.empty()) {
    const size_t i = order.front();
    keep_idx.push_back(i);

    std::vector<size_t> next;
    next.reserve(order.size() > 0 ? order.size() - 1 : 0);
    const float area_i = (cands[i].x2 - cands[i].x1 + 1.0f) * (cands[i].y2 - cands[i].y1 + 1.0f);
    for (size_t p = 1; p < order.size(); ++p) {
      const size_t j = order[p];
      const float xx1 = std::max(cands[i].x1, cands[j].x1);
      const float yy1 = std::max(cands[i].y1, cands[j].y1);
      const float xx2 = std::min(cands[i].x2, cands[j].x2);
      const float yy2 = std::min(cands[i].y2, cands[j].y2);
      const float w = std::max(0.0f, xx2 - xx1 + 1.0f);
      const float h = std::max(0.0f, yy2 - yy1 + 1.0f);
      const float inter = w * h;
      const float area_j = (cands[j].x2 - cands[j].x1 + 1.0f) * (cands[j].y2 - cands[j].y1 + 1.0f);
      const float den = area_i + area_j - inter;
      const float ovr = den > 0.0f ? (inter / den) : 0.0f;
      if (ovr <= nms_iou) {
        next.push_back(j);
      }
    }
    order.swap(next);
  }

  std::vector<Cand> kept;
  kept.reserve(keep_idx.size());
  for (const size_t i : keep_idx) {
    kept.push_back(cands[i]);
  }

  // Keep top-k after NMS (like RetinaFaceSpy).
  if (keep_top_k > 0 && static_cast<size_t>(keep_top_k) < kept.size()) {
    kept.resize(static_cast<size_t>(keep_top_k));
  }

  // Scale boxes from normalized model coords -> padded -> original
  out.clear();
  out.reserve(kept.size());
  for (const auto& k : kept) {
    Detection d;
    d.score = k.score;
    d.x1 = k.x1 * static_cast<float>(meta.padded_w) - static_cast<float>(meta.pad_left);
    d.x2 = k.x2 * static_cast<float>(meta.padded_w) - static_cast<float>(meta.pad_left);
    d.y1 = k.y1 * static_cast<float>(meta.padded_h) - static_cast<float>(meta.pad_top);
    d.y2 = k.y2 * static_cast<float>(meta.padded_h) - static_cast<float>(meta.pad_top);

    if (k.landm.has_value()) {
      std::array<float, 10> lm = *k.landm;
      for (int i = 0; i < 10; i += 2) {
        lm[static_cast<size_t>(i)] =
            lm[static_cast<size_t>(i)] * static_cast<float>(meta.padded_w) -
            static_cast<float>(meta.pad_left);
        lm[static_cast<size_t>(i + 1)] =
            lm[static_cast<size_t>(i + 1)] * static_cast<float>(meta.padded_h) -
            static_cast<float>(meta.pad_top);
      }
      d.landmarks = lm;
    }
    out.push_back(d);
  }
}

static void draw_detections(cv::Mat& bgr, const std::vector<Detection>& dets, int max_draw) {
  const size_t limit =
      (max_draw > 0) ? std::min(dets.size(), static_cast<size_t>(max_draw)) : dets.size();
  for (size_t i = 0; i < limit; ++i) {
    const auto& d = dets[i];
    if (!std::isfinite(d.x1) || !std::isfinite(d.y1) || !std::isfinite(d.x2) ||
        !std::isfinite(d.y2)) {
      throw std::runtime_error("non-finite face detection box at index " + std::to_string(i));
    }
    if (!std::isfinite(d.score)) {
      throw std::runtime_error("non-finite face detection score at index " + std::to_string(i));
    }
    const cv::Point p1(static_cast<int>(std::lround(d.x1)), static_cast<int>(std::lround(d.y1)));
    const cv::Point p2(static_cast<int>(std::lround(d.x2)), static_cast<int>(std::lround(d.y2)));
    cv::rectangle(bgr, p1, p2, cv::Scalar(0, 255, 0), 2);
    cv::putText(bgr, std::to_string(d.score).substr(0, 5), cv::Point(p1.x, std::max(0, p1.y - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    if (d.landmarks.has_value()) {
      const auto& lm = *d.landmarks;
      for (int j = 0; j < 10; j += 2) {
        if (!std::isfinite(lm[static_cast<size_t>(j)]) ||
            !std::isfinite(lm[static_cast<size_t>(j + 1)])) {
          throw std::runtime_error("non-finite face landmark at detection index " +
                                   std::to_string(i));
        }
        cv::circle(bgr,
                   cv::Point(static_cast<int>(std::lround(lm[static_cast<size_t>(j)])),
                             static_cast<int>(std::lround(lm[static_cast<size_t>(j + 1)]))),
                   2, cv::Scalar(0, 0, 255), -1);
      }
    }
  }
}

struct Config {
  fs::path input_dir;
  std::string model = "models/retinaface_mobilenet25_mod_0_mpk.tar.gz";
  fs::path output_dir;
  float conf = 0.4f;
  float nms = 0.9f;
  int top_k = 5000;
  int keep_top_k = 750;
  int max_draw = 50;
  bool landmarks = true;
  bool profile = false;
  int num_runs = 100;
  int timeout_ms = 20000;
};

static bool is_image_file(const fs::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

static std::vector<fs::path> image_paths_in_dir(const fs::path& input_dir) {
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image_file(entry.path())) {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
  return images;
}

static Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model = raw.string_or("model.path", "models/retinaface_mobilenet25_mod_0_mpk.tar.gz");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/datasets/coco");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/face-detector");
  cfg.conf = static_cast<float>(raw.double_or("decode.confidence_threshold", 0.4));
  cfg.nms = static_cast<float>(raw.double_or("decode.nms_iou", 0.9));
  cfg.top_k = raw.int_or("decode.top_k", 5000);
  cfg.keep_top_k = raw.int_or("decode.keep_top_k", 750);
  cfg.max_draw = raw.int_or("decode.max_draw", 50);
  cfg.landmarks = raw.bool_or("decode.landmarks", true);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.num_runs = raw.int_or("runtime.num_runs", 100);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", 20000);
  if (cfg.conf < 0.0f || cfg.conf > 1.0f) {
    throw std::runtime_error("decode.confidence_threshold must be in [0.0, 1.0]");
  }
  if (cfg.nms < 0.0f || cfg.nms > 1.0f) {
    throw std::runtime_error("decode.nms_iou must be in [0.0, 1.0]");
  }
  if (cfg.top_k < 1 || cfg.keep_top_k < 1 || cfg.max_draw < 0) {
    throw std::runtime_error("decode top/max values must be valid");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  return cfg;
}

static Config parse_config(int argc, char** argv) {
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

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Config args = parse_config(argc, argv);
    if (!fs::is_directory(args.input_dir)) {
      throw std::runtime_error("input_dir does not exist: " + args.input_dir.string());
    }
    if (!fs::exists(args.model)) {
      throw std::runtime_error("model does not exist: " + args.model);
    }

    const std::vector<fs::path> image_paths = image_paths_in_dir(args.input_dir);
    if (image_paths.empty()) {
      throw std::runtime_error("no images found in: " + args.input_dir.string());
    }

    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Tensor;
    model_opt.preprocess.input_max_width = kInferW;
    model_opt.preprocess.input_max_height = kInferH;
    model_opt.preprocess.input_max_depth = 3;

    simaai::neat::Model model(args.model, model_opt);

    simaai::neat::Graph graph;
    graph.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    graph.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
    graph.add(simaai::neat::nodes::groups::MLA(model));
    graph.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
    graph.add(simaai::neat::nodes::Output());

    cv::Mat dummy(kInferH, kInferW, CV_32FC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy_t = tensor_from_hwc_f32(dummy);
    auto run = graph.build(simaai::neat::TensorList{dummy_t});

    std::vector<Detection> dets;

    if (args.profile) {
      const fs::path& image_path = image_paths.front();
      cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr_u8.empty()) {
        throw std::runtime_error("failed to read image: " + image_path.string());
      }
      cv::Mat bgr_f32;
      bgr_u8.convertTo(bgr_f32, CV_32FC3);
      bgr_mean_subtract_inplace(bgr_f32);
      auto [padded, meta] = pad_to_aspect(bgr_f32, kInferW, kInferH);
      cv::Mat resized;
      cv::resize(padded, resized, cv::Size(kInferW, kInferH), 0, 0, cv::INTER_LINEAR);
      simaai::neat::Tensor input_t = tensor_from_hwc_f32(resized);

      const int runs = std::max(1, args.num_runs);
      std::vector<double> graph_times;
      std::vector<double> post_times;
      graph_times.reserve(runs);
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

        const auto t2 = std::chrono::steady_clock::now();
        const std::vector<simaai::neat::Tensor> out_tensors = collect_tensors(*out_sample);
        dets.clear();
        decode_outputs(out_tensors, dets, meta, args.conf, args.nms, args.top_k, args.keep_top_k,
                       args.landmarks);
        const auto t3 = std::chrono::steady_clock::now();

        const std::chrono::duration<double> dt_graph = t1 - t0;
        const std::chrono::duration<double> dt_post = t3 - t2;
        graph_times.push_back(dt_graph.count());
        post_times.push_back(dt_post.count());
      }

      const auto stats = [](const std::vector<double>& v) {
        struct S {
          double mean;
          double min;
          double max;
          double sum;
        };
        S s{0.0, v.empty() ? 0.0 : v[0], v.empty() ? 0.0 : v[0], 0.0};
        for (double x : v) {
          s.sum += x;
          s.min = std::min(s.min, x);
          s.max = std::max(s.max, x);
        }
        if (!v.empty()) {
          s.mean = s.sum / static_cast<double>(v.size());
        }
        return s;
      };

      const auto graph_stats = stats(graph_times);
      const auto post = stats(post_times);

      const double runs_d = static_cast<double>(graph_times.size());
      const double fps_graph = runs_d / graph_stats.sum;
      const double fps_post = runs_d / post.sum;
      const double total_sum = graph_stats.sum + post.sum;
      const double fps_overall = runs_d / total_sum;

      std::cout << "Profiling over " << graph_times.size() << " runs (image='"
                << image_path.string() << "'):\n";
      std::cout << "  Graph run (push+pull): mean=" << graph_stats.mean
                << "s, min=" << graph_stats.min << "s, max=" << graph_stats.max
                << "s, FPS=" << fps_graph << "\n";
      std::cout << "  Postprocessing (decode+NMS): mean=" << post.mean << "s, min=" << post.min
                << "s, max=" << post.max << "s, FPS=" << fps_post << "\n";
      std::cout << "  Overall (graph + post): mean=" << (total_sum / runs_d)
                << "s, min=" << (graph_stats.min + post.min)
                << "s, max=" << (graph_stats.max + post.max) << "s, FPS=" << fps_overall << "\n";

      std::cout << "Last run detections: " << dets.size() << "\n";
      for (size_t i = 0; i < std::min<size_t>(dets.size(), 20); ++i) {
        const auto& d = dets[i];
        std::cout << "  [" << i << "] score=" << d.score << " box=[" << d.x1 << "," << d.y1 << ","
                  << d.x2 << "," << d.y2 << "]\n";
      }
    } else {
      fs::create_directories(args.output_dir);
      int processed = 0;
      for (const fs::path& image_path : image_paths) {
        cv::Mat bgr_u8 = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr_u8.empty()) {
          throw std::runtime_error("failed to read image: " + image_path.string());
        }
        cv::Mat bgr_f32;
        bgr_u8.convertTo(bgr_f32, CV_32FC3);
        bgr_mean_subtract_inplace(bgr_f32);
        auto [padded, meta] = pad_to_aspect(bgr_f32, kInferW, kInferH);
        cv::Mat resized;
        cv::resize(padded, resized, cv::Size(kInferW, kInferH), 0, 0, cv::INTER_LINEAR);
        simaai::neat::Tensor input_t = tensor_from_hwc_f32(resized);

        if (!run.push(simaai::neat::TensorList{input_t})) {
          throw std::runtime_error("run.push failed");
        }

        auto out_sample = run.pull(args.timeout_ms);
        if (!out_sample.has_value()) {
          throw std::runtime_error("run.pull returned no sample");
        }

        const std::vector<simaai::neat::Tensor> out_tensors = collect_tensors(*out_sample);
        std::cout << "Model produced " << out_tensors.size() << " tensor(s)\n";
        for (size_t i = 0; i < out_tensors.size(); ++i) {
          const auto& t = out_tensors[i];
          std::cout << "  [" << i << "] dtype=" << static_cast<int>(t.dtype) << " shape=[";
          for (size_t d = 0; d < t.shape.size(); ++d) {
            std::cout << t.shape[d] << (d + 1 < t.shape.size() ? "," : "");
          }
          std::cout << "]\n";
        }

        decode_outputs(out_tensors, dets, meta, args.conf, args.nms, args.top_k, args.keep_top_k,
                       args.landmarks);

        std::cout << "Detections: " << dets.size() << "\n";
        for (size_t i = 0; i < std::min<size_t>(dets.size(), 20); ++i) {
          const auto& d = dets[i];
          std::cout << "  [" << i << "] score=" << d.score << " box=[" << d.x1 << "," << d.y1 << ","
                    << d.x2 << "," << d.y2 << "]\n";
        }

        cv::Mat overlay = bgr_u8.clone();
        draw_detections(overlay, dets, args.max_draw);
        const fs::path output_path =
            args.output_dir / (image_path.stem().string() + "_retinaface.png");
        if (!cv::imwrite(output_path.string(), overlay)) {
          throw std::runtime_error("failed to write: " + output_path.string());
        }
        ++processed;
        std::cout << "[" << processed << "/" << image_paths.size() << "] "
                  << image_path.filename().string() << ": " << dets.size() << " detections -> "
                  << output_path.filename().string() << "\n";
      }
      std::cout << "Done: " << processed << " images processed\n";
    }

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
