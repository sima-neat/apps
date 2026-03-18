/**
 * @example retinaface-face-detection.cpp
 * Minimal RetinaFace pipeline (tensor input): run inference on one image and decode boxes/landmarks.
 *
 * Usage:
 *   retinaface-face-detection <image_path> [--model <model.tar.gz>] [--output <out.jpg>]
 *                         [--conf <thr>] [--nms <iou>] [--no-landmarks]
 */
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

static std::pair<cv::Mat, PreprocMeta> pad_to_aspect(const cv::Mat& img, int target_w, int target_h) {
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
    cv::copyMakeBorder(img, padded, pad_top, pad_bottom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    meta.pad_top = pad_top;
    meta.pad_left = 0;
  } else {
    const int new_w = static_cast<int>(orig_h * target);
    const int pad_left = (new_w - orig_w) / 2;
    const int pad_right = new_w - orig_w - pad_left;
    cv::copyMakeBorder(img, padded, 0, 0, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
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

  auto buf = std::make_shared<std::vector<float>>(elems);
  std::memcpy(buf->data(), hwc_f32.ptr<float>(), bytes);

  simaai::neat::Tensor t;
  t.storage = simaai::neat::make_cpu_external_storage(buf->data(), bytes, buf, /*read_only=*/true);
  t.dtype = simaai::neat::TensorDType::Float32;
  t.layout = simaai::neat::TensorLayout::HWC;
  t.shape = {h, w, c};
  t.strides_bytes = {static_cast<int64_t>(w * c * sizeof(float)),
                     static_cast<int64_t>(c * sizeof(float)),
                     static_cast<int64_t>(sizeof(float))};
  return t;
}

static std::vector<simaai::neat::Tensor> tensors_from_sample(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value())
      throw std::runtime_error("tensor sample missing payload");
    return {*sample.tensor};
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    for (const auto& field : sample.fields) {
      auto child = tensors_from_sample(field);
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
          const float cx = (static_cast<float>(j) + 0.5f) * static_cast<float>(step) / static_cast<float>(image_w);
          const float cy = (static_cast<float>(i) + 0.5f) * static_cast<float>(step) / static_cast<float>(image_h);
          priors.push_back(Prior{cx, cy, s_kx, s_ky});
        }
      }
    }
  }
  return priors;
}

static inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Transpose from (W, C, H) to (H, W, C) into a contiguous vector.
// Your tensors print as [1, W, C, H] (e.g. [1,40,4,40]). Python does transpose(0,3,1,2),
// i.e. [1,W,C,H] -> [1,H,W,C]. This helper drops batch and produces (H,W,C).
static std::vector<float> transpose_wch_to_hwc(const std::vector<float>& in, int W, int C, int H) {
  const size_t N = static_cast<size_t>(W) * static_cast<size_t>(C) * static_cast<size_t>(H);
  if (in.size() != N) {
    throw std::runtime_error("transpose_wch_to_hwc: unexpected input size");
  }
  std::vector<float> out(static_cast<size_t>(H) * static_cast<size_t>(W) * static_cast<size_t>(C));
  for (int w = 0; w < W; ++w) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        const size_t src = (static_cast<size_t>(w) * static_cast<size_t>(C) + static_cast<size_t>(c)) *
                               static_cast<size_t>(H) +
                           static_cast<size_t>(h);
        const size_t dst = (static_cast<size_t>(h) * static_cast<size_t>(W) + static_cast<size_t>(w)) *
                               static_cast<size_t>(C) +
                           static_cast<size_t>(c);
        out[dst] = in[src];
      }
    }
  }
  return out;
}

static void decode_outputs(const std::vector<simaai::neat::Tensor>& tensors, std::vector<Detection>& out,
                           const PreprocMeta& meta, float conf_thr, float nms_iou, int top_k, int keep_top_k,
                           bool with_landmarks) {
  if (tensors.size() != 9) {
    throw std::runtime_error("expected exactly 9 tensors for RetinaFace (got " + std::to_string(tensors.size()) + ")");
  }

  // Follow the Python example exactly:
  // - treat raw outputs as (1, H, C, W) based on printed shapes
  // - apply transpose(0,3,1,2) => (1, W, H, C)
  // - flatten in width-major order

  // Order matches Python example: [land2, land1, land0, box2, box1, box0, cls2, cls1, cls0]
  const auto land2_raw = tensor_to_f32(tensors[0]); // [1,20,20,20]
  const auto land1_raw = tensor_to_f32(tensors[1]); // [1,40,20,40]
  const auto land0_raw = tensor_to_f32(tensors[2]); // [1,80,20,80]
  const auto box2_raw = tensor_to_f32(tensors[3]);  // [1,20,8,20]
  const auto box1_raw = tensor_to_f32(tensors[4]);  // [1,40,8,40]
  const auto box0_raw = tensor_to_f32(tensors[5]);  // [1,80,8,80]
  const auto cls2_raw = tensor_to_f32(tensors[6]);  // [1,20,4,20]
  const auto cls1_raw = tensor_to_f32(tensors[7]);  // [1,40,4,40]
  const auto cls0_raw = tensor_to_f32(tensors[8]);  // [1,80,4,80]

  // Convert W,C,H -> H,W,C (drop batch=1)
  const auto land0 = transpose_wch_to_hwc(land0_raw, /*W=*/80, /*C=*/20, /*H=*/80);
  const auto land1 = transpose_wch_to_hwc(land1_raw, /*W=*/40, /*C=*/20, /*H=*/40);
  const auto land2 = transpose_wch_to_hwc(land2_raw, /*W=*/20, /*C=*/20, /*H=*/20);

  const auto box0 = transpose_wch_to_hwc(box0_raw, /*W=*/80, /*C=*/8, /*H=*/80);
  const auto box1 = transpose_wch_to_hwc(box1_raw, /*W=*/40, /*C=*/8, /*H=*/40);
  const auto box2 = transpose_wch_to_hwc(box2_raw, /*W=*/20, /*C=*/8, /*H=*/20);

  const auto cls0 = transpose_wch_to_hwc(cls0_raw, /*W=*/80, /*C=*/4, /*H=*/80);
  const auto cls1 = transpose_wch_to_hwc(cls1_raw, /*W=*/40, /*C=*/4, /*H=*/40);
  const auto cls2 = transpose_wch_to_hwc(cls2_raw, /*W=*/20, /*C=*/4, /*H=*/20);

  const std::vector<Prior> priors = make_priors(kInferH, kInferW);
  if (priors.size() != 16800) {
    throw std::runtime_error("unexpected priors count: " + std::to_string(priors.size()));
  }

  struct Cand {
    float x1, y1, x2, y2, score;
    std::optional<std::array<float, 10>> landm;
  };

  std::vector<Cand> cands;
  cands.reserve(2000);

  size_t prior_idx = 0;
  auto process_level = [&](int H, int W, const std::vector<float>& box_whc, const std::vector<float>& cls_whc,
                           const std::vector<float>& land_whc, int boxC, int clsC, int landC) {
    // Iterate H-major then W-major to match RetinaFaceSpy + HWC flattening.
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const size_t cell = (static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x));
        const size_t box_base = cell * static_cast<size_t>(boxC);
        const size_t cls_base = cell * static_cast<size_t>(clsC);
        const size_t land_base = cell * static_cast<size_t>(landC);

        for (int a = 0; a < 2; ++a) {
          const Prior& p = priors[prior_idx++];

          // class logits are [bg, face]
          const float bg = cls_whc[cls_base + static_cast<size_t>(a * 2 + 0)];
          const float fg = cls_whc[cls_base + static_cast<size_t>(a * 2 + 1)];
          // softmax for 2-class
          const float m = std::max(bg, fg);
          const float e0 = std::exp(bg - m);
          const float e1 = std::exp(fg - m);
          const float prob = e1 / (e0 + e1);
          if (prob < conf_thr) {
            continue;
          }

          // bbox loc: [dx, dy, dw, dh]
          const float dx = box_whc[box_base + static_cast<size_t>(a * 4 + 0)];
          const float dy = box_whc[box_base + static_cast<size_t>(a * 4 + 1)];
          const float dw = box_whc[box_base + static_cast<size_t>(a * 4 + 2)];
          const float dh = box_whc[box_base + static_cast<size_t>(a * 4 + 3)];

          const float cx = p.cx + dx * kVariance[0] * p.sx;
          const float cy = p.cy + dy * kVariance[0] * p.sy;
          const float w = p.sx * std::exp(dw * kVariance[1]);
          const float h = p.sy * std::exp(dh * kVariance[1]);
          float x1 = cx - w / 2.0f;
          float y1 = cy - h / 2.0f;
          float x2 = cx + w / 2.0f;
          float y2 = cy + h / 2.0f;

          std::optional<std::array<float, 10>> landm;
          if (with_landmarks) {
            std::array<float, 10> lm{};
            for (int i = 0; i < 10; ++i) {
              const float v = land_whc[land_base + static_cast<size_t>(a * 10 + i)];
              // decode_landm uses: priors_xy + pre * var0 * priors_wh
              if (i % 2 == 0) {
                lm[static_cast<size_t>(i)] = p.cx + v * kVariance[0] * p.sx;
              } else {
                lm[static_cast<size_t>(i)] = p.cy + v * kVariance[0] * p.sy;
              }
            }
            landm = lm;
          }

          cands.push_back(Cand{x1, y1, x2, y2, prob, landm});
        }
      }
    }
  };

  // Levels correspond to 80x80 (step 8), 40x40 (step 16), 20x20 (step 32)
  process_level(80, 80, box0, cls0, land0, /*boxC=*/8, /*clsC=*/4, /*landC=*/20);
  process_level(40, 40, box1, cls1, land1, /*boxC=*/8, /*clsC=*/4, /*landC=*/20);
  process_level(20, 20, box2, cls2, land2, /*boxC=*/8, /*clsC=*/4, /*landC=*/20);

  // Sort by score desc and keep top-k before NMS (like RetinaFaceSpy).
  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
  if (top_k > 0 && static_cast<size_t>(top_k) < cands.size()) {
    cands.resize(static_cast<size_t>(top_k));
  }

  auto iou = [](const Cand& a, const Cand& b) -> float {
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, xx2 - xx1);
    const float h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float den = area_a + area_b - inter;
    return den > 0.0f ? inter / den : 0.0f;
  };

  std::vector<Cand> kept;
  kept.reserve(cands.size());
  for (const auto& cand : cands) {
    bool suppress = false;
    for (const auto& k : kept) {
      if (iou(cand, k) > nms_iou) {
        suppress = true;
        break;
      }
    }
    if (!suppress) {
      kept.push_back(cand);
    }
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
        lm[static_cast<size_t>(i)] = lm[static_cast<size_t>(i)] * static_cast<float>(meta.padded_w) -
                                     static_cast<float>(meta.pad_left);
        lm[static_cast<size_t>(i + 1)] = lm[static_cast<size_t>(i + 1)] * static_cast<float>(meta.padded_h) -
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
    const cv::Point p1(static_cast<int>(std::lround(d.x1)), static_cast<int>(std::lround(d.y1)));
    const cv::Point p2(static_cast<int>(std::lround(d.x2)), static_cast<int>(std::lround(d.y2)));
    cv::rectangle(bgr, p1, p2, cv::Scalar(0, 255, 0), 2);
    cv::putText(bgr, std::to_string(d.score).substr(0, 5), cv::Point(p1.x, std::max(0, p1.y - 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    if (d.landmarks.has_value()) {
      const auto& lm = *d.landmarks;
      for (int i = 0; i < 10; i += 2) {
        cv::circle(bgr, cv::Point(static_cast<int>(std::lround(lm[static_cast<size_t>(i)])),
                                  static_cast<int>(std::lround(lm[static_cast<size_t>(i + 1)]))),
                   2, cv::Scalar(0, 0, 255), -1);
      }
    }
  }
}

struct Args {
  fs::path image;
  std::string model = "/mnt/Bitbucket/sima-neat/apps/assets/models/retinaface_mobilenet25_mod_0_mpk.tar.gz";
  fs::path output;
  float conf = 0.4f;
  float nms = 0.9f;
  int top_k = 5000;
  int keep_top_k = 750;
  int max_draw = 50;
  bool landmarks = true;
  bool profile = false;
  int num_runs = 100;
};

static Args parse_args(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error("Usage: retinaface-face-detection <image_path> [--model ...] [--output ...] "
                             "[--conf ...] [--nms ...] [--top-k ...] [--keep-top-k ...] [--max-draw ...] "
                             "[--no-landmarks]");
  }

  Args a;
  a.image = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (arg == "--model") {
      a.model = need("--model");
    } else if (arg == "--output") {
      a.output = need("--output");
    } else if (arg == "--conf") {
      a.conf = std::stof(need("--conf"));
    } else if (arg == "--nms") {
      a.nms = std::stof(need("--nms"));
    } else if (arg == "--top-k") {
      a.top_k = std::stoi(need("--top-k"));
    } else if (arg == "--keep-top-k") {
      a.keep_top_k = std::stoi(need("--keep-top-k"));
    } else if (arg == "--max-draw") {
      a.max_draw = std::stoi(need("--max-draw"));
    } else if (arg == "--profile") {
      a.profile = true;
    } else if (arg == "--num-runs") {
      a.num_runs = std::stoi(need("--num-runs"));
    } else if (arg == "--no-landmarks") {
      a.landmarks = false;
    } else {
      throw std::runtime_error("unknown arg: " + arg);
    }
  }
  return a;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  try {
    const Args args = parse_args(argc, argv);
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

    cv::Mat bgr_f32;
    bgr_u8.convertTo(bgr_f32, CV_32FC3);
    bgr_mean_subtract_inplace(bgr_f32);
    auto [padded, meta] = pad_to_aspect(bgr_f32, kInferW, kInferH);

    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(kInferW, kInferH), 0, 0, cv::INTER_LINEAR);

    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "application/vnd.simaai.tensor";
    model_opt.format = "";
    model_opt.input_max_width = kInferW;
    model_opt.input_max_height = kInferH;
    model_opt.input_max_depth = 3;

    simaai::neat::Model model(args.model, model_opt);

    simaai::neat::Session session;
    session.add(simaai::neat::nodes::Input(model.input_appsrc_options(true)));
    session.add(simaai::neat::nodes::QuantTess(simaai::neat::QuantTessOptions(model)));
    session.add(simaai::neat::nodes::groups::MLA(model));
    session.add(simaai::neat::nodes::DetessDequant(simaai::neat::DetessDequantOptions(model)));
    session.add(simaai::neat::nodes::Output());

    // Build run with dummy tensor
    cv::Mat dummy(kInferH, kInferW, CV_32FC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy_t = tensor_from_hwc_f32(dummy);
    auto run = session.build(dummy_t, simaai::neat::RunMode::Sync);

    simaai::neat::Tensor input_t = tensor_from_hwc_f32(resized);
    std::vector<Detection> dets;

    if (args.profile) {
      const int runs = std::max(1, args.num_runs);
      std::vector<double> session_times;
      std::vector<double> post_times;
      session_times.reserve(runs);
      post_times.reserve(runs);

      for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!run.push(input_t)) {
          throw std::runtime_error("run.push failed during profiling");
        }
        auto out_sample = run.pull(/*timeout_ms=*/5000);
        const auto t1 = std::chrono::steady_clock::now();
        if (!out_sample.has_value()) {
          throw std::runtime_error("run.pull returned no sample during profiling");
        }

        const auto t2 = std::chrono::steady_clock::now();
        const std::vector<simaai::neat::Tensor> out_tensors = tensors_from_sample(*out_sample);
        dets.clear();
        decode_outputs(out_tensors, dets, meta, args.conf, args.nms, args.top_k, args.keep_top_k, args.landmarks);
        const auto t3 = std::chrono::steady_clock::now();

        const std::chrono::duration<double> dt_session = t1 - t0;
        const std::chrono::duration<double> dt_post = t3 - t2;
        session_times.push_back(dt_session.count());
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

      const auto sess = stats(session_times);
      const auto post = stats(post_times);

      const double runs_d = static_cast<double>(session_times.size());
      const double fps_session = runs_d / sess.sum;
      const double fps_post = runs_d / post.sum;
      const double total_sum = sess.sum + post.sum;
      const double fps_overall = runs_d / total_sum;

      std::cout << "Profiling over " << session_times.size() << " runs (image='" << args.image.string() << "'):\n";
      std::cout << "  Session (push+pull): mean=" << sess.mean << "s, min=" << sess.min << "s, max=" << sess.max
                << "s, FPS=" << fps_session << "\n";
      std::cout << "  Postprocessing (decode+NMS): mean=" << post.mean << "s, min=" << post.min
                << "s, max=" << post.max << "s, FPS=" << fps_post << "\n";
      std::cout << "  Overall (session + post): mean=" << (total_sum / runs_d) << "s, min="
                << (sess.min + post.min) << "s, max=" << (sess.max + post.max) << "s, FPS=" << fps_overall << "\n";

      std::cout << "Last run detections: " << dets.size() << "\n";
      for (size_t i = 0; i < std::min<size_t>(dets.size(), 20); ++i) {
        const auto& d = dets[i];
        std::cout << "  [" << i << "] score=" << d.score << " box=[" << d.x1 << "," << d.y1 << "," << d.x2 << ","
                  << d.y2 << "]\n";
      }
    } else {
      if (!run.push(input_t)) {
        throw std::runtime_error("run.push failed");
      }

      auto out_sample = run.pull(/*timeout_ms=*/5000);
      if (!out_sample.has_value()) {
        throw std::runtime_error("run.pull returned no sample");
      }

      const std::vector<simaai::neat::Tensor> out_tensors = tensors_from_sample(*out_sample);
      std::cout << "Model produced " << out_tensors.size() << " tensor(s)\n";
      for (size_t i = 0; i < out_tensors.size(); ++i) {
        const auto& t = out_tensors[i];
        std::cout << "  [" << i << "] dtype=" << static_cast<int>(t.dtype) << " shape=[";
        for (size_t d = 0; d < t.shape.size(); ++d) {
          std::cout << t.shape[d] << (d + 1 < t.shape.size() ? "," : "");
        }
        std::cout << "]\n";
      }

      decode_outputs(out_tensors, dets, meta, args.conf, args.nms, args.top_k, args.keep_top_k, args.landmarks);

      std::cout << "Detections: " << dets.size() << "\n";
      for (size_t i = 0; i < std::min<size_t>(dets.size(), 20); ++i) {
        const auto& d = dets[i];
        std::cout << "  [" << i << "] score=" << d.score << " box=[" << d.x1 << "," << d.y1 << "," << d.x2 << ","
                  << d.y2 << "]\n";
      }

      if (!args.output.empty()) {
        cv::Mat overlay = bgr_u8.clone();
        draw_detections(overlay, dets, args.max_draw);
        if (!cv::imwrite(args.output.string(), overlay)) {
          throw std::runtime_error("failed to write: " + args.output.string());
        }
        std::cout << "Wrote annotated image: " << args.output << "\n";
      }
    }

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}

