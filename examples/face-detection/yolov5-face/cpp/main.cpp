/**
 * @example yolov5-face.cpp
 * yolov5s-face inference: faces + 5 keypoints per face. Sync pipeline; the CVU
 * preproc plugin (EV74) handles resize + letterbox + INT8 quantize, the MLA
 * runs the model, and the APU decodes the six raw split heads into boxes +
 * landmarks because BBOX wire format does not carry landmark coordinates.
 *
 * Usage: yolov5-face
 *          --model <model.tar.gz> --labels <labels.txt>
 *          --input-dir <dir> --output-dir <dir>
 *          [--min-score 0.25] [--nms-iou 0.45]
 *          [--profile] [--no-overlay] [--num-runs 1]
 */
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInferSize = 800;
// MPK 0_preproc.json was compiled for NV12 1280x720 input → BGR uint8 frames up to
// this size are letterboxed to kInferSize on-device by the EV74 (CVU) preproc plugin.
constexpr int kMaxInputW = 1280;
constexpr int kMaxInputH = 720;
constexpr float kDefaultMinScore = 0.25f;
constexpr float kDefaultNmsIou = 0.45f;
constexpr int kTimeoutMs = 10000;
constexpr int kNumAnchors = 3;
constexpr int kNumLandmarks = 5;
constexpr int kBoxChan = 18; // 3 anchors × (xywh + obj + cls)
constexpr int kLmChan = 30;  // 3 anchors × (5 landmarks × xy)
constexpr int kPerAnchorBox = 6;
constexpr int kPerAnchorLm = 10;

// Mirrored from compilation.py:62-71 (yolov5s-face anchors / strides).
constexpr float kStrides[3] = {8.0f, 16.0f, 32.0f};
constexpr float kAnchors[3][kNumAnchors][2] = {
    {{4.f,   5.f},   {8.f,   10.f},  {13.f,  16.f}},
    {{23.f,  29.f},  {43.f,  55.f},  {73.f,  105.f}},
    {{146.f, 217.f}, {231.f, 300.f}, {335.f, 433.f}},
};

const std::array<cv::Scalar, kNumLandmarks> kLmColors = {
    cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 0),
    cv::Scalar(255, 0, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 128, 255),
};

struct Config {
  std::string model_path;
  std::string labels_path;
  std::string input_dir;
  std::string output_dir;
  float min_score = kDefaultMinScore;
  float nms_iou = kDefaultNmsIou;
  bool profile = false;
  bool overlay = true;
  int num_runs = 1;
};

struct Detection {
  float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
  float score = 0.f;
  std::array<cv::Point2f, kNumLandmarks> landmarks{};
};

struct LetterboxParams {
  float scale;
  int pad_l;
  int pad_t;
};

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --model <model.tar.gz> --labels <labels.txt>\n"
            << "       --input-dir <dir> --output-dir <dir>\n"
            << "       [--min-score " << kDefaultMinScore << "] [--nms-iou "
            << kDefaultNmsIou << "]\n"
            << "       [--profile] [--no-overlay] [--num-runs 1]\n";
}

int parse_args(int argc, char** argv, Config& cfg) {
  static const struct option long_opts[] = {
      {"model", required_argument, nullptr, 'm'},
      {"labels", required_argument, nullptr, 'l'},
      {"input-dir", required_argument, nullptr, 'i'},
      {"output-dir", required_argument, nullptr, 'o'},
      {"min-score", required_argument, nullptr, 's'},
      {"nms-iou", required_argument, nullptr, 'n'},
      {"profile", no_argument, nullptr, 'p'},
      {"no-overlay", no_argument, nullptr, 'O'},
      {"num-runs", required_argument, nullptr, 'R'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "", long_opts, nullptr)) != -1) {
    switch (opt) {
    case 'm': cfg.model_path = optarg; break;
    case 'l': cfg.labels_path = optarg; break;
    case 'i': cfg.input_dir = optarg; break;
    case 'o': cfg.output_dir = optarg; break;
    case 's': cfg.min_score = std::stof(optarg); break;
    case 'n': cfg.nms_iou = std::stof(optarg); break;
    case 'p': cfg.profile = true; break;
    case 'O': cfg.overlay = false; break;
    case 'R': cfg.num_runs = std::stoi(optarg); break;
    case 'h': print_usage(argv[0]); return 1;
    default: print_usage(argv[0]); return 1;
    }
  }

  if (cfg.model_path.empty() || cfg.labels_path.empty() ||
      cfg.input_dir.empty() || cfg.output_dir.empty()) {
    std::cerr << "Error: --model, --labels, --input-dir, and --output-dir are required.\n";
    print_usage(argv[0]);
    return 1;
  }
  if (cfg.min_score < 0.0f || cfg.min_score > 1.0f) {
    std::cerr << "Error: --min-score must be in [0.0, 1.0], got " << cfg.min_score << "\n";
    return 1;
  }
  if (cfg.nms_iou < 0.0f || cfg.nms_iou > 1.0f) {
    std::cerr << "Error: --nms-iou must be in [0.0, 1.0], got " << cfg.nms_iou << "\n";
    return 1;
  }
  if (cfg.num_runs < 1) {
    std::cerr << "Error: --num-runs must be >= 1, got " << cfg.num_runs << "\n";
    return 1;
  }
  return 0;
}

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
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

// Compute (scale, pad_l, pad_t) for the on-device letterbox without applying it.
// The actual scale + center-pad is done by the MPK's CVU preproc on EV74
// (`0_preproc.json` configures padding=CENTER, interpolation=BILINEAR). We need
// the same parameters on the APU only to inverse-map model-canvas coords back
// to original-image pixels.
LetterboxParams letterbox_params(int orig_w, int orig_h, int target_w, int target_h) {
  const float scale = std::min(static_cast<float>(target_w) / orig_w,
                               static_cast<float>(target_h) / orig_h);
  const int nw = static_cast<int>(std::round(orig_w * scale));
  const int nh = static_cast<int>(std::round(orig_h * scale));
  const int pad_l = (target_w - nw) / 2;
  const int pad_t = (target_h - nh) / 2;
  return {scale, pad_l, pad_t};
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

std::vector<simaai::neat::Tensor> collect_tensors(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor) {
    if (!sample.tensor.has_value()) {
      throw std::runtime_error("tensor sample missing payload");
    }
    return {*sample.tensor};
  }
  std::vector<simaai::neat::Tensor> out;
  for (const auto& field : sample.fields) {
    auto child = collect_tensors(field);
    out.insert(out.end(), child.begin(), child.end());
  }
  return out;
}

// Owns the runtime's dense byte buffer and exposes it as a channels-last (h, w, c)
// float32 view via floats(). The buffer is moved in from copy_dense_bytes_tight()
// without a second memcpy.
struct HwcTensor {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<uint8_t> bytes;

  const float* floats() const {
    return reinterpret_cast<const float*>(bytes.data());
  }
};

// Wrap a 4D NHWC float32 tensor as an HwcTensor (zero re-copy). NEAT's CVU
// detessellation step always emits NHWC for this MPK; anything else is an error.
HwcTensor tensor_to_hwc(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  if (t.shape.size() != 4 || t.shape[0] != 1) {
    throw std::runtime_error("expected 4D tensor with shape [1,H,W,C]");
  }
  const int c = static_cast<int>(t.shape[3]);
  if (c != kBoxChan && c != kLmChan) {
    throw std::runtime_error(
        "expected NHWC tensor with channel dim " + std::to_string(kBoxChan) +
        " or " + std::to_string(kLmChan) + ", got shape [" +
        std::to_string(t.shape[0]) + "," + std::to_string(t.shape[1]) + "," +
        std::to_string(t.shape[2]) + "," + std::to_string(c) + "]");
  }
  HwcTensor out;
  out.h = static_cast<int>(t.shape[1]);
  out.w = static_cast<int>(t.shape[2]);
  out.c = c;
  out.bytes = t.copy_dense_bytes_tight();
  return out;
}

float iou(const Detection& a, const Detection& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
  const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
  const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
  const float den = area_a + area_b - inter;
  return den > 0.f ? inter / den : 0.f;
}

// Mirrored from compilation.py:258-326 — pair box (18ch) and landmark (30ch) heads
// per pyramid level, decode anchors+grid, score = sigmoid(obj)*sigmoid(cls), NMS.
std::vector<Detection> decode_yolov5face_split(const simaai::neat::Sample& sample,
                                                float min_score, float nms_iou_t) {
  const auto tensors = collect_tensors(sample);
  if (tensors.size() != 6) {
    throw std::runtime_error("expected 6 split tensors, got " +
                             std::to_string(tensors.size()));
  }

  struct LevelHeads {
    HwcTensor box;
    HwcTensor lm;
    bool have_box = false;
    bool have_lm = false;
  };
  std::map<int, LevelHeads> groups;
  for (const auto& t : tensors) {
    HwcTensor hwc = tensor_to_hwc(t);
    const int size = std::max(hwc.h, hwc.w);
    if (hwc.c == kBoxChan) {
      groups[size].box = std::move(hwc);
      groups[size].have_box = true;
    } else if (hwc.c == kLmChan) {
      groups[size].lm = std::move(hwc);
      groups[size].have_lm = true;
    }
  }

  if (groups.size() != 3) {
    throw std::runtime_error("expected 3 pyramid levels, got " +
                             std::to_string(groups.size()));
  }

  // Largest spatial size first (P3 → P4 → P5), aligns with strides[0..2].
  std::vector<int> sizes;
  sizes.reserve(groups.size());
  for (const auto& kv : groups) sizes.push_back(kv.first);
  std::sort(sizes.begin(), sizes.end(), std::greater<int>());

  std::vector<Detection> candidates;
  for (size_t lvl = 0; lvl < sizes.size(); ++lvl) {
    const auto& lh = groups[sizes[lvl]];
    if (!lh.have_box || !lh.have_lm) {
      throw std::runtime_error("level missing box or lm head");
    }
    if (lh.box.h != lh.lm.h || lh.box.w != lh.lm.w) {
      throw std::runtime_error("box/lm spatial mismatch at level " + std::to_string(lvl));
    }
    const float stride = kStrides[lvl];
    const int ny = lh.box.h;
    const int nx = lh.box.w;

    const float* box_data = lh.box.floats();
    const float* lm_data = lh.lm.floats();
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const size_t cell_box_off =
            (static_cast<size_t>(y) * nx + x) * kBoxChan;
        const size_t cell_lm_off =
            (static_cast<size_t>(y) * nx + x) * kLmChan;
        for (int a = 0; a < kNumAnchors; ++a) {
          const float* bp = &box_data[cell_box_off + a * kPerAnchorBox];
          const float* lp = &lm_data[cell_lm_off + a * kPerAnchorLm];

          const float obj = sigmoid(bp[4]);
          const float cls = sigmoid(bp[5]);
          const float score = obj * cls;
          if (score < min_score) continue;

          const float aw = kAnchors[lvl][a][0];
          const float ah = kAnchors[lvl][a][1];

          const float sx = sigmoid(bp[0]);
          const float sy = sigmoid(bp[1]);
          const float sw = sigmoid(bp[2]);
          const float sh = sigmoid(bp[3]);

          const float cx = (sx * 2.0f - 0.5f + static_cast<float>(x)) * stride;
          const float cy = (sy * 2.0f - 0.5f + static_cast<float>(y)) * stride;
          const float bw = (sw * 2.0f) * (sw * 2.0f) * aw;
          const float bh = (sh * 2.0f) * (sh * 2.0f) * ah;

          Detection d;
          d.x1 = cx - bw * 0.5f;
          d.y1 = cy - bh * 0.5f;
          d.x2 = cx + bw * 0.5f;
          d.y2 = cy + bh * 0.5f;
          d.score = score;
          for (int k = 0; k < kNumLandmarks; ++k) {
            d.landmarks[k].x = lp[2 * k]     * aw + static_cast<float>(x) * stride;
            d.landmarks[k].y = lp[2 * k + 1] * ah + static_cast<float>(y) * stride;
          }
          candidates.push_back(d);
        }
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Detection& a, const Detection& b) { return a.score > b.score; });
  std::vector<Detection> keep;
  keep.reserve(candidates.size());
  for (const auto& cand : candidates) {
    bool suppressed = false;
    for (const auto& k : keep) {
      if (iou(k, cand) > nms_iou_t) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) keep.push_back(cand);
  }
  return keep;
}

void unletterbox(std::vector<Detection>& dets, float scale, int pad_l, int pad_t,
                 int orig_w, int orig_h) {
  const float ow = static_cast<float>(orig_w);
  const float oh = static_cast<float>(orig_h);
  for (auto& d : dets) {
    d.x1 = std::clamp((d.x1 - pad_l) / scale, 0.f, ow);
    d.y1 = std::clamp((d.y1 - pad_t) / scale, 0.f, oh);
    d.x2 = std::clamp((d.x2 - pad_l) / scale, 0.f, ow);
    d.y2 = std::clamp((d.y2 - pad_t) / scale, 0.f, oh);
    for (auto& lm : d.landmarks) {
      lm.x = (lm.x - pad_l) / scale;
      lm.y = (lm.y - pad_t) / scale;
    }
  }
}

void draw_overlay(cv::Mat& image, const std::vector<Detection>& dets,
                  const std::string& label) {
  const cv::Scalar green(0, 255, 0);
  for (const auto& d : dets) {
    const int x1 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(d.x1))));
    const int y1 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(d.y1))));
    const int x2 = std::max(0, std::min(image.cols - 1, static_cast<int>(std::round(d.x2))));
    const int y2 = std::max(0, std::min(image.rows - 1, static_cast<int>(std::round(d.y2))));
    if (x2 <= x1 || y2 <= y1) continue;

    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), green, 2);
    const std::string text = label + " " + cv::format("%.2f", d.score);
    int baseline = 0;
    const cv::Size tsz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int y_top = std::max(0, y1 - tsz.height - 4);
    const int y_bot = std::max(0, y1);
    cv::rectangle(image, cv::Point(x1, y_top), cv::Point(x1 + tsz.width, y_bot),
                  green, cv::FILLED);
    cv::putText(image, text, cv::Point(x1, std::max(0, y1 - 2)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    for (int k = 0; k < kNumLandmarks; ++k) {
      cv::circle(image,
                 cv::Point(static_cast<int>(std::round(d.landmarks[k].x)),
                           static_cast<int>(std::round(d.landmarks[k].y))),
                 2, kLmColors[k], cv::FILLED);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  Config cfg;
  int rc = parse_args(argc, argv, cfg);
  if (rc != 0) return rc;

  const fs::path input_dir = cfg.input_dir;
  const fs::path output_dir = cfg.output_dir;
  const fs::path labels_path = cfg.labels_path;

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  fs::create_directories(output_dir);

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
    model_opt.media_type = "video/x-raw";
    model_opt.format = "BGR";
    model_opt.input_max_width = kMaxInputW;
    model_opt.input_max_height = kMaxInputH;
    model_opt.input_max_depth = 3;

    simaai::neat::Model model(cfg.model_path, model_opt);

    simaai::neat::Session session;
    session.add(model.session());
    std::cout << "[BUILD] Pipeline:\n" << session.describe_backend() << "\n";

    // On-device preproc caches kernel params per input shape; warming up at
    // kMaxInput* leaves the first real frame to pay a ~2 s reconfig. Use the
    // first real image's shape so the cache is primed for the loop.
    cv::Mat first_bgr = cv::imread(images.front().string(), cv::IMREAD_COLOR);
    if (first_bgr.empty()) {
      std::cerr << "Cannot read first image " << images.front() << "\n";
      return 3;
    }
    if (first_bgr.cols > kMaxInputW || first_bgr.rows > kMaxInputH) {
      std::cerr << "First image dims " << first_bgr.cols << "x" << first_bgr.rows
                << " exceed device preproc capacity " << kMaxInputW << "x" << kMaxInputH
                << "\n";
      return 3;
    }
    simaai::neat::Tensor warm = simaai::neat::from_cv_mat(
        first_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, /*read_only=*/true);
    auto run = session.build(warm, simaai::neat::RunMode::Sync);
    run.push_and_pull(warm, kTimeoutMs);
    std::cout << "[WARMUP] done\n";

    const int total_images = static_cast<int>(images.size()) * cfg.num_runs;
    if (cfg.num_runs > 1) {
      std::cout << "Looping " << cfg.num_runs << "x over " << images.size()
                << " images (" << total_images << " total)\n";
    }

    using clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    const auto pipeline_start = clock::now();
    int processed = 0;

    for (int run_idx = 0; run_idx < cfg.num_runs; ++run_idx) {
      for (const auto& image_path : images) {
        const auto img_start = clock::now();

        cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
          std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
          continue;
        }
        const int orig_w = bgr.cols;
        const int orig_h = bgr.rows;
        if (orig_w > kMaxInputW || orig_h > kMaxInputH) {
          std::cerr << "Skipping " << image_path.filename() << ": dims " << orig_w << "x"
                    << orig_h << " exceed device preproc capacity " << kMaxInputW << "x"
                    << kMaxInputH << "\n";
          continue;
        }

        LetterboxParams lb = letterbox_params(orig_w, orig_h, kInferSize, kInferSize);
        simaai::neat::Tensor input = simaai::neat::from_cv_mat(
            bgr, simaai::neat::ImageSpec::PixelFormat::BGR, /*read_only=*/true);

        const auto infer_start = clock::now();
        simaai::neat::Sample out = run.push_and_pull(input, kTimeoutMs);
        const auto infer_end = clock::now();

        auto dets = decode_yolov5face_split(out, cfg.min_score, cfg.nms_iou);
        unletterbox(dets, lb.scale, lb.pad_l, lb.pad_t, orig_w, orig_h);
        const auto decode_end = clock::now();

        if (cfg.overlay) {
          draw_overlay(bgr, dets, labels[0]);
          const fs::path out_path = output_dir / (image_path.stem().string() + ".png");
          if (!cv::imwrite(out_path.string(), bgr)) {
            std::cerr << "Failed to write: " << out_path << "\n";
            continue;
          }
        }
        const auto img_end = clock::now();

        ++processed;
        if (cfg.overlay) {
          std::cout << "[" << processed << "/" << total_images << "] "
                    << image_path.filename() << " -> "
                    << image_path.stem().string() << ".png ("
                    << dets.size() << " faces)\n";
        } else {
          std::cout << "[" << processed << "/" << total_images << "] "
                    << image_path.filename() << " (" << dets.size() << " faces)\n";
        }

        if (cfg.profile) {
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

    run.close();

    if (cfg.profile && processed > 0) {
      const auto pipeline_end = clock::now();
      const auto total_s = std::chrono::duration<double>(pipeline_end - pipeline_start).count();
      const auto avg_ms = (total_s * 1000.0) / static_cast<double>(processed);
      const auto fps = static_cast<double>(processed) / total_s;
      std::cout << "[PROFILE] Total: " << processed << " images in "
                << cv::format("%.2f", total_s) << "s (avg "
                << cv::format("%.1f", avg_ms) << "ms/image, "
                << cv::format("%.1f", fps) << " FPS)\n";
    }

    std::cout << "Done: " << processed << " images processed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
