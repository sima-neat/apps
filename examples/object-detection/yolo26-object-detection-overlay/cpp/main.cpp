/**
 * @example yolo26-object-detection-overlay.cpp
 * Minimal yolo26m sync pipeline: infer detections for every image in a folder.
 *
 * Usage: yolo26-object-detection-overlay
 *          --model <model.tar.gz> --labels <labels.txt>
 *          --input-dir <dir> --output-dir <dir>
 *          [--min-score 0.25] [--nms-iou 0.45]
 *          [--profile] [--no-overlay]
 */
#include "neat.h"
#include "support/object_detection/obj_detection_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInferSize = 640;
constexpr float kDefaultMinScore = 0.25f;
constexpr float kDefaultNmsIou = 0.45f;
constexpr int kMaxDet = 100;
constexpr int kTimeoutMs = 5000;

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

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " --model <model.tar.gz> --labels <labels.txt>\n"
            << "       --input-dir <dir> --output-dir <dir>\n"
            << "       [--min-score " << kDefaultMinScore << "] [--nms-iou " << kDefaultNmsIou
            << "]\n"
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
    case 'm':
      cfg.model_path = optarg;
      break;
    case 'l':
      cfg.labels_path = optarg;
      break;
    case 'i':
      cfg.input_dir = optarg;
      break;
    case 'o':
      cfg.output_dir = optarg;
      break;
    case 's':
      cfg.min_score = std::stof(optarg);
      break;
    case 'n':
      cfg.nms_iou = std::stof(optarg);
      break;
    case 'p':
      cfg.profile = true;
      break;
    case 'O':
      cfg.overlay = false;
      break;
    case 'R':
      cfg.num_runs = std::stoi(optarg);
      break;
    case 'h':
      print_usage(argv[0]);
      return 1;
    default:
      print_usage(argv[0]);
      return 1;
    }
  }

  if (cfg.model_path.empty() || cfg.labels_path.empty() || cfg.input_dir.empty() ||
      cfg.output_dir.empty()) {
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

// Return tensor data as a flat float32 vector plus (H, W, C) dims.
// Expects rank-3 (H,W,C) or rank-4 (1,H,W,C) float32 tensors.
struct HWCTensor {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;
};

HWCTensor tensor_to_hwc(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  HWCTensor out;
  if (t.shape.size() == 4 && t.shape[0] == 1) {
    out.h = static_cast<int>(t.shape[1]);
    out.w = static_cast<int>(t.shape[2]);
    out.c = static_cast<int>(t.shape[3]);
  } else if (t.shape.size() == 3) {
    out.h = static_cast<int>(t.shape[0]);
    out.w = static_cast<int>(t.shape[1]);
    out.c = static_cast<int>(t.shape[2]);
  } else {
    throw std::runtime_error("unexpected tensor rank");
  }
  const size_t elems = static_cast<size_t>(out.h) * out.w * out.c;
  const auto bytes = t.copy_dense_bytes_tight();
  out.data.resize(elems);
  std::memcpy(out.data.data(), bytes.data(), elems * sizeof(float));
  return out;
}

float iou_xyxy(const objdet::Box& a, const objdet::Box& b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float den = area_a + area_b - inter;
  return den > 0.0f ? (inter / den) : 0.0f;
}

float class_confidence(float value) {
  return std::clamp(value, 0.0f, 1.0f);
}

// Decode YOLO26 detess/dequant output into the package's postprocess contract:
// 3 regression tensors as absolute (cx,cy,w,h) in 640x640 model space, plus
// 3 class-probability tensors.
std::vector<objdet::Box> decode_yolo26_boxes(const simaai::neat::TensorList& tensors,
                                             float min_score, float nms_iou) {
  if (tensors.size() < 6) {
    throw std::runtime_error("expected at least 6 tensors, got " + std::to_string(tensors.size()));
  }
  const std::array<HWCTensor, 3> regs = {tensor_to_hwc(tensors[0]), tensor_to_hwc(tensors[1]),
                                         tensor_to_hwc(tensors[2])};
  const std::array<HWCTensor, 3> clss = {tensor_to_hwc(tensors[3]), tensor_to_hwc(tensors[4]),
                                         tensor_to_hwc(tensors[5])};

  std::vector<objdet::Box> candidates;
  for (size_t lvl = 0; lvl < regs.size(); ++lvl) {
    const auto& reg = regs[lvl];
    const auto& cls = clss[lvl];
    if (reg.c != 4) {
      throw std::runtime_error("expected 4-channel regression (cx,cy,w,h), got c=" +
                               std::to_string(reg.c));
    }
    if (reg.h != cls.h || reg.w != cls.w) {
      throw std::runtime_error("reg/cls spatial mismatch");
    }
    for (int y = 0; y < reg.h; ++y) {
      for (int x = 0; x < reg.w; ++x) {
        const size_t cls_base = (static_cast<size_t>(y) * reg.w + x) * cls.c;
        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < cls.c; ++c) {
          const float s = class_confidence(cls.data[cls_base + c]);
          if (s > best_score) {
            best_score = s;
            best_class = c;
          }
        }
        if (best_score < min_score || best_class < 0)
          continue;

        const size_t reg_base = (static_cast<size_t>(y) * reg.w + x) * 4U;
        const float bcx = reg.data[reg_base + 0];
        const float bcy = reg.data[reg_base + 1];
        const float bw = reg.data[reg_base + 2];
        const float bh = reg.data[reg_base + 3];
        objdet::Box box;
        box.x1 = std::max(0.0f, bcx - bw * 0.5f);
        box.y1 = std::max(0.0f, bcy - bh * 0.5f);
        box.x2 = std::min(static_cast<float>(kInferSize), bcx + bw * 0.5f);
        box.y2 = std::min(static_cast<float>(kInferSize), bcy + bh * 0.5f);
        box.score = best_score;
        box.class_id = best_class;
        if (box.x2 > box.x1 && box.y2 > box.y1) {
          candidates.push_back(box);
        }
      }
    }
  }

  // Per-class NMS, highest-score-first.
  std::sort(candidates.begin(), candidates.end(),
            [](const objdet::Box& a, const objdet::Box& b) { return a.score > b.score; });
  std::vector<objdet::Box> keep;
  keep.reserve(kMaxDet);
  for (const auto& b : candidates) {
    bool suppressed = false;
    for (const auto& k : keep) {
      if (k.class_id == b.class_id && iou_xyxy(k, b) > nms_iou) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      keep.push_back(b);
      if (static_cast<int>(keep.size()) >= kMaxDet)
        break;
    }
  }
  return keep;
}

std::vector<objdet::Box> scale_boxes_to_original(const std::vector<objdet::Box>& boxes, int out_w,
                                                 int out_h) {
  const float sx = static_cast<float>(out_w) / static_cast<float>(kInferSize);
  const float sy = static_cast<float>(out_h) / static_cast<float>(kInferSize);
  std::vector<objdet::Box> out;
  out.reserve(boxes.size());
  for (const auto& b : boxes) {
    out.push_back(objdet::Box{b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy, b.score, b.class_id});
  }
  return out;
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

void draw_boxes(cv::Mat& frame, const std::vector<objdet::Box>& boxes,
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
  int rc = parse_args(argc, argv, cfg);
  if (rc != 0) {
    return rc;
  }

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
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.preprocess.input_max_width = kInferSize;
    model_opt.preprocess.input_max_height = kInferSize;
    model_opt.preprocess.input_max_depth = 3;
    model_opt.processcvu.post_run_target = "A65";

    simaai::neat::Model model(cfg.model_path, model_opt);

    simaai::neat::Session session;
    session.add(model.session());
    std::cout << "[BUILD] Pipeline:\n" << session.describe_backend() << "\n";

    cv::Mat dummy_bgr(kInferSize, kInferSize, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy =
        simaai::neat::Tensor::from_cv_mat(dummy_bgr, simaai::neat::ImageSpec::PixelFormat::BGR);
    auto run = session.build(simaai::neat::TensorList{dummy}, simaai::neat::RunMode::Sync);

    // Warmup inference to stabilize timing before profiling.
    run.run(simaai::neat::TensorList{dummy}, kTimeoutMs);
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

        const int orig_w = bgr.cols;
        const int orig_h = bgr.rows;

        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(kInferSize, kInferSize), 0, 0, cv::INTER_LINEAR);
        simaai::neat::Tensor input =
            simaai::neat::Tensor::from_cv_mat(resized, simaai::neat::ImageSpec::PixelFormat::BGR);

        const auto infer_start = std::chrono::steady_clock::now();
        simaai::neat::TensorList out = run.run(simaai::neat::TensorList{input}, kTimeoutMs);
        const auto infer_end = std::chrono::steady_clock::now();

        const std::vector<objdet::Box> boxes_infer =
            decode_yolo26_boxes(out, cfg.min_score, cfg.nms_iou);
        const auto decode_end = std::chrono::steady_clock::now();

        const auto boxes_orig = scale_boxes_to_original(boxes_infer, orig_w, orig_h);

        if (cfg.overlay) {
          draw_boxes(bgr, boxes_orig, labels);
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
                    << " (" << boxes_orig.size() << " detections)\n";
        } else {
          std::cout << "[" << processed << "/" << total_images << "] " << image_path.filename()
                    << " (" << boxes_orig.size() << " detections)\n";
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

    run.close();

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
