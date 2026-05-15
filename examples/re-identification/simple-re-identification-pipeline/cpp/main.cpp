/**
 * @example simple-re-identification-pipeline.cpp
 * Pairwise ReID pipeline: infer embeddings for two images and compare similarity.
 *
 * Usage:
 *   simple-re-identification-pipeline <image1> <image2>
 *     [--metric cosine|euclidean]
 *     [--threshold <float>]
 *     [--output-dir <path>]
 *     [--output-type image|json|both]
 *     [--model <model.tar.gz>]
 *     [--profile]
 */
#include "neat.h"

#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kInputW = 128;
constexpr int kInputH = 256;
constexpr int kTimeoutMs = 5000;

struct Args {
  fs::path image1;
  fs::path image2;
  std::string metric = "cosine";
  bool has_threshold = false;
  double threshold = 0.0;
  fs::path output_dir;
  std::string output_type = "both";
  fs::path model;
  bool profile = false;
};

fs::path default_output_dir() {
  return fs::path(__FILE__).parent_path().parent_path() / "output_dir";
}

fs::path default_model_path() {
  return fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path() /
         "assets" / "models" / "reid_mpk.tar.gz";
}

void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <image1> <image2>"
            << " [--metric cosine|euclidean]"
            << " [--threshold <float>]"
            << " [--output-dir <path>]"
            << " [--output-type image|json|both]"
            << " [--model <model.tar.gz>]"
            << " [--profile]\n";
}

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

double parse_double(const std::string& s) {
  size_t consumed = 0;
  const double v = std::stod(s, &consumed);
  if (consumed != s.size() || !std::isfinite(v)) {
    throw std::runtime_error("invalid float: " + s);
  }
  return v;
}

int parse_args(int argc, char** argv, Args& args, std::string& err) {
  args.output_dir = default_output_dir();
  args.model = default_model_path();

  if (argc < 3) {
    print_usage(argv[0]);
    return 2;
  }

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string cur = argv[i];
    if (cur == "--metric") {
      if (i + 1 >= argc) {
        err = "missing value for --metric";
        return 2;
      }
      args.metric = argv[++i];
      if (args.metric != "cosine" && args.metric != "euclidean") {
        err = "invalid --metric value: " + args.metric;
        return 2;
      }
    } else if (cur == "--threshold") {
      if (i + 1 >= argc) {
        err = "missing value for --threshold";
        return 2;
      }
      try {
        args.threshold = parse_double(argv[++i]);
        args.has_threshold = true;
      } catch (const std::exception& e) {
        err = e.what();
        return 2;
      }
    } else if (cur == "--output-dir") {
      if (i + 1 >= argc) {
        err = "missing value for --output-dir";
        return 2;
      }
      args.output_dir = argv[++i];
    } else if (cur == "--output-type") {
      if (i + 1 >= argc) {
        err = "missing value for --output-type";
        return 2;
      }
      args.output_type = argv[++i];
      if (args.output_type != "image" && args.output_type != "json" && args.output_type != "both") {
        err = "invalid --output-type value: " + args.output_type;
        return 2;
      }
    } else if (cur == "--model") {
      if (i + 1 >= argc) {
        err = "missing value for --model";
        return 2;
      }
      args.model = argv[++i];
    } else if (cur == "--profile") {
      args.profile = true;
    } else if (cur.rfind("--", 0) == 0) {
      err = "unrecognized argument: " + cur;
      return 2;
    } else {
      positional.push_back(cur);
    }
  }

  if (positional.size() != 2) {
    err = "expected exactly two image paths";
    return 2;
  }

  args.image1 = positional[0];
  args.image2 = positional[1];

  if (!args.has_threshold) {
    args.threshold = (args.metric == "cosine") ? 0.65 : 25.0;
  }

  return 0;
}

size_t dtype_bytes(simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
  case simaai::neat::TensorDType::Int8:
    return 1;
  case simaai::neat::TensorDType::UInt16:
  case simaai::neat::TensorDType::Int16:
    return 2;
  case simaai::neat::TensorDType::Int32:
  case simaai::neat::TensorDType::Float32:
    return 4;
  case simaai::neat::TensorDType::Float64:
    return 8;
  }
  return 1;
}

float read_elem(const uint8_t* data, size_t idx, simaai::neat::TensorDType dtype) {
  switch (dtype) {
  case simaai::neat::TensorDType::UInt8:
    return static_cast<float>(reinterpret_cast<const uint8_t*>(data)[idx]);
  case simaai::neat::TensorDType::Int8:
    return static_cast<float>(reinterpret_cast<const int8_t*>(data)[idx]);
  case simaai::neat::TensorDType::UInt16:
    return static_cast<float>(reinterpret_cast<const uint16_t*>(data)[idx]);
  case simaai::neat::TensorDType::Int16:
    return static_cast<float>(reinterpret_cast<const int16_t*>(data)[idx]);
  case simaai::neat::TensorDType::Int32:
    return static_cast<float>(reinterpret_cast<const int32_t*>(data)[idx]);
  case simaai::neat::TensorDType::Float32:
    return reinterpret_cast<const float*>(data)[idx];
  case simaai::neat::TensorDType::Float64:
    return static_cast<float>(reinterpret_cast<const double*>(data)[idx]);
  }
  return 0.0f;
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

std::vector<float> tensor_to_float_vector(const simaai::neat::Tensor& t) {
  if (!t.is_dense()) {
    throw std::runtime_error("output tensor is not dense");
  }

  size_t elems = 1;
  if (!t.shape.empty()) {
    for (const auto d : t.shape) {
      elems *= static_cast<size_t>(d);
    }
  }

  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  const size_t elem_sz = dtype_bytes(t.dtype);

  if (t.shape.empty()) {
    if (elem_sz == 0 || (raw.size() % elem_sz) != 0) {
      throw std::runtime_error("cannot infer element count from raw tensor bytes");
    }
    elems = raw.size() / elem_sz;
  }

  if (raw.size() < elems * elem_sz) {
    throw std::runtime_error("tensor byte size is smaller than expected");
  }

  std::vector<float> out(elems, 0.0f);
  for (size_t i = 0; i < elems; ++i) {
    out[i] = read_elem(raw.data(), i, t.dtype);
  }
  return out;
}

cv::Mat preprocess_image_like_python(const cv::Mat& bgr) {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat rgb_f32;
  rgb.convertTo(rgb_f32, CV_32FC3);

  const cv::Scalar mean(0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0);
  const cv::Scalar stdv(0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0);

  cv::Mat normalized;
  cv::subtract(rgb_f32, mean, normalized);
  cv::divide(normalized, stdv, normalized);

  cv::Mat resized_f32;
  cv::resize(normalized, resized_f32, cv::Size(kInputW, kInputH), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat out_u8(kInputH, kInputW, CV_8UC3);
  for (int y = 0; y < kInputH; ++y) {
    const cv::Vec3f* src_row = resized_f32.ptr<cv::Vec3f>(y);
    cv::Vec3b* dst_row = out_u8.ptr<cv::Vec3b>(y);
    for (int x = 0; x < kInputW; ++x) {
      for (int c = 0; c < 3; ++c) {
        const int iv = static_cast<int>(src_row[x][c]);
        dst_row[x][c] = static_cast<uint8_t>(iv & 0xFF);
      }
    }
  }

  return out_u8;
}

double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    throw std::runtime_error("cosine similarity expects equal non-empty embeddings");
  }

  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  if (na <= 0.0 || nb <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

double euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) {
    throw std::runtime_error("euclidean distance expects equal non-empty embeddings");
  }

  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    sum += d * d;
  }
  return std::sqrt(sum);
}

double fit_font_scale(const std::string& text, int max_width, int thickness,
                      double start_scale = 3.0) {
  double scale = start_scale;
  while (scale > 0.1) {
    int baseline = 0;
    const cv::Size size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    if (size.width <= max_width) {
      return scale;
    }
    scale -= 0.05;
  }
  return scale;
}

cv::Mat resize_to_height(const cv::Mat& img, int h) {
  const double ratio = static_cast<double>(h) / static_cast<double>(img.rows);
  cv::Mat out;
  cv::resize(img, out, cv::Size(static_cast<int>(std::round(img.cols * ratio)), h));
  return out;
}

void save_comparison_image(const fs::path& path1, const fs::path& path2, double sim,
                           const std::string& decision, double threshold, const std::string& metric,
                           const fs::path& output_path) {
  cv::Mat img1 = cv::imread(path1.string(), cv::IMREAD_COLOR);
  cv::Mat img2 = cv::imread(path2.string(), cv::IMREAD_COLOR);
  if (img1.empty() || img2.empty()) {
    throw std::runtime_error("failed to reload input images for visualization");
  }

  constexpr int target_h = 400;
  img1 = resize_to_height(img1, target_h);
  img2 = resize_to_height(img2, target_h);

  cv::Mat divider(target_h, 4, CV_8UC3, cv::Scalar(200, 200, 200));
  std::vector<cv::Mat> strip = {img1, divider, img2};
  cv::Mat canvas;
  cv::hconcat(strip, canvas);

  const int canvas_w = canvas.cols;
  const int max_text_w = static_cast<int>(canvas_w * 0.85);

  const int decision_thickness = 3;
  const double decision_scale = fit_font_scale(decision, max_text_w, decision_thickness);
  int decision_baseline = 0;
  const cv::Size decision_size = cv::getTextSize(decision, cv::FONT_HERSHEY_SIMPLEX, decision_scale,
                                                 decision_thickness, &decision_baseline);

  const std::string metric_label =
      (metric == "cosine") ? "Cosine similarity" : "Euclidean distance";
  const std::string details = metric_label + ": " + cv::format("%.4f", sim) +
                              "   |   Threshold: " + cv::format("%.2f", threshold);
  const int details_thickness = 1;
  const double details_scale = fit_font_scale(details, max_text_w, details_thickness);
  int details_baseline = 0;
  const cv::Size details_size = cv::getTextSize(details, cv::FONT_HERSHEY_SIMPLEX, details_scale,
                                                details_thickness, &details_baseline);

  constexpr int padding = 12;
  const int bar_h = decision_size.height + details_size.height + decision_baseline +
                    details_baseline + (padding * 3);
  cv::Mat bar(bar_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));

  cv::vconcat(canvas, bar, canvas);

  const cv::Scalar decision_color =
      (decision == "SAME") ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 220);
  const int decision_x = (canvas_w - decision_size.width) / 2;
  const int decision_y = target_h + padding + decision_size.height;
  cv::putText(canvas, decision, cv::Point(decision_x, decision_y), cv::FONT_HERSHEY_SIMPLEX,
              decision_scale, decision_color, decision_thickness, cv::LINE_AA);

  const int details_x = (canvas_w - details_size.width) / 2;
  const int details_y = decision_y + decision_baseline + padding + details_size.height;
  cv::putText(canvas, details, cv::Point(details_x, details_y), cv::FONT_HERSHEY_SIMPLEX,
              details_scale, cv::Scalar(200, 200, 200), details_thickness, cv::LINE_AA);

  if (!cv::imwrite(output_path.string(), canvas)) {
    throw std::runtime_error("failed to write comparison image: " + output_path.string());
  }
  std::cout << "Comparison image saved to: " << output_path << "\n";
}

void save_result_json(const fs::path& output_path, const fs::path& image_a, const fs::path& image_b,
                      const std::string& metric, double score, double threshold,
                      const std::string& decision) {
  nlohmann::json payload = {
      {"image_a", image_a.string()},
      {"image_b", image_b.string()},
      {"metric", metric},
      {"score", score},
      {"threshold", threshold},
      {"decision", decision},
  };

  std::ofstream out(output_path);
  if (!out.good()) {
    throw std::runtime_error("failed to open result json for writing: " + output_path.string());
  }
  out << payload.dump(2) << "\n";
  std::cout << "Result json saved to: " << output_path << "\n";
}

void print_profile(double infer_a_s, double infer_b_s, double total_s) {
  std::cout << "\n[PROFILE] Timing report\n";
  std::cout << "[PROFILE]   note: one warmup run was performed before timing\n";
  std::cout << "[PROFILE]   inference image_a : " << std::fixed << std::setprecision(1)
            << infer_a_s * 1000.0 << " ms\n";
  std::cout << "[PROFILE]   inference image_b : " << std::fixed << std::setprecision(1)
            << infer_b_s * 1000.0 << " ms\n";
  std::cout << "[PROFILE]   total inference   : " << std::fixed << std::setprecision(1)
            << (infer_a_s + infer_b_s) * 1000.0 << " ms\n";
  std::cout << "[PROFILE]   end-to-end        : " << std::fixed << std::setprecision(1)
            << total_s * 1000.0 << " ms\n";
}

std::vector<float> run_inference_embedding(simaai::neat::Run& run, const fs::path& image_path,
                                           double& infer_time_s) {
  cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Cannot read image: " + image_path.string());
  }

  cv::Mat preprocessed = preprocess_image_like_python(bgr);
  simaai::neat::Tensor input = simaai::neat::Tensor::from_cv_mat(
      preprocessed, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);

  const auto t0 = std::chrono::steady_clock::now();
  if (!run.push(simaai::neat::TensorList{input})) {
    throw std::runtime_error("run.push failed for: " + image_path.filename().string());
  }
  auto out = run.pull(kTimeoutMs);
  const auto t1 = std::chrono::steady_clock::now();
  infer_time_s = std::chrono::duration<double>(t1 - t0).count();
  if (!out.has_value()) {
    throw std::runtime_error("run.pull timeout for: " + image_path.filename().string());
  }

  const auto tensors = collect_tensors(*out);
  if (tensors.empty()) {
    throw std::runtime_error("No tensor output for: " + image_path.filename().string());
  }

  return tensor_to_float_vector(tensors.front());
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  Args args;
  std::string parse_err;
  const int parse_rc = parse_args(argc, argv, args, parse_err);
  if (parse_rc != 0) {
    if (!parse_err.empty()) {
      std::cerr << "Error: " << parse_err << "\n";
      print_usage(argv[0]);
    }
    return parse_rc;
  }

  if (!fs::is_regular_file(args.model)) {
    std::cerr << "Model file does not exist: " << args.model << "\n";
    return 2;
  }
  if (!fs::is_regular_file(args.image1) || !is_image(args.image1)) {
    std::cerr << "Not a valid image file: " << args.image1 << "\n";
    return 2;
  }
  if (!fs::is_regular_file(args.image2) || !is_image(args.image2)) {
    std::cerr << "Not a valid image file: " << args.image2 << "\n";
    return 2;
  }

  fs::create_directories(args.output_dir);

  try {
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::RGB;
    model_opt.preprocess.input_max_width = kInputW;
    model_opt.preprocess.input_max_height = kInputH;
    model_opt.preprocess.input_max_depth = 3;

    simaai::neat::Model model(args.model.string(), model_opt);
    simaai::neat::Session session;
    session.add(model.session());

    cv::Mat dummy_rgb(kInputH, kInputW, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy = simaai::neat::Tensor::from_cv_mat(
        dummy_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, simaai::neat::TensorMemory::EV74);
    auto run = session.build(simaai::neat::TensorList{dummy}, simaai::neat::RunMode::Async);

    // Warmup run before any timed inference.
    if (!run.push(simaai::neat::TensorList{dummy})) {
      throw std::runtime_error("warmup push failed");
    }
    if (!run.pull(10000).has_value()) {
      throw std::runtime_error("warmup pull timeout");
    }
    std::cout << "Model warmed up.\n";

    const auto t_total_0 = std::chrono::steady_clock::now();
    std::cout << "Processing: " << args.image1.filename().string() << "\n";
    double infer_a_s = 0.0;
    const std::vector<float> emb_a = run_inference_embedding(run, args.image1, infer_a_s);

    std::cout << "Processing: " << args.image2.filename().string() << "\n";
    double infer_b_s = 0.0;
    const std::vector<float> emb_b = run_inference_embedding(run, args.image2, infer_b_s);

    double score = 0.0;
    std::string decision;
    if (args.metric == "cosine") {
      score = cosine_similarity(emb_a, emb_b);
      decision = (score >= args.threshold) ? "SAME" : "DIFFERENT";
      std::cout << "\nCosine similarity : " << std::fixed << std::setprecision(6) << score << "\n";
      std::cout << "Threshold         : " << std::fixed << std::setprecision(2) << args.threshold
                << "\n";
      std::cout << "Decision          : " << decision << "\n";
    } else {
      score = euclidean_distance(emb_a, emb_b);
      decision = (score <= args.threshold) ? "SAME" : "DIFFERENT";
      std::cout << "\nEuclidean distance: " << std::fixed << std::setprecision(6) << score << "\n";
      std::cout << "Threshold         : " << std::fixed << std::setprecision(2) << args.threshold
                << "\n";
      std::cout << "Decision          : " << decision << "\n";
    }

    const auto t_total_1 = std::chrono::steady_clock::now();
    const double total_s = std::chrono::duration<double>(t_total_1 - t_total_0).count();

    if (args.output_type == "image" || args.output_type == "both") {
      save_comparison_image(args.image1, args.image2, score, decision, args.threshold, args.metric,
                            args.output_dir / "comparison.jpg");
    }

    if (args.output_type == "json" || args.output_type == "both") {
      save_result_json(args.output_dir / "result.json", args.image1, args.image2, args.metric,
                       score, args.threshold, decision);
    }

    if (args.profile) {
      print_profile(infer_a_s, infer_b_s, total_s);
    }

    run.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
