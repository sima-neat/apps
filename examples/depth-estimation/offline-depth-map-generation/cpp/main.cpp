/**
 * @example offline-depth-map-generation.cpp
 * Minimal Depth Anything V2 pipeline: infer depth for every image in a folder.
 *
 * Usage: offline-depth-map-generation <model.tar.gz> <input_dir> <output_dir>
 */
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers (adapted from live-rtsp-depth-estimation.cpp)
// ---------------------------------------------------------------------------

static size_t dtype_bytes(simaai::neat::TensorDType dtype) {
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

static float read_elem(const uint8_t* data, size_t idx, simaai::neat::TensorDType dtype) {
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

static bool depth_tensor_to_colormap(const simaai::neat::Tensor& t, cv::Mat& bgr_out) {
  if (!t.is_dense())
    return false;

  // Determine H, W from tensor shape by collecting non-unit dimensions.
  // Shape [1,518,1,518] → spatial dims are 518×518.
  int w = 0, h = 0;
  {
    std::vector<int> spatial;
    for (auto d : t.shape)
      if (d > 1)
        spatial.push_back(static_cast<int>(d));
    if (spatial.size() >= 2) {
      h = spatial[0];
      w = spatial[1];
    } else if (spatial.size() == 1) {
      h = spatial[0];
      w = spatial[0];
    }
  }
  if (w <= 0 || h <= 0)
    return false;

  const size_t total_elems = static_cast<size_t>(w) * h;
  const size_t elem_sz = dtype_bytes(t.dtype);
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  std::cout << "[DEPTH] Resolved H=" << h << " W=" << w << " total_elems=" << total_elems
            << " elem_sz=" << elem_sz << " raw_bytes=" << raw.size()
            << " expected_bytes=" << (total_elems * elem_sz) << "\n";
  if (raw.size() < total_elems * elem_sz)
    return false;

  cv::Mat depth_f(h, w, CV_32FC1);
  float minv = std::numeric_limits<float>::infinity();
  float maxv = -std::numeric_limits<float>::infinity();

  for (int y = 0; y < h; ++y) {
    float* row = depth_f.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      size_t idx = static_cast<size_t>(x) * h + y;
      float v = read_elem(raw.data(), idx, t.dtype);
      row[x] = v;
      minv = std::min(minv, v);
      maxv = std::max(maxv, v);
    }
  }

  std::cout << "[DEPTH] Value range: min=" << minv << " max=" << maxv << "\n";

  cv::Mat depth_u8;
  if (std::isfinite(minv) && std::isfinite(maxv) && maxv > minv) {
    cv::normalize(depth_f, depth_u8, 0, 255, cv::NORM_MINMAX);
    depth_u8.convertTo(depth_u8, CV_8U);
  } else {
    depth_u8 = cv::Mat(h, w, CV_8U, cv::Scalar(0));
  }

  cv::applyColorMap(depth_u8, bgr_out, cv::COLORMAP_INFERNO);
  return true;
}

static std::vector<simaai::neat::Tensor> tensors_from_sample(const simaai::neat::Sample& sample) {
  if (sample.kind == simaai::neat::SampleKind::Tensor && sample.tensor.has_value()) {
    return {*sample.tensor};
  }
  if (sample.kind == simaai::neat::SampleKind::Bundle) {
    std::vector<simaai::neat::Tensor> out;
    out.reserve(sample.fields.size());
    for (const auto& field : sample.fields) {
      if (field.kind != simaai::neat::SampleKind::Tensor || !field.tensor.has_value())
        throw std::runtime_error("bundle field missing tensor");
      out.push_back(*field.tensor);
    }
    return out;
  }
  throw std::runtime_error("expected tensor output");
}

static bool is_image(const fs::path& p) {
  auto ext = p.extension().string();
  for (auto& c : ext)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <input_dir> <output_dir>\n";
    return 1;
  }

  const std::string tar_gz = argv[1];
  const fs::path input_dir = argv[2];
  const fs::path output_dir = argv[3];

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  fs::create_directories(output_dir);

  // Collect image paths
  std::vector<fs::path> images;
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && is_image(entry.path()))
      images.push_back(entry.path());
  }
  std::sort(images.begin(), images.end());

  if (images.empty()) {
    std::cerr << "No images found in " << input_dir << "\n";
    return 3;
  }
  std::cout << "Found " << images.size() << " images\n";

  // Configure model (518x518 input from MPK transform table)
  constexpr int kInferSize = 518;

  std::cout << "[BUILD] Loading model...\n";
  simaai::neat::Model::Options model_opt;
  model_opt.media_type = "video/x-raw";
  model_opt.format = "RGB";
  model_opt.input_max_width = kInferSize;
  model_opt.input_max_height = kInferSize;
  model_opt.input_max_depth = 3;
  simaai::neat::Model model(tar_gz, model_opt);
  std::cout << "[BUILD] Model loaded\n";

  // Build session manually (like ANPR pattern) so we can pass a dummy tensor
  simaai::neat::Session sess;
  sess.add(model.session());
  std::cout << "[BUILD] Pipeline:\n" << sess.describe_backend() << "\n";

  // Create dummy input tensor for build()
  cv::Mat dummy_rgb(kInferSize, kInferSize, CV_8UC3, cv::Scalar(0, 0, 0));
  simaai::neat::Tensor dummy_input = simaai::neat::from_cv_mat(
      dummy_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, /*read_only=*/true);

  simaai::neat::RunOptions run_opt;
  run_opt.queue_depth = 4;
  run_opt.overflow_policy = simaai::neat::OverflowPolicy::Block;

  std::cout << "[BUILD] Building pipeline...\n";
  auto run = sess.build(dummy_input, simaai::neat::RunMode::Async, run_opt);
  std::cout << "[BUILD] Pipeline built\n";

  int processed = 0;
  for (const auto& img_path : images) {
    cv::Mat bgr = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::cerr << "Skipping unreadable: " << img_path.filename() << "\n";
      continue;
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, {kInferSize, kInferSize});

    simaai::neat::Tensor input = simaai::neat::from_cv_mat(
        rgb, simaai::neat::ImageSpec::PixelFormat::RGB, /*read_only=*/true);

    std::cout << "[INFER] Pushing " << img_path.filename() << "...\n";
    if (!run.push(input)) {
      std::cerr << "Push failed for " << img_path.filename() << "\n";
      continue;
    }

    std::cout << "[INFER] Pulling...\n";
    auto out_opt = run.pull();
    if (!out_opt.has_value()) {
      std::cerr << "Pull failed for " << img_path.filename() << "\n";
      continue;
    }
    std::cout << "[INFER] Got output\n";

    auto tensors = tensors_from_sample(*out_opt);
    if (tensors.empty()) {
      std::cerr << "No output tensors for " << img_path.filename() << "\n";
      continue;
    }

    const auto& t = tensors.front();
    std::cout << "[INFER] Output tensor: dims=" << t.shape.size();
    for (size_t i = 0; i < t.shape.size(); ++i)
      std::cout << (i == 0 ? " shape=[" : ",") << t.shape[i];
    if (!t.shape.empty())
      std::cout << "]";
    std::cout << " dtype=" << static_cast<int>(t.dtype) << "\n";

    cv::Mat colormap;
    if (!depth_tensor_to_colormap(t, colormap)) {
      std::cerr << "Depth conversion failed for " << img_path.filename() << "\n";
      continue;
    }

    // Create side-by-side subplot: original (left) | depth colormap (right)
    cv::Mat input_resized;
    cv::resize(bgr, input_resized, {colormap.cols, colormap.rows});
    cv::Mat subplot(colormap.rows, colormap.cols * 2, CV_8UC3);
    input_resized.copyTo(subplot(cv::Rect(0, 0, colormap.cols, colormap.rows)));
    colormap.copyTo(subplot(cv::Rect(colormap.cols, 0, colormap.cols, colormap.rows)));

    fs::path out_path = output_dir / (img_path.stem().string() + ".png");
    cv::imwrite(out_path.string(), subplot);
    std::cout << "[" << ++processed << "/" << images.size() << "] " << img_path.filename() << " -> "
              << out_path.filename() << "\n";
  }

  run.close();
  std::cout << "Done: " << processed << " images processed\n";
  return 0;
}
