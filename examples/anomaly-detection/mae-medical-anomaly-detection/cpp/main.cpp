/**
 * @example mae-medical-anomaly-detection.cpp
 * MAE-based medical anomaly detection pipeline using two NEAT models.
 *
 * Usage: mae-medical-anomaly-detection <mae_model.tar.gz> <cls_model.tar.gz> <input.npy> [<input2.npy> ...] [--output-dir <dir>]
 */
#include "neat.h"
#include "support/runtime/example_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kPatchSize = 16;
constexpr int kInferHeight = 224;
constexpr int kInferWidth = 224;
constexpr int kTimeoutMs = 5000;

// Simple .npy file loader for float32 arrays
// Supports only little-endian, C-contiguous, float32 arrays
// Returns data and shape as a pair
std::pair<std::vector<float>, std::vector<int64_t>> load_npy_float32_with_shape(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open .npy file: " + path.string());
  }

  // Read magic string
  char magic[6];
  file.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Invalid .npy file: " + path.string());
  }

  // Read version (1 byte)
  uint8_t version_major, version_minor;
  file.read(reinterpret_cast<char*>(&version_major), 1);
  file.read(reinterpret_cast<char*>(&version_minor), 1);

  // Read header length
  uint16_t header_len = 0;
  if (version_major == 1) {
    file.read(reinterpret_cast<char*>(&header_len), 2);
  } else {
    uint32_t header_len_32 = 0;
    file.read(reinterpret_cast<char*>(&header_len_32), 4);
    header_len = static_cast<uint16_t>(header_len_32);
  }

  // Read header
  std::string header(header_len, '\0');
  file.read(header.data(), header_len);

  // Parse shape and dtype from header (simplified parser)
  // Expected format: {'descr': '<f4' or '<f8', 'fortran_order': False, 'shape': (H, W) or (H, W, C), ...}
  
  // Parse dtype
  bool is_float64 = false;
  size_t descr_start = header.find("'descr'");
  if (descr_start != std::string::npos) {
    size_t descr_value_start = header.find("'", descr_start + 7);
    if (descr_value_start != std::string::npos) {
      size_t descr_value_end = header.find("'", descr_value_start + 1);
      if (descr_value_end != std::string::npos) {
        std::string descr = header.substr(descr_value_start + 1, descr_value_end - descr_value_start - 1);
        if (descr.find("f8") != std::string::npos || descr.find("float64") != std::string::npos) {
          is_float64 = true;
        }
      }
    }
  }
  
  size_t shape_start = header.find("'shape'");
  if (shape_start == std::string::npos) {
    throw std::runtime_error("Could not find shape in .npy header");
  }

  size_t shape_open = header.find('(', shape_start);
  size_t shape_close = header.find(')', shape_open);
  if (shape_open == std::string::npos || shape_close == std::string::npos) {
    throw std::runtime_error("Invalid shape in .npy header");
  }

  std::vector<int64_t> shape;
  std::string shape_str = header.substr(shape_open + 1, shape_close - shape_open - 1);
  size_t pos = 0;
  while (pos < shape_str.size()) {
    while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) {
      ++pos;
    }
    if (pos >= shape_str.size()) break;
    size_t num_start = pos;
    while (pos < shape_str.size() && std::isdigit(static_cast<unsigned char>(shape_str[pos]))) {
      ++pos;
    }
    if (pos > num_start) {
      shape.push_back(std::stoll(shape_str.substr(num_start, pos - num_start)));
    }
  }

  if (shape.empty()) {
    throw std::runtime_error("Empty shape in .npy file");
  }

  // Calculate total elements
  int64_t total_elems = 1;
  for (int64_t dim : shape) {
    total_elems *= dim;
  }

  // Read data (handle both float32 and float64)
  std::vector<float> data(total_elems);
  if (is_float64) {
    std::vector<double> data_double(total_elems);
    file.read(reinterpret_cast<char*>(data_double.data()), total_elems * sizeof(double));
    if (file.gcount() != static_cast<std::streamsize>(total_elems * sizeof(double))) {
      throw std::runtime_error("Unexpected file size in .npy file (float64)");
    }
    // Convert double to float
    for (int64_t i = 0; i < total_elems; ++i) {
      data[static_cast<size_t>(i)] = static_cast<float>(data_double[static_cast<size_t>(i)]);
    }
  } else {
    file.read(reinterpret_cast<char*>(data.data()), total_elems * sizeof(float));
    if (file.gcount() != static_cast<std::streamsize>(total_elems * sizeof(float))) {
      throw std::runtime_error("Unexpected file size in .npy file (float32)");
    }
  }

  return {data, shape};
}

std::vector<float> load_npy_float32(const fs::path& path) {
  return load_npy_float32_with_shape(path).first;
}

cv::Mat load_npy_as_rgb(const fs::path& path) {
  auto [npy_data, shape] = load_npy_float32_with_shape(path);

  // Parse dimensions from shape
  int h = 0, w = 0, c = 1;
  if (shape.size() == 2) {
    // (H, W)
    h = static_cast<int>(shape[0]);
    w = static_cast<int>(shape[1]);
    c = 1;
  } else if (shape.size() == 3) {
    // (H, W, C) or (C, H, W) - assume (H, W, C) for medical images
    h = static_cast<int>(shape[0]);
    w = static_cast<int>(shape[1]);
    c = static_cast<int>(shape[2]);
  } else if (shape.size() == 1) {
    // 1D array - try to infer square
    int64_t total = shape[0];
    int64_t side = static_cast<int64_t>(std::sqrt(total));
    if (side * side == total) {
      h = static_cast<int>(side);
      w = static_cast<int>(side);
      c = 1;
    } else {
      throw std::runtime_error("Cannot infer 2D dimensions from 1D shape");
    }
  } else {
    throw std::runtime_error("Unsupported shape dimensions: " + std::to_string(shape.size()));
  }

  if (h <= 0 || w <= 0) {
    throw std::runtime_error("Invalid dimensions from .npy file: h=" + std::to_string(h) +
                             " w=" + std::to_string(w));
  }

  // Extract first channel if multi-channel
  cv::Mat img_float(h, w, CV_32FC1);
  if (c == 1) {
    std::memcpy(img_float.data, npy_data.data(), h * w * sizeof(float));
  } else {
    // Take first channel (assuming HWC layout)
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        img_float.at<float>(y, x) = npy_data[static_cast<size_t>(y * w * c + x * c)];
      }
    }
  }

  // Normalize to [0, 255] uint8
  double min_val, max_val;
  cv::minMaxLoc(img_float, &min_val, &max_val);
  cv::Mat img_u8;
  if (std::abs(max_val - min_val) < 1e-8) {
    img_float.convertTo(img_u8, CV_8UC1, 0.0);
  } else {
    img_float.convertTo(img_u8, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
  }

  // Resize to model input size
  cv::Mat resized;
  cv::resize(img_u8, resized, cv::Size(kInferWidth, kInferHeight), 0, 0, cv::INTER_LINEAR);

  // Duplicate to 3 channels (RGB)
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);

  return rgb;
}

std::vector<float> get_deterministic_mask(int num_patches, int seed = 42) {
  std::mt19937 gen(static_cast<unsigned int>(seed));
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  std::vector<float> random_vals(num_patches);
  for (int i = 0; i < num_patches; ++i) {
    random_vals[i] = dis(gen);
  }

  std::vector<float> mask(num_patches);
  const int step = 4;
  for (int i = 0; i < num_patches; i += step) {
    float max_val = *std::max_element(random_vals.begin() + i,
                                      random_vals.begin() + std::min(i + step, num_patches));
    for (int j = i; j < std::min(i + step, num_patches); ++j) {
      mask[j] = (random_vals[j] != max_val) ? 1.0f : 0.0f;  // 1 = remove, 0 = keep
    }
  }

  return mask;
}

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("Expected Float32 tensor");
  }
  std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  if (raw.empty()) {
    throw std::runtime_error("Tensor output is empty");
  }
  const size_t bytes = raw.size();
  if (bytes % sizeof(float) != 0) {
    throw std::runtime_error("Tensor size is not a multiple of float");
  }
  const size_t elems = bytes / sizeof(float);
  std::vector<float> out(elems);
  std::memcpy(out.data(), raw.data(), elems * sizeof(float));
  return out;
}

std::vector<float> stable_softmax(const std::vector<float>& logits) {
  if (logits.empty()) {
    return {};
  }
  float max_val = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  std::vector<float> exp_vals(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    exp_vals[i] = std::exp(static_cast<double>(logits[i] - max_val));
    sum += exp_vals[i];
  }
  if (sum == 0.0) {
    throw std::runtime_error("Softmax sum is zero");
  }
  std::vector<float> probs(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = static_cast<float>(exp_vals[i] / sum);
  }
  return probs;
}

cv::Mat reshape_reconstruction(const std::vector<float>& recon_raw, int h, int w) {
  cv::Mat recon;
  if (recon_raw.size() == static_cast<size_t>(h * w)) {
    // (H, W)
    recon = cv::Mat(h, w, CV_32FC1);
    std::memcpy(recon.data, recon_raw.data(), h * w * sizeof(float));
  } else if (recon_raw.size() == static_cast<size_t>(h * w * 1)) {
    // (H, W, 1) or (1, H, W)
    recon = cv::Mat(h, w, CV_32FC1);
    std::memcpy(recon.data, recon_raw.data(), h * w * sizeof(float));
  } else if (recon_raw.size() == static_cast<size_t>(1 * h * w)) {
    // (1, H, W)
    recon = cv::Mat(h, w, CV_32FC1);
    std::memcpy(recon.data, recon_raw.data(), h * w * sizeof(float));
  } else if (recon_raw.size() == static_cast<size_t>(h * w * 3)) {
    // (H, W, 3) or (3, H, W)
    // Try (H, W, 3) first
    recon = cv::Mat(h, w, CV_32FC3);
    std::memcpy(recon.data, recon_raw.data(), h * w * 3 * sizeof(float));
    // Convert to single channel by taking first channel
    cv::Mat channels[3];
    cv::split(recon, channels);
    recon = channels[0];
  } else if (recon_raw.size() == static_cast<size_t>(3 * h * w)) {
    // (3, H, W) - transpose needed
    cv::Mat recon_chw(3, h * w, CV_32FC1);
    std::memcpy(recon_chw.data, recon_raw.data(), 3 * h * w * sizeof(float));
    cv::Mat recon_hwc(h, w, CV_32FC3);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          recon_hwc.at<cv::Vec3f>(y, x)[c] = recon_chw.at<float>(c, y * w + x);
        }
      }
    }
    cv::Mat channels[3];
    cv::split(recon_hwc, channels);
    recon = channels[0];
  } else {
    throw std::runtime_error("Unexpected reconstruction tensor size: " +
                             std::to_string(recon_raw.size()));
  }

  // Normalize if needed
  double min_val, max_val;
  cv::minMaxLoc(recon, &min_val, &max_val);
  if (max_val > 1.0f) {
    recon = recon / 255.0f;
  }

  return recon;
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <mae_model.tar.gz> <cls_model.tar.gz> <input.npy> [<input2.npy> ...] "
                 "[--output-dir <dir>]\n";
    return 1;
  }

  std::string mae_model_path = argv[1];
  std::string cls_model_path = argv[2];
  std::vector<fs::path> input_paths;
  fs::path output_dir;

  // Parse arguments
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--output-dir" && i + 1 < argc) {
      output_dir = argv[++i];
    } else {
      input_paths.push_back(argv[i]);
    }
  }

  if (input_paths.empty()) {
    std::cerr << "No input .npy files provided\n";
    return 1;
  }

  if (!output_dir.empty()) {
    fs::create_directories(output_dir);
  }

  const int h = kInferHeight;
  const int w = kInferWidth;

  // Configure models
  simaai::neat::Model::Options mae_opt;
  mae_opt.media_type = "video/x-raw";
  mae_opt.format = "RGB";
  mae_opt.input_max_width = w;
  mae_opt.input_max_height = h;
  mae_opt.input_max_depth = 3;

  simaai::neat::Model::Options cls_opt;
  cls_opt.media_type = "video/x-raw";
  cls_opt.format = "RGB";
  cls_opt.input_max_width = w;
  cls_opt.input_max_height = h;
  cls_opt.input_max_depth = 3;

  std::cout << "[LOAD] MAE model: " << mae_model_path << "\n";
  std::cout << "[LOAD] Classifier model: " << cls_model_path << "\n";
  simaai::neat::Model mae_model(mae_model_path, mae_opt);
  simaai::neat::Model cls_model(cls_model_path, cls_opt);

  // Precompute deterministic mask
  const int num_patches = (h / kPatchSize) * (w / kPatchSize);
  std::vector<float> mask = get_deterministic_mask(num_patches, 42);
  const int grid_h = h / kPatchSize;
  const int grid_w = w / kPatchSize;

  // Expand mask to pixel level
  cv::Mat mask_px(h, w, CV_32FC1, cv::Scalar(0.0f));
  for (int py = 0; py < grid_h; ++py) {
    for (int px = 0; px < grid_w; ++px) {
      const int patch_idx = py * grid_w + px;
      const float mask_val = mask[static_cast<size_t>(patch_idx)];
      for (int dy = 0; dy < kPatchSize; ++dy) {
        for (int dx = 0; dx < kPatchSize; ++dx) {
          const int y = py * kPatchSize + dy;
          const int x = px * kPatchSize + dx;
          if (y < h && x < w) {
            mask_px.at<float>(y, x) = mask_val;
          }
        }
      }
    }
  }

  // Process each input
  for (const auto& npy_path : input_paths) {
    std::cout << "\n--- Processing: " << npy_path << " ---\n";

    try {
      // Load and preprocess
      cv::Mat rgb = load_npy_as_rgb(npy_path);
      
      // Debug: print image stats
      double img_min, img_max, img_mean;
      cv::minMaxLoc(rgb, &img_min, &img_max);
      img_mean = cv::mean(rgb)[0];
      std::cout << "Input image stats: min=" << img_min << " max=" << img_max 
                << " mean=" << img_mean << "\n";
      
      simaai::neat::Tensor img_tensor =
          simaai::neat::from_cv_mat(rgb, simaai::neat::ImageSpec::PixelFormat::RGB,
                                    /*read_only=*/true);

      // Stage 1: MAE reconstruction
      auto mae_out = mae_model.run(img_tensor, kTimeoutMs);
      if (!mae_out.tensor.has_value()) {
        std::cerr << "MAE model returned empty output\n";
        continue;
      }

      std::vector<float> recon_raw = tensor_to_floats(*mae_out.tensor);
      std::cout << "MAE recon raw size: " << recon_raw.size() << " elements\n";
      std::cout << "MAE recon tensor shape: [";
      for (size_t i = 0; i < mae_out.tensor->shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << mae_out.tensor->shape[i];
      }
      std::cout << "]\n";

      // Convert RGB to float [0, 1]
      cv::Mat img_float;
      rgb.convertTo(img_float, CV_32FC3, 1.0 / 255.0);

      // Reshape reconstruction using actual tensor shape
      cv::Mat recon_np;
      const auto& tensor_shape = mae_out.tensor->shape;
      if (tensor_shape.size() == 2) {
        // (H, W)
        recon_np = cv::Mat(static_cast<int>(tensor_shape[0]), static_cast<int>(tensor_shape[1]), CV_32FC1);
        std::memcpy(recon_np.data, recon_raw.data(), recon_raw.size() * sizeof(float));
      } else if (tensor_shape.size() == 3) {
        if (tensor_shape[0] == 1) {
          // (1, H, W)
          recon_np = cv::Mat(static_cast<int>(tensor_shape[1]), static_cast<int>(tensor_shape[2]), CV_32FC1);
          std::memcpy(recon_np.data, recon_raw.data(), recon_raw.size() * sizeof(float));
        } else if (tensor_shape[2] == 1 || tensor_shape[2] == 3) {
          // (H, W, C)
          int recon_h = static_cast<int>(tensor_shape[0]);
          int recon_w = static_cast<int>(tensor_shape[1]);
          int recon_c = static_cast<int>(tensor_shape[2]);
          cv::Mat recon_hwc(recon_h, recon_w, recon_c == 3 ? CV_32FC3 : CV_32FC1);
          std::memcpy(recon_hwc.data, recon_raw.data(), recon_raw.size() * sizeof(float));
          if (recon_c == 3) {
            cv::Mat channels[3];
            cv::split(recon_hwc, channels);
            recon_np = channels[0];
          } else {
            recon_np = recon_hwc;
          }
        } else {
          // (C, H, W) - transpose needed
          int recon_c = static_cast<int>(tensor_shape[0]);
          int recon_h = static_cast<int>(tensor_shape[1]);
          int recon_w = static_cast<int>(tensor_shape[2]);
          cv::Mat recon_chw(recon_c, recon_h * recon_w, CV_32FC1);
          std::memcpy(recon_chw.data, recon_raw.data(), recon_raw.size() * sizeof(float));
          cv::Mat recon_hwc(recon_h, recon_w, recon_c == 3 ? CV_32FC3 : CV_32FC1);
          for (int c = 0; c < recon_c; ++c) {
            for (int y = 0; y < recon_h; ++y) {
              for (int x = 0; x < recon_w; ++x) {
                if (recon_c == 3) {
                  recon_hwc.at<cv::Vec3f>(y, x)[c] = recon_chw.at<float>(c, y * recon_w + x);
                } else {
                  recon_hwc.at<float>(y, x) = recon_chw.at<float>(c, y * recon_w + x);
                }
              }
            }
          }
          if (recon_c == 3) {
            cv::Mat channels[3];
            cv::split(recon_hwc, channels);
            recon_np = channels[0];
          } else {
            recon_np = recon_hwc;
          }
        }
      } else if (tensor_shape.size() == 4 && tensor_shape[0] == 1) {
        // (1, C, H, W) or (1, H, W, C)
        // For [1, 224, 224, 1], this is (1, H, W, C) with C=1
        // For [1, 1, 224, 224] or [1, 3, 224, 224], this is (1, C, H, W)
        if (tensor_shape[1] == 1 || tensor_shape[1] == 3) {
          // (1, C, H, W) - e.g., [1, 1, 224, 224] or [1, 3, 224, 224]
          int recon_c = static_cast<int>(tensor_shape[1]);
          int recon_h = static_cast<int>(tensor_shape[2]);
          int recon_w = static_cast<int>(tensor_shape[3]);
          // Skip batch dimension: data starts at offset 0 (contiguous)
          cv::Mat recon_chw(recon_c, recon_h * recon_w, CV_32FC1);
          std::memcpy(recon_chw.data, recon_raw.data(), recon_c * recon_h * recon_w * sizeof(float));
          cv::Mat recon_hwc(recon_h, recon_w, recon_c == 3 ? CV_32FC3 : CV_32FC1);
          for (int c = 0; c < recon_c; ++c) {
            for (int y = 0; y < recon_h; ++y) {
              for (int x = 0; x < recon_w; ++x) {
                if (recon_c == 3) {
                  recon_hwc.at<cv::Vec3f>(y, x)[c] = recon_chw.at<float>(c, y * recon_w + x);
                } else {
                  recon_hwc.at<float>(y, x) = recon_chw.at<float>(c, y * recon_w + x);
                }
              }
            }
          }
          if (recon_c == 3) {
            cv::Mat channels[3];
            cv::split(recon_hwc, channels);
            recon_np = channels[0];
          } else {
            recon_np = recon_hwc;
          }
        } else {
          // (1, H, W, C) - e.g., [1, 224, 224, 1] or [1, 224, 224, 3]
          int recon_h = static_cast<int>(tensor_shape[1]);
          int recon_w = static_cast<int>(tensor_shape[2]);
          int recon_c = static_cast<int>(tensor_shape[3]);
          // Skip batch dimension: data is contiguous, just extract HWC part
          cv::Mat recon_hwc(recon_h, recon_w, recon_c == 3 ? CV_32FC3 : CV_32FC1);
          std::memcpy(recon_hwc.data, recon_raw.data(), recon_h * recon_w * recon_c * sizeof(float));
          if (recon_c == 3) {
            cv::Mat channels[3];
            cv::split(recon_hwc, channels);
            recon_np = channels[0];
          } else {
            recon_np = recon_hwc;
          }
        }
      } else {
        // Fallback to old reshape function
        recon_np = reshape_reconstruction(recon_raw, h, w);
      }

      // Resize if needed
      if (recon_np.rows != h || recon_np.cols != w) {
        cv::Mat resized;
        cv::resize(recon_np, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        recon_np = resized;
      }

      // Normalize if needed
      double min_val, max_val;
      cv::minMaxLoc(recon_np, &min_val, &max_val);
      if (max_val > 1.0f) {
        recon_np = recon_np / 255.0f;
      }

      // Paste reconstruction: im_paste = img * (1 - mask) + recon * mask
      // mask_px: 1.0 = remove (use recon), 0.0 = keep (use original)
      cv::Mat mask_3ch;
      cv::cvtColor(mask_px, mask_3ch, cv::COLOR_GRAY2RGB);
      
      // Create (1 - mask) matrix
      cv::Mat ones_mask = cv::Mat::ones(h, w, CV_32FC3);
      cv::Mat inv_mask_3ch;
      cv::subtract(ones_mask, mask_3ch, inv_mask_3ch);
      
      // img * (1 - mask)
      cv::Mat img_masked;
      cv::multiply(img_float, inv_mask_3ch, img_masked);
      
      // recon * mask
      cv::Mat recon_3ch;
      cv::cvtColor(recon_np, recon_3ch, cv::COLOR_GRAY2RGB);
      cv::Mat recon_masked;
      cv::multiply(recon_3ch, mask_3ch, recon_masked);
      
      // Combine: im_paste = img * (1 - mask) + recon * mask
      cv::Mat im_paste;
      cv::add(img_masked, recon_masked, im_paste);

      // Stage 2: Anomaly classification on diff
      cv::Mat diff;
      cv::absdiff(img_float, im_paste, diff);
      cv::Mat diff_u8;
      diff.convertTo(diff_u8, CV_8UC3, 255.0);

      simaai::neat::Tensor diff_tensor =
          simaai::neat::from_cv_mat(diff_u8, simaai::neat::ImageSpec::PixelFormat::RGB,
                                    /*read_only=*/true);

      auto cls_out = cls_model.run(diff_tensor, kTimeoutMs);
      if (!cls_out.tensor.has_value()) {
        std::cerr << "Classifier model returned empty output\n";
        continue;
      }

      std::vector<float> logits = tensor_to_floats(*cls_out.tensor);
      if (logits.size() < 2) {
        std::cerr << "Unexpected classifier output size: " << logits.size() << " (expected at least 2)\n";
        continue;
      }

      std::vector<float> probs = stable_softmax(logits);
      const float normal_prob = probs[0];
      const float anomalous_prob = probs[1];
      const std::string label = (normal_prob > anomalous_prob) ? "normal" : "anomalous";

      std::cout << "===== Inference Result =====\n";
      std::cout << "  Normal prob : " << std::fixed << std::setprecision(4) << normal_prob << "\n";
      std::cout << "  Anomaly prob: " << std::fixed << std::setprecision(4) << anomalous_prob << "\n";
      std::cout << "  Prediction  : " << label << "\n";
      std::cout << "============================\n";

      // Save visualization if output dir specified
      if (!output_dir.empty()) {
        cv::Mat vis_img(h, w * 3, CV_8UC3);
        cv::Mat orig_u8;
        img_float.convertTo(orig_u8, CV_8UC3, 255.0);
        cv::Mat recon_u8;
        recon_np.convertTo(recon_u8, CV_8UC1, 255.0);
        cv::Mat recon_u8_3ch;
        cv::cvtColor(recon_u8, recon_u8_3ch, cv::COLOR_GRAY2RGB);

        orig_u8.copyTo(vis_img(cv::Rect(0, 0, w, h)));
        recon_u8_3ch.copyTo(vis_img(cv::Rect(w, 0, w, h)));
        diff_u8.copyTo(vis_img(cv::Rect(w * 2, 0, w, h)));

        // Add text labels
        cv::putText(vis_img, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
        cv::putText(vis_img, "Reconstruction", cv::Point(w + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);
        std::string diff_label = "Diff | " + label + " (" + std::to_string(anomalous_prob).substr(0, 4) + ")";
        cv::putText(vis_img, diff_label, cv::Point(w * 2 + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);

        fs::path out_path = output_dir / (npy_path.stem().string() + "_viz.png");
        if (!cv::imwrite(out_path.string(), vis_img)) {
          std::cerr << "Failed to write visualization: " << out_path << "\n";
        } else {
          std::cout << "Saved visualization: " << out_path << "\n";
        }
      }

    } catch (const std::exception& e) {
      std::cerr << "Error processing " << npy_path << ": " << e.what() << "\n";
      continue;
    }
  }

  return 0;
}
