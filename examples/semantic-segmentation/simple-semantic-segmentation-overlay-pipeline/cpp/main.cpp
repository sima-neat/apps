/**
 * @example simple-semantic-segmentation-overlay-pipeline.cpp
 * Minimal semantic-segmentation overlay example for FCN-HRNet models.
 *
 * Usage:
 *   simple-semantic-segmentation-overlay-pipeline <model.tar.gz> <input_dir> <output_dir>
 */
#include "neat.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

const simaai::neat::Tensor* find_first_tensor(const simaai::neat::Sample& s) {
  if (s.kind == simaai::neat::SampleKind::Tensor && s.tensor.has_value()) {
    return &(*s.tensor);
  }
  if (s.kind == simaai::neat::SampleKind::Bundle) {
    for (const auto& field : s.fields) {
      if (const auto* t = find_first_tensor(field)) {
        return t;
      }
    }
  }
  return nullptr;
}

cv::Vec3b color_for_class(int class_id) {
  static const cv::Vec3b kVoc21Palette[21] = {
      cv::Vec3b(0, 0, 0),     // background
      cv::Vec3b(0, 0, 128),   // aeroplane
      cv::Vec3b(0, 128, 0),   // bicycle
      cv::Vec3b(0, 128, 128), // bird
      cv::Vec3b(128, 0, 0),   // boat
      cv::Vec3b(128, 0, 128), // bottle
      cv::Vec3b(0, 255, 0),   // bus (green)
      cv::Vec3b(255, 0, 0),   // car (blue)
      cv::Vec3b(0, 0, 64),    // cat
      cv::Vec3b(0, 64, 0),    // chair
      cv::Vec3b(0, 64, 64),   // cow
      cv::Vec3b(64, 0, 0),    // diningtable
      cv::Vec3b(64, 0, 64),   // dog
      cv::Vec3b(64, 64, 0),   // horse
      cv::Vec3b(64, 64, 64),  // motorbike
      cv::Vec3b(0, 0, 192),   // person
      cv::Vec3b(0, 192, 0),   // pottedplant
      cv::Vec3b(0, 192, 192), // sheep
      cv::Vec3b(192, 0, 0),   // sofa
      cv::Vec3b(192, 0, 192), // train
      cv::Vec3b(192, 192, 0), // tvmonitor
  };

  if (class_id < 0) {
    return cv::Vec3b(0, 0, 0);
  }
  if (class_id < 21) {
    return kVoc21Palette[class_id];
  }
  return cv::Vec3b(static_cast<uint8_t>((37 * class_id) % 255),
                   static_cast<uint8_t>((67 * class_id) % 255),
                   static_cast<uint8_t>((97 * class_id) % 255));
}

bool logits_to_label_map(const simaai::neat::Tensor& t, cv::Mat& labels, std::string& err) {
  if (!t.is_dense()) {
    err = "output tensor is not dense";
    return false;
  }
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    err = "expected Float32 segmentation logits";
    return false;
  }

  std::vector<int64_t> dims = t.shape;
  if (!dims.empty() && dims[0] == 1 && dims.size() > 3) {
    dims.erase(dims.begin()); // drop batch
  }
  if (!dims.empty() && dims[0] == 1 && dims.size() > 3) {
    dims.erase(dims.begin()); // drop depth=1 if present
  }
  if (dims.size() != 3) {
    err = "unsupported output shape rank";
    return false;
  }

  const int d0 = static_cast<int>(dims[0]);
  const int d1 = static_cast<int>(dims[1]);
  const int d2 = static_cast<int>(dims[2]);
  if (d0 <= 0 || d1 <= 0 || d2 <= 0) {
    err = "invalid output shape";
    return false;
  }

  int class_axis = -1;
  if (d0 == 21)
    class_axis = 0;
  else if (d1 == 21)
    class_axis = 1;
  else if (d2 == 21)
    class_axis = 2;
  else {
    // fallback: assume last dim is channels
    class_axis = 2;
  }

  int h = 0;
  int w = 0;
  int c = 0;
  if (class_axis == 0) {
    c = d0;
    h = d1;
    w = d2;
  } else if (class_axis == 1) {
    h = d0;
    c = d1;
    w = d2;
  } else {
    h = d0;
    w = d1;
    c = d2;
  }

  if (h <= 0 || w <= 0 || c <= 0) {
    err = "invalid output shape";
    return false;
  }

  const std::vector<uint8_t> raw = t.copy_dense_bytes_tight();
  const size_t total = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
  if (raw.size() < total * sizeof(float)) {
    err = "output tensor byte size mismatch";
    return false;
  }

  const float* ptr = reinterpret_cast<const float*>(raw.data());
  labels = cv::Mat(h, w, CV_8UC1, cv::Scalar(0));

  for (int y = 0; y < h; ++y) {
    uint8_t* out_row = labels.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      float best = -std::numeric_limits<float>::infinity();
      int best_id = 0;
      for (int ch = 0; ch < c; ++ch) {
        size_t idx = 0;
        if (class_axis == 0) {
          idx = (static_cast<size_t>(ch) * static_cast<size_t>(h) + static_cast<size_t>(y)) *
                    static_cast<size_t>(w) +
                static_cast<size_t>(x);
        } else if (class_axis == 1) {
          idx = (static_cast<size_t>(y) * static_cast<size_t>(c) + static_cast<size_t>(ch)) *
                    static_cast<size_t>(w) +
                static_cast<size_t>(x);
        } else {
          idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) *
                    static_cast<size_t>(c) +
                static_cast<size_t>(ch);
        }
        const float v = ptr[idx];
        if (v > best) {
          best = v;
          best_id = ch;
        }
      }
      out_row[x] = static_cast<uint8_t>(std::max(0, std::min(255, best_id)));
    }
  }
  return true;
}

std::string class_name(int id) {
  static const char* kVoc21[] = {
      "background", "aeroplane", "bicycle",     "bird",  "boat",  "bottle", "bus",
      "car",        "cat",       "chair",       "cow",   "table", "dog",    "horse",
      "motorbike",  "person",    "pottedplant", "sheep", "sofa",  "train",  "tvmonitor",
  };
  if (id >= 0 && id < 21) {
    return kVoc21[id];
  }
  return "class_" + std::to_string(id);
}

void print_histogram(const cv::Mat& labels, const fs::path& image_path) {
  std::map<int, int> hist;
  for (int y = 0; y < labels.rows; ++y) {
    const uint8_t* row = labels.ptr<uint8_t>(y);
    for (int x = 0; x < labels.cols; ++x) {
      hist[static_cast<int>(row[x])] += 1;
    }
  }
  std::vector<std::pair<int, int>> items(hist.begin(), hist.end());
  std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::cout << "[classes] " << image_path.filename() << ": ";
  const int show = std::min<int>(5, static_cast<int>(items.size()));
  for (int i = 0; i < show; ++i) {
    if (i > 0)
      std::cout << ", ";
    std::cout << class_name(items[i].first) << "=" << items[i].second;
  }
  std::cout << "\n";
}

void draw_overlay(cv::Mat& bgr, const cv::Mat& labels, float alpha = 0.55f) {
  const float inv_alpha = 1.0f - alpha;
  for (int y = 0; y < bgr.rows; ++y) {
    cv::Vec3b* pix = bgr.ptr<cv::Vec3b>(y);
    const uint8_t* cls = labels.ptr<uint8_t>(y);
    for (int x = 0; x < bgr.cols; ++x) {
      const int cid = static_cast<int>(cls[x]);
      if (cid <= 0) {
        continue;
      }
      const cv::Vec3b col = color_for_class(cid);
      pix[x][0] = static_cast<uint8_t>(inv_alpha * pix[x][0] + alpha * col[0]);
      pix[x][1] = static_cast<uint8_t>(inv_alpha * pix[x][1] + alpha * col[1]);
      pix[x][2] = static_cast<uint8_t>(inv_alpha * pix[x][2] + alpha * col[2]);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <input_dir> <output_dir>\n";
    return 1;
  }

  constexpr int kInputW = 512;
  constexpr int kInputH = 512;

  const std::string model_path = argv[1];
  const fs::path input_dir = argv[2];
  const fs::path output_dir = argv[3];

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  fs::create_directories(output_dir);

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

  try {
    simaai::neat::Model::Options model_opt;
    model_opt.media_type = "video/x-raw";
    model_opt.format = "RGB";
    model_opt.preproc.input_width = kInputW;
    model_opt.preproc.input_height = kInputH;
    model_opt.preproc.input_img_type = "RGB";
    model_opt.input_max_width = kInputW;
    model_opt.input_max_height = kInputH;
    model_opt.input_max_depth = 3;

    simaai::neat::Model model(model_path, model_opt);

    simaai::neat::Session session;
    session.add(model.session());

    cv::Mat dummy_rgb(kInputH, kInputW, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy =
        simaai::neat::from_cv_mat(dummy_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);
    auto run = session.build(dummy, simaai::neat::RunMode::Sync);

    std::cout << "Pipeline:\n" << session.describe_backend() << "\n";
    std::cout << "Found " << images.size() << " images\n";

    int ok = 0;
    for (const auto& image_path : images) {
      cv::Mat src_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (src_bgr.empty()) {
        std::cerr << "Skipping unreadable image: " << image_path.filename() << "\n";
        continue;
      }

      cv::Mat resized_bgr;
      cv::resize(src_bgr, resized_bgr, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_LINEAR);

      cv::Mat resized_rgb;
      cv::cvtColor(resized_bgr, resized_rgb, cv::COLOR_BGR2RGB);
      simaai::neat::Tensor input =
          simaai::neat::from_cv_mat(resized_rgb, simaai::neat::ImageSpec::PixelFormat::RGB, true);

      if (!run.push(input)) {
        std::cerr << "Push failed for: " << image_path.filename() << "\n";
        continue;
      }

      std::optional<simaai::neat::Sample> out_opt;
      for (int i = 0; i < 3 && !out_opt.has_value(); ++i) {
        out_opt = run.pull(/*timeout_ms=*/5000);
      }
      if (!out_opt.has_value()) {
        std::cerr << "Pull timeout for: " << image_path.filename() << "\n";
        continue;
      }

      const simaai::neat::Tensor* out_tensor = find_first_tensor(*out_opt);
      if (!out_tensor) {
        std::cerr << "No tensor output for: " << image_path.filename() << "\n";
        continue;
      }

      cv::Mat label_small;
      std::string err;
      if (!logits_to_label_map(*out_tensor, label_small, err)) {
        std::cerr << "Decode failed for " << image_path.filename() << ": " << err << "\n";
        continue;
      }

      cv::Mat label_resized;
      cv::resize(label_small, label_resized, cv::Size(kInputW, kInputH), 0, 0, cv::INTER_NEAREST);
      print_histogram(label_resized, image_path);

      cv::Mat overlay = resized_bgr.clone();
      draw_overlay(overlay, label_resized);

      const fs::path out_path = output_dir / (image_path.stem().string() + "_overlay.png");
      if (!cv::imwrite(out_path.string(), overlay)) {
        std::cerr << "Failed to write: " << out_path << "\n";
        continue;
      }

      std::cout << "Wrote: " << out_path << "\n";
      ++ok;
    }

    run.close();
    std::cout << "Processed " << ok << " / " << images.size() << " images\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
