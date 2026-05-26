/**
 * @example human-pose-estimator.cpp
 * Lightweight OpenPose sync pipeline: infer pose keypoints for every image in a folder.
 *
 * Usage: human-pose-estimator [--config <path>] [--profile]
 */
#include "utils/config.h"
#include "neat.h"
#include "support/runtime/example_utils.h"

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
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Fixed constants
constexpr int kNumParts = 18;
constexpr int kNumPairs = 19;

// Skeleton limb pairs (part_a, part_b)
constexpr std::array<std::pair<int, int>, kNumPairs> kPosePairs = {{{1, 2},
                                                                    {1, 5},
                                                                    {2, 3},
                                                                    {3, 4},
                                                                    {5, 6},
                                                                    {6, 7},
                                                                    {1, 8},
                                                                    {8, 9},
                                                                    {9, 10},
                                                                    {1, 11},
                                                                    {11, 12},
                                                                    {12, 13},
                                                                    {1, 0},
                                                                    {0, 14},
                                                                    {14, 16},
                                                                    {0, 15},
                                                                    {15, 17},
                                                                    {2, 16},
                                                                    {5, 17}}};

// PAF channel indices (x_channel, y_channel) for each limb pair
constexpr std::array<std::pair<int, int>, kNumPairs> kPafChannels = {{{12, 13},
                                                                      {20, 21},
                                                                      {14, 15},
                                                                      {16, 17},
                                                                      {22, 23},
                                                                      {24, 25},
                                                                      {0, 1},
                                                                      {2, 3},
                                                                      {4, 5},
                                                                      {6, 7},
                                                                      {8, 9},
                                                                      {10, 11},
                                                                      {28, 29},
                                                                      {30, 31},
                                                                      {34, 35},
                                                                      {32, 33},
                                                                      {36, 37},
                                                                      {18, 19},
                                                                      {26, 27}}};

// Keypoint colors (BGR)
constexpr std::array<std::array<int, 3>, kNumParts> kKpColors = {{{255, 0, 0},
                                                                  {255, 85, 0},
                                                                  {255, 170, 0},
                                                                  {255, 255, 0},
                                                                  {170, 255, 0},
                                                                  {85, 255, 0},
                                                                  {0, 255, 0},
                                                                  {0, 255, 85},
                                                                  {0, 255, 170},
                                                                  {0, 255, 255},
                                                                  {0, 170, 255},
                                                                  {0, 85, 255},
                                                                  {0, 0, 255},
                                                                  {85, 0, 255},
                                                                  {170, 0, 255},
                                                                  {255, 0, 255},
                                                                  {255, 0, 170},
                                                                  {255, 0, 85}}};

struct TensorHWC {
  int h = 0;
  int w = 0;
  int c = 0;
  std::vector<float> data;

  float at(int y, int x, int ch) const {
    return data[static_cast<size_t>((y * w + x) * c + ch)];
  }
};

struct Keypoint {
  int id = -1;
  float x = 0.0f;
  float y = 0.0f;
  int grid_x = 0;
  int grid_y = 0;
  float score = 0.0f;
  int part_id = -1;
};

struct Person {
  std::array<int, kNumParts> kpt_indices;
  float total_score = 0.0f;
  int valid_joints_count = 0;

  Person() {
    kpt_indices.fill(-1);
  }

  bool has(int part_id) const {
    return kpt_indices[part_id] >= 0;
  }
};

struct Connection {
  int a_idx = -1;
  int b_idx = -1;
  float score = 0.0f;
  int a_id = -1;
  int b_id = -1;
};

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
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

TensorHWC tensor_to_hwc_f32(const simaai::neat::Tensor& t) {
  if (t.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }

  TensorHWC out;
  if (t.shape.size() == 4) {
    if (t.shape[0] != 1) {
      throw std::runtime_error("only batch=1 is supported");
    }
    out.h = static_cast<int>(t.shape[1]);
    out.w = static_cast<int>(t.shape[2]);
    out.c = static_cast<int>(t.shape[3]);
  } else if (t.shape.size() == 3) {
    out.h = static_cast<int>(t.shape[0]);
    out.w = static_cast<int>(t.shape[1]);
    out.c = static_cast<int>(t.shape[2]);
  } else {
    throw std::runtime_error("unexpected tensor rank for HWC decode");
  }

  const size_t elems =
      static_cast<size_t>(out.h) * static_cast<size_t>(out.w) * static_cast<size_t>(out.c);
  const std::vector<uint8_t> bytes = t.copy_dense_bytes_tight();
  if (bytes.size() < elems * sizeof(float)) {
    throw std::runtime_error("tensor byte size is smaller than expected");
  }

  out.data.resize(elems);
  std::memcpy(out.data.data(), bytes.data(), elems * sizeof(float));
  return out;
}

// ---------------------------------------------------------------------------
// Letterbox: resize + pad to preserve aspect ratio
// ---------------------------------------------------------------------------
struct LetterboxResult {
  cv::Mat img;
  float ratio = 1.0f;
  int pad_left = 0;
  int pad_top = 0;
};

LetterboxResult letterbox(const cv::Mat& src, int target_size) {
  LetterboxResult result;
  const int src_h = src.rows;
  const int src_w = src.cols;

  result.ratio = std::min(static_cast<float>(target_size) / static_cast<float>(src_h),
                          static_cast<float>(target_size) / static_cast<float>(src_w));

  const int new_w = static_cast<int>(std::round(static_cast<float>(src_w) * result.ratio));
  const int new_h = static_cast<int>(std::round(static_cast<float>(src_h) * result.ratio));

  cv::Mat resized;
  if (new_w != src_w || new_h != src_h) {
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  } else {
    resized = src;
  }

  const int pad_w = target_size - new_w;
  const int pad_h = target_size - new_h;

  result.pad_top = static_cast<int>(std::floor(static_cast<float>(pad_h) / 2.0f));
  const int bottom = pad_h - result.pad_top;
  result.pad_left = static_cast<int>(std::floor(static_cast<float>(pad_w) / 2.0f));
  const int right = pad_w - result.pad_left;

  cv::copyMakeBorder(resized, result.img, result.pad_top, bottom, result.pad_left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
  return result;
}

// ---------------------------------------------------------------------------
// Upsample heatmap/PAF tensor using bicubic interpolation
// ---------------------------------------------------------------------------
TensorHWC upsample_tensor(const TensorHWC& src, float factor) {
  const int new_h = static_cast<int>(static_cast<float>(src.h) * factor);
  const int new_w = static_cast<int>(static_cast<float>(src.w) * factor);

  TensorHWC dst;
  dst.h = new_h;
  dst.w = new_w;
  dst.c = src.c;
  dst.data.resize(static_cast<size_t>(new_h) * static_cast<size_t>(new_w) *
                  static_cast<size_t>(src.c));

  // Treat the source tensor as a single multi-channel float image.
  // Note: const_cast is safe here because we do not modify the source data.
  cv::Mat src_mat(src.h, src.w, CV_32FC(src.c), const_cast<float*>(src.data.data()));
  // Destination multi-channel image backed directly by dst.data.
  cv::Mat dst_mat(new_h, new_w, CV_32FC(dst.c), dst.data.data());
  // Single bicubic resize over all channels.
  cv::resize(src_mat, dst_mat, dst_mat.size(), 0, 0, cv::INTER_CUBIC);
  return dst;
}

// ---------------------------------------------------------------------------
// Peak extraction with 4-neighbor comparison + spatial NMS
// ---------------------------------------------------------------------------
std::vector<Keypoint> extract_keypoints(const TensorHWC& heatmap, int infer_size,
                                        float keypoint_score_threshold, int nms_radius) {
  const float stride_x = static_cast<float>(infer_size) / static_cast<float>(heatmap.w);
  const float stride_y = static_cast<float>(infer_size) / static_cast<float>(heatmap.h);

  std::vector<Keypoint> all_keypoints;
  int kpt_id = 0;

  for (int part_id = 0; part_id < kNumParts; ++part_id) {
    // Build padded probability map
    const int ph = heatmap.h + 4; // pad by 2 on each side
    const int pw = heatmap.w + 4;
    std::vector<float> padded(static_cast<size_t>(ph) * static_cast<size_t>(pw), 0.0f);

    // Fill interior (offset by 2)
    for (int y = 0; y < heatmap.h; ++y) {
      for (int x = 0; x < heatmap.w; ++x) {
        float val = heatmap.at(y, x, part_id);
        if (val < keypoint_score_threshold)
          val = 0.0f;
        padded[static_cast<size_t>((y + 2) * pw + (x + 2))] = val;
      }
    }

    // 4-neighbor peak finding on the inner region [1..ph-2, 1..pw-2]
    // Then trim the result to [2..ph-3, 2..pw-3] to get original coordinates
    struct Candidate {
      int x, y;
      float score;
    };
    std::vector<Candidate> candidates;

    for (int y = 2; y < ph - 2; ++y) {
      for (int x = 2; x < pw - 2; ++x) {
        const float center = padded[static_cast<size_t>(y * pw + x)];
        if (center <= 0.0f)
          continue;

        // Strictly greater than all 4 neighbors (matching demo.py)
        const float left = padded[static_cast<size_t>(y * pw + (x - 1))];
        const float right_val = padded[static_cast<size_t>(y * pw + (x + 1))];
        const float up = padded[static_cast<size_t>((y - 1) * pw + x)];
        const float down = padded[static_cast<size_t>((y + 1) * pw + x)];

        if (center > left && center > right_val && center > up && center > down) {
          // Map back to original coordinates (subtract pad offset of 2)
          candidates.push_back({x - 2, y - 2, center});
        }
      }
    }

    // Sort by x coordinate (matches demo.py itemgetter(0))
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.x < b.x; });

    // Spatial NMS: suppress peaks within nms_radius pixels
    std::vector<bool> suppressed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (suppressed[i])
        continue;
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (suppressed[j])
          continue;
        const float dx = static_cast<float>(candidates[i].x - candidates[j].x);
        const float dy = static_cast<float>(candidates[i].y - candidates[j].y);
        if (std::hypot(dx, dy) < static_cast<float>(nms_radius)) {
          suppressed[j] = true;
        }
      }
    }

    // Collect surviving keypoints
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (suppressed[i])
        continue;
      Keypoint kp;
      kp.id = kpt_id++;
      kp.x = static_cast<float>(candidates[i].x) * stride_x;
      kp.y = static_cast<float>(candidates[i].y) * stride_y;
      kp.grid_x = candidates[i].x;
      kp.grid_y = candidates[i].y;
      kp.score = candidates[i].score;
      kp.part_id = part_id;
      all_keypoints.push_back(kp);
    }
  }

  return all_keypoints;
}

// ---------------------------------------------------------------------------
// PAF score: line integral between two keypoints
// ---------------------------------------------------------------------------
float compute_paf_score(const TensorHWC& paf, const Keypoint& a, const Keypoint& b, int paf_x_idx,
                        int paf_y_idx, float paf_score_threshold, float paf_success_ratio,
                        int paf_num_samples) {
  const float dx = static_cast<float>(b.grid_x - a.grid_x);
  const float dy = static_cast<float>(b.grid_y - a.grid_y);
  const float distance = std::hypot(dx, dy);

  if (distance < 1e-5f)
    return 0.0f;

  const float vec_x = dx / distance;
  const float vec_y = dy / distance;

  int valid_points = 0;
  float valid_sum = 0.0f;

  for (int s = 0; s < paf_num_samples; ++s) {
    const float t = static_cast<float>(s) / static_cast<float>(paf_num_samples - 1);
    int sx = static_cast<int>(
        std::round(static_cast<float>(a.grid_x) + t * static_cast<float>(b.grid_x - a.grid_x)));
    int sy = static_cast<int>(
        std::round(static_cast<float>(a.grid_y) + t * static_cast<float>(b.grid_y - a.grid_y)));

    sx = std::clamp(sx, 0, paf.w - 1);
    sy = std::clamp(sy, 0, paf.h - 1);

    const float paf_vx = paf.at(sy, sx, paf_x_idx);
    const float paf_vy = paf.at(sy, sx, paf_y_idx);
    const float dot = paf_vx * vec_x + paf_vy * vec_y;

    if (dot > paf_score_threshold) {
      valid_sum += dot;
      ++valid_points;
    }
  }

  float paf_score = (valid_points > 0) ? (valid_sum / static_cast<float>(valid_points)) : 0.0f;

  if (static_cast<float>(valid_points) > paf_success_ratio * static_cast<float>(paf_num_samples) &&
      paf_score > 0.0f) {
    return paf_score;
  }
  return 0.0f;
}

// ---------------------------------------------------------------------------
// Group keypoints into people using greedy tree forward-propagation
// ---------------------------------------------------------------------------
std::vector<Person> group_keypoints(const std::vector<Keypoint>& keypoints, const TensorHWC& paf,
                                    float paf_score_threshold, float paf_success_ratio,
                                    int paf_num_samples, int min_valid_joints,
                                    float min_avg_person_score) {
  // Bucket keypoints by part_id
  std::array<std::vector<int>, kNumParts> kpts_by_part;
  for (size_t i = 0; i < keypoints.size(); ++i) {
    kpts_by_part[keypoints[i].part_id].push_back(static_cast<int>(i));
  }

  std::vector<Person> people;

  for (int pair_idx = 0; pair_idx < kNumPairs; ++pair_idx) {
    const auto [part_a, part_b] = kPosePairs[pair_idx];
    const auto& cands_a = kpts_by_part[part_a];
    const auto& cands_b = kpts_by_part[part_b];

    if (cands_a.empty() || cands_b.empty())
      continue;

    const auto [paf_x_idx, paf_y_idx] = kPafChannels[pair_idx];

    // Score all candidate connections
    std::vector<Connection> candidate_conns;
    for (int ai : cands_a) {
      for (int bi : cands_b) {
        float score = compute_paf_score(paf, keypoints[ai], keypoints[bi], paf_x_idx, paf_y_idx,
                                        paf_score_threshold, paf_success_ratio, paf_num_samples);
        if (score > 0.0f) {
          candidate_conns.push_back({ai, bi, score, keypoints[ai].id, keypoints[bi].id});
        }
      }
    }

    // Greedy selection: sort by score descending, pick non-conflicting
    std::sort(candidate_conns.begin(), candidate_conns.end(),
              [](const Connection& a, const Connection& b) { return a.score > b.score; });

    std::set<int> used_a, used_b;
    std::vector<Connection> valid_conns;
    for (const auto& conn : candidate_conns) {
      if (used_a.count(conn.a_id) == 0 && used_b.count(conn.b_id) == 0) {
        valid_conns.push_back(conn);
        used_a.insert(conn.a_id);
        used_b.insert(conn.b_id);
      }
    }

    // Assemble into people
    if (pair_idx == 0) {
      // First pair: create new person for each valid connection
      for (const auto& conn : valid_conns) {
        Person p;
        p.kpt_indices[part_a] = conn.a_idx;
        p.kpt_indices[part_b] = conn.b_idx;
        p.total_score = keypoints[conn.a_idx].score + keypoints[conn.b_idx].score + conn.score;
        p.valid_joints_count = 2;
        people.push_back(p);
      }
    } else if (pair_idx == 17 || pair_idx == 18) {
      // Ear connections: attach but don't create new people or add to score
      for (const auto& conn : valid_conns) {
        const int a_id = keypoints[conn.a_idx].id;
        const int b_id = keypoints[conn.b_idx].id;
        for (auto& p : people) {
          if (p.has(part_a) && keypoints[p.kpt_indices[part_a]].id == a_id && !p.has(part_b)) {
            p.kpt_indices[part_b] = conn.b_idx;
          } else if (p.has(part_b) && keypoints[p.kpt_indices[part_b]].id == b_id &&
                     !p.has(part_a)) {
            p.kpt_indices[part_a] = conn.a_idx;
          }
        }
      }
    } else {
      // Regular pairs: try to extend existing people, or create new ones
      for (const auto& conn : valid_conns) {
        const int a_id = keypoints[conn.a_idx].id;
        int matched = 0;
        for (auto& p : people) {
          if (p.has(part_a) && keypoints[p.kpt_indices[part_a]].id == a_id) {
            p.kpt_indices[part_b] = conn.b_idx;
            p.valid_joints_count += 1;
            p.total_score += keypoints[conn.b_idx].score + conn.score;
            ++matched;
          }
        }
        if (matched == 0) {
          Person p;
          p.kpt_indices[part_a] = conn.a_idx;
          p.kpt_indices[part_b] = conn.b_idx;
          p.total_score = keypoints[conn.a_idx].score + keypoints[conn.b_idx].score + conn.score;
          p.valid_joints_count = 2;
          people.push_back(p);
        }
      }
    }
  }

  // Filter: require >= 3 joints and average score >= 0.2
  std::vector<Person> final_people;
  for (const auto& p : people) {
    if (p.valid_joints_count < min_valid_joints)
      continue;
    if ((p.total_score / static_cast<float>(p.valid_joints_count)) >= min_avg_person_score) {
      final_people.push_back(p);
    }
  }
  return final_people;
}

// ---------------------------------------------------------------------------
// Scale keypoints: undo letterbox transform back to original image coordinates
// ---------------------------------------------------------------------------
void scale_keypoints(std::vector<Keypoint>& keypoints, float ratio, int pad_left, int pad_top) {
  for (auto& kp : keypoints) {
    kp.x = (kp.x - static_cast<float>(pad_left)) / ratio;
    kp.y = (kp.y - static_cast<float>(pad_top)) / ratio;
  }
}

// ---------------------------------------------------------------------------
// Draw skeleton poses onto a frame
// ---------------------------------------------------------------------------
void draw_poses(cv::Mat& frame, const std::vector<Person>& people,
                const std::vector<Keypoint>& keypoints) {
  for (const auto& person : people) {
    // Draw bones
    for (int pair_idx = 0; pair_idx < kNumPairs; ++pair_idx) {
      if (pair_idx == 17 || pair_idx == 18)
        continue; // hide ear connections
      const auto [part_a, part_b] = kPosePairs[pair_idx];
      if (!person.has(part_a) || !person.has(part_b))
        continue;

      const auto& ka = keypoints[person.kpt_indices[part_a]];
      const auto& kb = keypoints[person.kpt_indices[part_b]];
      const auto& col = kKpColors[part_a % kNumParts];
      cv::line(frame, cv::Point(static_cast<int>(ka.x), static_cast<int>(ka.y)),
               cv::Point(static_cast<int>(kb.x), static_cast<int>(kb.y)),
               cv::Scalar(col[0], col[1], col[2]), 2);
    }

    // Draw joints
    for (int part_id = 0; part_id < kNumParts; ++part_id) {
      if (!person.has(part_id))
        continue;
      const auto& kp = keypoints[person.kpt_indices[part_id]];
      const auto& col = kKpColors[part_id % kNumParts];
      const cv::Point center(static_cast<int>(kp.x), static_cast<int>(kp.y));
      cv::circle(frame, center, 4, cv::Scalar(col[0], col[1], col[2]), cv::FILLED);
      cv::circle(frame, center, 4, cv::Scalar(255, 255, 255), 1);
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  using namespace human_pose_estimator;

  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  fs::path config_path = default_config_path();
  bool enable_profile = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "OpenPose simple pose estimation pipeline.\n\n";
      std::cout << "Usage: " << argv[0] << " [--config <path>] [--profile]\n\n";
      std::cout << "Options:\n";
      std::cout << "  --config <path>  Path to YAML configuration. Default: "
                << default_config_path() << "\n";
      std::cout << "  --profile        Enable performance profiling.\n";
      return 0;
    } else if (arg == "--config") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --config requires a path\n";
        return 2;
      }
      config_path = argv[++i];
    } else if (arg == "--profile") {
      enable_profile = true;
    } else {
      std::cerr << "Error: unknown argument: " << arg << "\n";
      return 2;
    }
  }
  if (!fs::exists(config_path)) {
    std::cerr << "Error: config file not found: " << config_path << "\n";
    return 2;
  }

  AppConfig cfg;
  try {
    cfg = load_app_config(config_path);
  } catch (const std::exception& e) {
    std::cerr << "Error: failed to load config " << config_path << ": " << e.what() << "\n";
    return 2;
  }

  const std::string model_path = cfg.model.path;
  const fs::path input_dir = cfg.io.input_dir;
  const fs::path output_dir = cfg.io.output_dir;

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
  std::cout << "Found " << images.size() << " images\n";

  try {
    simaai::neat::Model::Options model_opt;
    model_opt.preprocess.kind = simaai::neat::InputKind::Image;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.preprocess.input_max_width = cfg.runtime.infer_size;
    model_opt.preprocess.input_max_height = cfg.runtime.infer_size;
    model_opt.preprocess.input_max_depth = 3;

    simaai::neat::Model model(model_path, model_opt);

    simaai::neat::Session session;
    session.add(model.session());
    std::cout << "[BUILD] Pipeline:\n" << session.describe_backend() << "\n";

    cv::Mat dummy_bgr(cfg.runtime.infer_size, cfg.runtime.infer_size, CV_8UC3, cv::Scalar(0, 0, 0));
    simaai::neat::Tensor dummy = simaai::neat::Tensor::from_cv_mat(
        dummy_bgr, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74);
    auto run = session.build(simaai::neat::TensorList{dummy}, simaai::neat::RunMode::Async);

    int processed = 0;
    double total_e2e_time = 0.0;
    double total_infer_time = 0.0;

    for (const auto& image_path : images) {
      auto e2e_start = std::chrono::high_resolution_clock::now();

      cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
        continue;
      }

      auto lb = letterbox(bgr, cfg.runtime.infer_size);

      simaai::neat::Tensor input = simaai::neat::Tensor::from_cv_mat(
          lb.img, simaai::neat::ImageSpec::PixelFormat::BGR, simaai::neat::TensorMemory::EV74);

      auto infer_start = std::chrono::high_resolution_clock::now();
      if (!run.push(simaai::neat::TensorList{input})) {
        std::cerr << "Push failed for: " << image_path.filename() << "\n";
        continue;
      }
      auto out = run.pull(cfg.runtime.timeout_ms);
      auto infer_end = std::chrono::high_resolution_clock::now();
      if (!out.has_value()) {
        std::cerr << "Pull timeout for: " << image_path.filename() << "\n";
        continue;
      }

      const auto tensors = collect_tensors(*out);
      if (tensors.size() < 2) {
        std::cerr << "Expected at least 2 tensors, got " << tensors.size() << "\n";
        continue;
      }

      TensorHWC heatmap = tensor_to_hwc_f32(tensors[0]);
      TensorHWC paf = tensor_to_hwc_f32(tensors[1]);

      // Upsample by configured factor using bicubic interpolation
      heatmap = upsample_tensor(heatmap, static_cast<float>(cfg.runtime.upsample_factor));
      paf = upsample_tensor(paf, static_cast<float>(cfg.runtime.upsample_factor));

      // Extract keypoints and group into people
      std::vector<Keypoint> raw_kpts =
          extract_keypoints(heatmap, cfg.runtime.infer_size,
                            static_cast<float>(cfg.decode.keypoint_score), cfg.decode.nms_radius);
      std::vector<Person> people = group_keypoints(
          raw_kpts, paf, static_cast<float>(cfg.decode.paf_score),
          static_cast<float>(cfg.decode.paf_success_ratio), cfg.decode.paf_samples,
          cfg.decode.min_valid_joints, static_cast<float>(cfg.decode.min_avg_person_score));

      // Scale keypoints back to original image coordinates
      scale_keypoints(raw_kpts, lb.ratio, lb.pad_left, lb.pad_top);

      // Draw and save
      draw_poses(bgr, people, raw_kpts);

      const fs::path out_path = output_dir / (image_path.stem().string() + ".png");
      if (!cv::imwrite(out_path.string(), bgr)) {
        std::cerr << "Failed to write: " << out_path << "\n";
        continue;
      }

      auto e2e_end = std::chrono::high_resolution_clock::now();
      double e2e_time = std::chrono::duration<double, std::milli>(e2e_end - e2e_start).count();
      double infer_time =
          std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
      total_e2e_time += e2e_time;
      total_infer_time += infer_time;

      ++processed;
      std::cout << "[" << processed << "/" << images.size() << "] " << image_path.filename()
                << " -> " << out_path.filename() << " (" << people.size() << " detections)";
      if (enable_profile) {
        std::cout << std::fixed << std::setprecision(2) << " | e2e=" << e2e_time
                  << "ms infer=" << infer_time << "ms";
      }
      std::cout << "\n";
    }

    run.close();
    std::cout << "Done: " << processed << " images processed\n";
    if (enable_profile && processed > 0) {
      std::cout << std::fixed << std::setprecision(2)
                << "[PROFILE] Average end-to-end: " << (total_e2e_time / processed) << "ms\n";
      std::cout << std::fixed << std::setprecision(2)
                << "[PROFILE] Average model inference: " << (total_infer_time / processed)
                << "ms\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 4;
  }
}
