/**
 * @example yolo26-pose-estimator.cpp
 * Minimal YOLO26 pose-estimation pipeline for every image in a folder.
 *
 * Usage: yolo26-pose-estimator [--config <path>]
 */
#include "neat.h"
#include "support/runtime/config_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kDefaultMinScore = 0.55f;
constexpr float kDefaultNmsIou = 0.60f;
constexpr float kKeypointScore = 0.50f;
constexpr int kMaxDet = 50;
constexpr int kTimeoutMs = 20000;
constexpr int kPoseKeypoints = 17;
constexpr int kPoseColumns = 3;

constexpr std::array<std::pair<int, int>, 16> kSkeleton = {{
    {5, 7},
    {7, 9},
    {6, 8},
    {8, 10},
    {5, 6},
    {5, 11},
    {6, 12},
    {11, 12},
    {11, 13},
    {13, 15},
    {12, 14},
    {14, 16},
    {0, 1},
    {0, 2},
    {1, 3},
    {2, 4},
}};

const std::array<cv::Scalar, 17> kKeypointColors = {{
    cv::Scalar(255, 128, 0),
    cv::Scalar(255, 153, 51),
    cv::Scalar(255, 178, 102),
    cv::Scalar(230, 230, 0),
    cv::Scalar(255, 153, 255),
    cv::Scalar(153, 204, 255),
    cv::Scalar(255, 102, 255),
    cv::Scalar(255, 51, 255),
    cv::Scalar(102, 178, 255),
    cv::Scalar(51, 153, 255),
    cv::Scalar(255, 153, 153),
    cv::Scalar(255, 102, 102),
    cv::Scalar(255, 51, 51),
    cv::Scalar(153, 255, 153),
    cv::Scalar(102, 255, 102),
    cv::Scalar(51, 255, 51),
    cv::Scalar(0, 255, 0),
}};

struct Config {
  std::string model_path;
  std::string input_dir;
  std::string output_dir;
  float min_score = kDefaultMinScore;
  float nms_iou = kDefaultNmsIou;
  int max_detections = kMaxDet;
  int timeout_ms = kTimeoutMs;
  bool profile = false;
  bool overlay = true;
  bool debug_pose = false;
  int num_runs = 1;
};

struct Keypoint {
  float x = 0.0f;
  float y = 0.0f;
  float score = 0.0f;
};

struct PoseDetection {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float score = 0.0f;
  int class_id = 0;
  std::array<Keypoint, kPoseKeypoints> keypoints;
};

Config load_config(const fs::path& path) {
  const auto raw = sima_examples::ScalarConfig::load(path);
  Config cfg;
  cfg.model_path = raw.string_or("model.path", "assets/models/yolo26m-pose-bf16-b1.tar.gz");
  cfg.input_dir = raw.string_or("io.input_dir", "assets/test_images");
  cfg.output_dir = raw.string_or("io.output_dir", "sandbox/yolo26-pose-estimator");
  cfg.min_score = static_cast<float>(raw.double_or("decode.score_threshold", kDefaultMinScore));
  cfg.nms_iou = static_cast<float>(raw.double_or("decode.nms_iou", kDefaultNmsIou));
  cfg.max_detections = raw.int_or("decode.max_detections", kMaxDet);
  cfg.timeout_ms = raw.int_or("runtime.timeout_ms", kTimeoutMs);
  cfg.num_runs = raw.int_or("runtime.num_runs", 1);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.overlay = raw.bool_or("output.overlay", true);
  cfg.debug_pose = raw.bool_or("debug.pose", false);
  if (cfg.min_score < 0.0f || cfg.min_score > 1.0f) {
    throw std::runtime_error("decode.score_threshold must be in [0.0, 1.0]");
  }
  if (cfg.nms_iou < 0.0f || cfg.nms_iou > 1.0f) {
    throw std::runtime_error("decode.nms_iou must be in [0.0, 1.0]");
  }
  if (cfg.max_detections < 1) {
    throw std::runtime_error("decode.max_detections must be >= 1");
  }
  if (cfg.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  if (cfg.num_runs < 1) {
    throw std::runtime_error("runtime.num_runs must be >= 1");
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

bool is_image(const fs::path& p) {
  std::string ext = p.extension().string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp");
}

int clear_output_images(const fs::path& output_dir, const fs::path& input_dir) {
  if (fs::weakly_canonical(output_dir) == fs::weakly_canonical(input_dir)) {
    std::cerr << "Skipping output cleanup because output_dir matches input_dir: " << output_dir
              << "\n";
    return 0;
  }

  int removed = 0;
  for (const auto& entry : fs::directory_iterator(output_dir)) {
    if (entry.is_regular_file() && is_image(entry.path())) {
      fs::remove(entry.path());
      ++removed;
    }
  }
  return removed;
}

std::vector<float> tensor_to_floats(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::Float32) {
    throw std::runtime_error("expected Float32 tensor");
  }
  std::size_t elements = 1;
  for (const int64_t dim : tensor.shape) {
    if (dim == 0) {
      return {};
    }
    elements *= static_cast<std::size_t>(dim);
  }
  const std::vector<std::uint8_t> bytes = tensor.copy_dense_bytes_tight();
  if (bytes.size() % sizeof(float) != 0) {
    throw std::runtime_error("float tensor byte size is not aligned");
  }
  if (bytes.size() != elements * sizeof(float)) {
    throw std::runtime_error("float tensor byte size does not match tensor shape");
  }
  std::vector<float> values(elements);
  if (!values.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

std::vector<PoseDetection> decode_poses(const simaai::neat::TensorList& outputs, int image_width,
                                        int image_height, int max_detections) {
  if (outputs.empty()) {
    throw std::runtime_error("model returned no pose tensors");
  }

  const auto decoded =
      simaai::neat::decode_pose(outputs, image_width, image_height, max_detections, false);
  std::vector<PoseDetection> poses;
  for (const auto& item : decoded) {
    const auto boxes = tensor_to_floats(item.boxes);
    const auto keypoints = tensor_to_floats(item.keypoints);
    const size_t box_count = boxes.size() / 6U;
    const size_t keypoint_count = keypoints.size() / (kPoseKeypoints * kPoseColumns);
    const size_t count = std::min(box_count, keypoint_count);
    for (size_t i = 0; i < count; ++i) {
      const float* box = boxes.data() + i * 6U;
      if (box[2] <= box[0] || box[3] <= box[1]) {
        continue;
      }

      PoseDetection pose;
      pose.x1 = box[0];
      pose.y1 = box[1];
      pose.x2 = box[2];
      pose.y2 = box[3];
      pose.score = box[4];
      pose.class_id = static_cast<int>(box[5]);

      const float* point_base = keypoints.data() + i * kPoseKeypoints * kPoseColumns;
      for (int k = 0; k < kPoseKeypoints; ++k) {
        const float* point = point_base + k * kPoseColumns;
        pose.keypoints[static_cast<size_t>(k)] = Keypoint{point[0], point[1], point[2]};
      }
      poses.push_back(pose);
      if (static_cast<int>(poses.size()) >= max_detections) {
        return poses;
      }
    }
  }
  return poses;
}

bool valid_keypoint(const Keypoint& point, const PoseDetection& pose, int width, int height) {
  const float box_w = pose.x2 - pose.x1;
  const float box_h = pose.y2 - pose.y1;
  const float margin = std::max(8.0f, 0.10f * std::max(box_w, box_h));
  return point.score >= kKeypointScore && point.x >= 0.0f && point.y >= 0.0f &&
         point.x < static_cast<float>(width) && point.y < static_cast<float>(height) &&
         point.x >= pose.x1 - margin && point.x <= pose.x2 + margin &&
         point.y >= pose.y1 - margin && point.y <= pose.y2 + margin;
}

void draw_pose(cv::Mat& frame, const PoseDetection& pose) {
  const int x1 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(pose.x1))));
  const int y1 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(pose.y1))));
  const int x2 = std::max(0, std::min(frame.cols - 1, static_cast<int>(std::round(pose.x2))));
  const int y2 = std::max(0, std::min(frame.rows - 1, static_cast<int>(std::round(pose.y2))));
  if (x2 > x1 && y2 > y1) {
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    const std::string text = "person " + cv::format("%.2f", pose.score);
    cv::putText(frame, text, cv::Point(x1, std::max(0, y1 - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  }

  for (const auto& [start, end] : kSkeleton) {
    const Keypoint& p0 = pose.keypoints[static_cast<size_t>(start)];
    const Keypoint& p1 = pose.keypoints[static_cast<size_t>(end)];
    if (!valid_keypoint(p0, pose, frame.cols, frame.rows) ||
        !valid_keypoint(p1, pose, frame.cols, frame.rows)) {
      continue;
    }
    cv::line(frame,
             cv::Point(static_cast<int>(std::round(p0.x)), static_cast<int>(std::round(p0.y))),
             cv::Point(static_cast<int>(std::round(p1.x)), static_cast<int>(std::round(p1.y))),
             cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
  }

  for (size_t i = 0; i < pose.keypoints.size(); ++i) {
    const Keypoint& point = pose.keypoints[i];
    if (!valid_keypoint(point, pose, frame.cols, frame.rows)) {
      continue;
    }
    cv::circle(
        frame,
        cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y))), 3,
        kKeypointColors[i % kKeypointColors.size()], cv::FILLED, cv::LINE_AA);
  }
}

void draw_poses(cv::Mat& frame, const std::vector<PoseDetection>& poses) {
  for (const auto& pose : poses) {
    draw_pose(frame, pose);
  }
}

void log_pose_debug(const fs::path& image_path, int width, int height, const PoseDetection& pose) {
  std::cout << "[POSE_DEBUG] image=" << image_path.filename().string() << " frame=" << width << "x"
            << height << " bbox=(" << cv::format("%.1f", pose.x1) << ","
            << cv::format("%.1f", pose.y1) << "," << cv::format("%.1f", pose.x2) << ","
            << cv::format("%.1f", pose.y2) << ") score=" << cv::format("%.3f", pose.score)
            << " class_id=" << pose.class_id << "\n";
  for (size_t i = 0; i < pose.keypoints.size(); ++i) {
    const Keypoint& point = pose.keypoints[i];
    std::cout << "[POSE_DEBUG] kp[" << i << "]=(" << cv::format("%.1f", point.x) << ","
              << cv::format("%.1f", point.y) << ") score=" << cv::format("%.3f", point.score)
              << "\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  Config cfg;
  try {
    cfg = parse_config(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  const fs::path input_dir = cfg.input_dir;
  const fs::path output_dir = cfg.output_dir;

  if (!fs::is_directory(input_dir)) {
    std::cerr << "Input directory does not exist: " << input_dir << "\n";
    return 2;
  }
  fs::create_directories(output_dir);
  const int removed_outputs = clear_output_images(output_dir, input_dir);
  if (removed_outputs > 0) {
    std::cout << "Cleared " << removed_outputs << " stale output images\n";
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
    model_opt.preprocess.enable = simaai::neat::AutoFlag::On;
    model_opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
    model_opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
    model_opt.decode_type = simaai::neat::BoxDecodeType::YoloV26Pose;
    model_opt.score_threshold = cfg.min_score;
    model_opt.nms_iou_threshold = cfg.nms_iou;
    model_opt.top_k = cfg.max_detections;

    simaai::neat::Model model(cfg.model_path, model_opt);

    cv::Mat seed_bgr = cv::imread(images.front().string(), cv::IMREAD_COLOR);
    if (seed_bgr.empty()) {
      throw std::runtime_error("failed to read build seed image: " + images.front().string());
    }
    std::cout << "[BUILD] Building pipeline...\n";
    auto runner = model.build(std::vector<cv::Mat>{seed_bgr}, simaai::neat::Model::RouteOptions{});
    std::cout << "[BUILD] Pipeline built\n";

    runner.run(std::vector<cv::Mat>{seed_bgr}, cfg.timeout_ms);
    std::cout << "[WARMUP] done\n";

    const int total_images = static_cast<int>(images.size()) * cfg.num_runs;
    if (cfg.num_runs > 1) {
      std::cout << "Looping " << cfg.num_runs << "x over " << images.size() << " images ("
                << total_images << " total)\n";
    }

    const auto pipeline_start = std::chrono::steady_clock::now();
    int processed = 0;
    bool printed_pose_debug = false;

    for (int run_idx = 0; run_idx < cfg.num_runs; ++run_idx) {
      for (const auto& image_path : images) {
        const auto img_start = std::chrono::steady_clock::now();

        cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (bgr.empty()) {
          std::cerr << "Skipping unreadable: " << image_path.filename() << "\n";
          continue;
        }

        const auto infer_start = std::chrono::steady_clock::now();
        simaai::neat::TensorList out = runner.run(std::vector<cv::Mat>{bgr}, cfg.timeout_ms);
        const auto infer_end = std::chrono::steady_clock::now();

        const std::vector<PoseDetection> poses =
            decode_poses(out, bgr.cols, bgr.rows, cfg.max_detections);
        const auto decode_end = std::chrono::steady_clock::now();

        if (cfg.debug_pose && !printed_pose_debug && !poses.empty()) {
          log_pose_debug(image_path, bgr.cols, bgr.rows, poses.front());
          printed_pose_debug = true;
        }

        if (cfg.overlay) {
          draw_poses(bgr, poses);
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
                    << " (" << poses.size() << " poses)\n";
        } else {
          std::cout << "[" << processed << "/" << total_images << "] " << image_path.filename()
                    << " (" << poses.size() << " poses)\n";
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

    runner.close();

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
