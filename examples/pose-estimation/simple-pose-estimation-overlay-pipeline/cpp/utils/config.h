#pragma once

#include <filesystem>
#include <string>

namespace simple_pose_estimation_overlay_pipeline {

struct ModelConfig {
  std::string path;
};

struct IoConfig {
  std::string input_dir;
  std::string output_dir;
};

struct RuntimeConfig {
  int infer_size = 640;
  int timeout_ms = 1000;
  double upsample_factor = 4.0;
};

struct DecodeConfig {
  double keypoint_score = 0.1;
  int nms_radius = 6;
  double paf_score = 0.05;
  double paf_success_ratio = 0.8;
  int paf_samples = 10;
  int min_valid_joints = 3;
  double min_avg_person_score = 0.2;
};

struct AppConfig {
  ModelConfig model;
  IoConfig io;
  RuntimeConfig runtime;
  DecodeConfig decode;
};

std::filesystem::path default_config_path();
AppConfig load_app_config(const std::filesystem::path& path);

}  // namespace simple_pose_estimation_overlay_pipeline
