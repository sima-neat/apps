#pragma once

#include "model_family_api.cpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace multistream_yolox_yolov8_optiview {

enum class VideoMode {
  Clean,
  Annotated,
};

struct ModelConfig {
  std::string path;
  ModelFamily family = ModelFamily::Auto;
};

struct AppConfig {
  ModelConfig model;
  std::vector<std::string> rtsp_urls;
  bool tcp = false;
  int latency_ms = 100;
  int worker_count = 1;
  int mailbox_depth = 1;
  bool profile = false;
  int frames = 0;
  int fps = 0;
  double min_score = 0.25;
  double nms_iou = 0.45;
  int max_detections = 100;
  std::string optiview_host;
  int optiview_video_port_base = 9000;
  int optiview_json_port_base = 9100;
  double optiview_json_offset_ms = 0.0;
  VideoMode video_mode = VideoMode::Clean;
  std::optional<std::string> output_dir;
  int save_every = 0;
};

std::filesystem::path default_config_path();
std::string to_string(VideoMode mode);
AppConfig load_app_config(const std::filesystem::path& path);

} // namespace multistream_yolox_yolov8_optiview
