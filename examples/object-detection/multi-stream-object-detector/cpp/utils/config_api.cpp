#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace multi_stream_object_detector {

enum class VideoMode {
  Clean,
  Annotated,
};

struct ModelConfig {
  std::string path;
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
  std::string insight_host;
  int insight_video_port_base = 9000;
  int insight_metadata_port_base = 9100;
  double insight_metadata_offset_ms = 0.0;
  bool video_enabled = true;
  VideoMode video_mode = VideoMode::Clean;
  std::optional<std::string> output_dir;
  int save_every = 0;
};

std::filesystem::path default_config_path();
std::string to_string(VideoMode mode);
bool metadata_output_enabled(const AppConfig& cfg);
AppConfig load_app_config(const std::filesystem::path& path);

} // namespace multi_stream_object_detector
