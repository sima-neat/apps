#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace multi_camera_people_tracking {

struct AppConfig {
  std::string model;
  std::vector<std::string> rtsp_urls;
  std::optional<std::string> output_dir;
  int frames = 0;
  std::string optiview_host;
  int optiview_video_port_base = 9000;
  int optiview_json_port_base = 9100;
  int fps = 0;
  int bitrate_kbps = 2500;
  int save_every = 0;
  bool profile = false;
  int person_class_id = 0;
  std::optional<double> detection_threshold;
  std::optional<double> nms_iou_threshold;
  std::optional<int> top_k;
  float tracker_iou_threshold = 0.3f;
  int tracker_max_missing = 15;
  int latency_ms = 200;
  bool tcp = false;
};

std::filesystem::path default_config_path();
AppConfig load_app_config(const std::filesystem::path& path);

} // namespace multi_camera_people_tracking
