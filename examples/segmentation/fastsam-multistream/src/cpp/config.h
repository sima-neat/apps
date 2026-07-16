#pragma once

#include <string>
#include <vector>

namespace app {

struct AppConfig {
  std::string model_path;
  std::string clip_image_path;
  std::string clip_text_path;
  std::string clip_consts_path;
  std::string clip_text_features_path;
  int clip_min_area = 100;
  double clip_max_frac = 0.5;
  double clip_max_box_frac = 0.8;
  int clip_max_crops = 0;
  double clip_min_score = 0.65;
  int clip_crop_workers = 0;
  int clip_interval = 1;
  double track_iou = 0.3;
  std::vector<std::string> rtsp_urls;
  int latency_ms = 200;
  bool tcp = true;
  int infer_size = 640;
  int timeout_ms = 20000;
  int queue_depth = 8;
  int max_fps = 0;
  int warmup_frames = 30;
  double score_threshold = 0.7;
  double nms_iou = 0.9;
  int max_detections = 300;
  int frames = 0;
  std::string text;
  bool profile = false;
  int profile_interval = 30;
  std::string insight_host;
  int video_port_base = 9000;
  int metadata_port_base = 9100;
  int bitrate_kbps = 1000;
  bool video_enabled = true;
};

AppConfig load_config(const std::string& path);

}  // namespace app
