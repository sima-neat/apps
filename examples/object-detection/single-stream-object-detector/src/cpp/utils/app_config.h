#pragma once

// This example's settings: the keys it reads from config.yaml, their defaults and their rules.

#include <filesystem>
#include <string>

namespace utils {

enum class SourceType { Rtsp, Http, Usb };
enum class SourceCodec { H264, H265, Mjpeg };

struct AppConfig {
  std::string model_path;
  std::filesystem::path labels_path;
  std::string source_url;
  // The key that names the source: source.<type>.url, or source.usb.camera.
  std::string source_key = "source.rtsp.url";
  SourceType source_type = SourceType::Rtsp;
  // source.type=usb: the camera's stable ID (usb_camera_ids()) and one of its MJPEG modes.
  std::string camera;
  int source_width = 1920;
  int source_height = 1080;
  SourceCodec source_codec = SourceCodec::H264;
  int latency_ms = 200;
  bool tcp = true;
  int source_fps = 0;
  bool ssl_strict = true;
  int frames = 0;
  double min_score = 0.55;
  double nms_iou = 0.60;
  int max_detections = 50;
  bool profile = false;
  int profile_interval = 100;
  std::string insight_host = "127.0.0.1";
  int video_port = 9000;
  int metadata_port = 9100;
  std::filesystem::path save_dir;
  int save_every = 0;
};

const char* source_type_name(SourceType value);
const char* source_codec_name(SourceCodec value);

// Reads and validates config.yaml. Throws naming the key at fault.
AppConfig load_app_config(const std::filesystem::path& config_path);

} // namespace utils
