#include "app_config.h"

#include "config.h"
#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace utils {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

SourceType parse_source_type(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "rtsp") {
    return SourceType::Rtsp;
  }
  if (lowered == "http" || lowered == "https") {
    return SourceType::Http;
  }
  if (lowered == "usb") {
    return SourceType::Usb;
  }
  throw std::runtime_error("source.type must be rtsp, http, or usb");
}

SourceCodec parse_source_codec(const std::string& value) {
  const std::string lowered = lower_copy(value);
  if (lowered == "h264" || lowered == "avc" || lowered == "h.264") {
    return SourceCodec::H264;
  }
  if (lowered == "h265" || lowered == "hevc" || lowered == "h.265") {
    return SourceCodec::H265;
  }
  if (lowered == "mjpeg" || lowered == "jpeg") {
    return SourceCodec::Mjpeg;
  }
  throw std::runtime_error("source.rtsp.codec must be h264, h265, or mjpeg");
}

void validate_config(const AppConfig& cfg) {
  if (cfg.source_type == SourceType::Usb) {
    require(!cfg.camera.empty(), "source.usb.camera must be set");
    require(cfg.source_width > 0 && cfg.source_height > 0,
            "source.usb.width and source.usb.height must be > 0");
    require(cfg.source_fps > 0, "source.usb.fps must be > 0");
    // The camera's JPEG goes to Insight as RTP/JPEG (RFC 2435): 8-pixel blocks, 2040 at most.
    require(cfg.source_width <= 2040 && cfg.source_height <= 2040 && cfg.source_width % 8 == 0 &&
                cfg.source_height % 8 == 0,
            "source.usb.width and source.usb.height must be multiples of 8 and at "
            "most 2040 (RTP/JPEG)");
  } else {
    require(!cfg.source_url.empty(), cfg.source_key + " must be set");
  }
  require(!cfg.model_path.empty(), "model.path must be set");
  require(!cfg.labels_path.empty(), "model.labels must be set");
  require(!cfg.insight_host.empty(), "output.insight.host must be set");
  const std::string src = std::string("source.") + source_type_name(cfg.source_type) + ".";
  require(cfg.latency_ms >= 0, src + "latency_ms must be >= 0");
  require(cfg.source_fps >= 0, src + "fps must be >= 0");
  require(cfg.frames >= 0, "inference.frames must be >= 0");
  require(cfg.min_score >= 0.0 && cfg.min_score <= 1.0,
          "inference.min_score must be between 0 and 1");
  require(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0, "inference.nms_iou must be between 0 and 1");
  require(cfg.max_detections > 0, "inference.max_detections must be > 0");
  require(cfg.profile_interval > 0, "runtime.profile_interval must be > 0");
  require(cfg.video_port > 0, "output.insight.video_port must be > 0");
  require(cfg.metadata_port > 0, "output.insight.metadata_port must be > 0");
  require(cfg.save_every >= 0, "output.save_every must be >= 0");
}

} // namespace

const char* source_type_name(SourceType value) {
  if (value == SourceType::Usb)
    return "usb";
  return value == SourceType::Rtsp ? "rtsp" : "http";
}

const char* source_codec_name(SourceCodec value) {
  if (value == SourceCodec::H264)
    return "h264";
  return value == SourceCodec::H265 ? "h265" : "mjpeg";
}

AppConfig load_app_config(const std::filesystem::path& config_path) {
  const auto raw = ScalarConfig::load(config_path);
  AppConfig cfg;
  const auto default_labels =
      std::filesystem::path(SIMANEAT_APPS_EXAMPLE_SOURCE_DIR).parent_path() / "common" /
      "coco_label.txt";
  cfg.model_path = raw.string_or("model.path", "");
  cfg.labels_path = raw.string_or("model.labels", default_labels.string());
  // Only the section source.type names is read; the others may hold placeholders.
  cfg.source_type = parse_source_type(raw.string_or("source.type", "rtsp"));
  const std::string src = std::string("source.") + source_type_name(cfg.source_type) + ".";
  cfg.source_key = src + (cfg.source_type == SourceType::Usb ? "camera" : "url");
  cfg.source_url = raw.string_or(src + "url", "");
  cfg.camera = raw.string_or(src + "camera", "");
  cfg.source_width = raw.int_or(src + "width", 1920);
  cfg.source_height = raw.int_or(src + "height", 1080);
  // HTTP streams and USB cameras are MJPEG; only RTSP names a codec.
  cfg.source_codec = cfg.source_type == SourceType::Rtsp
                         ? parse_source_codec(raw.string_or(src + "codec", "h264"))
                         : SourceCodec::Mjpeg;
  cfg.latency_ms = raw.int_or(src + "latency_ms", 200);
  cfg.tcp = raw.bool_or(src + "tcp", true);
  cfg.source_fps = raw.int_or(src + "fps", 0);
  cfg.ssl_strict = raw.bool_or(src + "ssl_strict", true);
  cfg.frames = raw.int_or("inference.frames", 0);
  cfg.min_score = raw.double_or("inference.min_score", 0.55);
  cfg.nms_iou = raw.double_or("inference.nms_iou", 0.60);
  cfg.max_detections = raw.int_or("inference.max_detections", 50);
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = raw.int_or("runtime.profile_interval", 100);
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port = raw.int_or("output.insight.video_port", 9000);
  cfg.metadata_port = raw.int_or("output.insight.metadata_port", 9100);
  cfg.save_dir = raw.string_or("output.save_dir", "");
  cfg.save_every = raw.int_or("output.save_every", 0);
  validate_config(cfg);
  return cfg;
}

} // namespace utils
