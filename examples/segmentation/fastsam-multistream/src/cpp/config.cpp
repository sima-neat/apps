#include "config.h"

#include "support/runtime/config_utils.h"
#include "support/runtime/example_utils.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace app {
namespace {

using sima_examples::require;
using sima_examples::trim_copy;

std::string strip_inline_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  std::string out;
  out.reserve(line.size());
  for (const char c : line) {
    if (c == '\'' && !in_double) {
      in_single = !in_single;
    } else if (c == '"' && !in_single) {
      in_double = !in_double;
    } else if (c == '#' && !in_single && !in_double) {
      break;
    }
    out.push_back(c);
  }
  return out;
}

std::string unquote(std::string value) {
  value = trim_copy(value);
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                            (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

// Parse the source.rtsp_urls: YAML sequence (sima_examples::ScalarConfig skips lists).
std::vector<std::string> parse_rtsp_urls(const std::string& path) {
  std::ifstream in(path);
  require(in.good(), "config file not found: " + path);
  std::vector<std::string> urls;
  bool in_list = false;
  int list_indent = -1;
  std::string raw_line;
  while (std::getline(in, raw_line)) {
    if (!raw_line.empty() && raw_line.back() == '\r') {
      raw_line.pop_back();
    }
    const std::string no_comment = strip_inline_comment(raw_line);
    if (trim_copy(no_comment).empty()) {
      continue;
    }
    int indent = 0;
    while (indent < static_cast<int>(no_comment.size()) &&
           (no_comment[static_cast<std::size_t>(indent)] == ' ' ||
            no_comment[static_cast<std::size_t>(indent)] == '\t')) {
      ++indent;
    }
    const std::string line = trim_copy(no_comment);

    if (in_list && indent <= list_indent && line.rfind("- ", 0) != 0) {
      in_list = false;
    }
    if (!in_list && line == "rtsp_urls:") {
      in_list = true;
      list_indent = indent;
      continue;
    }
    if (in_list && line.rfind("- ", 0) == 0) {
      const std::string url = unquote(line.substr(2));
      if (!url.empty()) {
        urls.push_back(url);
      }
    }
  }
  return urls;
}

}  // namespace

AppConfig load_config(const std::string& path) {
  const sima_examples::ScalarConfig raw = sima_examples::ScalarConfig::load(path);
  AppConfig cfg;
  cfg.model_path = raw.string_or("model.path", "");
  cfg.clip_image_path = raw.string_or("clip.image_encoder_path", "");
  cfg.clip_text_path = raw.string_or("clip.text_encoder_path", "");
  cfg.clip_consts_path = raw.string_or("clip.text_host_consts", "");
  cfg.clip_text_features_path = raw.string_or("clip.text_features_path", "");
  cfg.clip_max_crops = raw.int_or("clip.max_crops", 0);
  cfg.clip_max_box_frac = raw.double_or("clip.max_box_frac", 0.8);
  cfg.clip_min_score = raw.double_or("clip.min_score", 0.2);
  cfg.clip_interval = raw.int_or("clip.interval", 1);
  cfg.track_iou = raw.double_or("clip.track_iou", 0.3);
  cfg.rtsp_urls = parse_rtsp_urls(path);
  cfg.latency_ms = raw.int_or("source.latency_ms", 200);
  cfg.tcp = raw.bool_or("source.tcp", true);
  cfg.infer_size = raw.int_or("runtime.infer_size", 640);
  cfg.queue_depth = raw.int_or("runtime.queue_depth", 8);
  cfg.max_fps = raw.int_or("runtime.max_fps", 0);
  cfg.warmup_frames = raw.int_or("runtime.warmup_frames", 30);
  cfg.frames = raw.int_or("runtime.frames", 0);
  cfg.score_threshold = raw.double_or("decode.score_threshold", 0.7);
  cfg.nms_iou = raw.double_or("decode.nms_iou", 0.9);
  cfg.max_detections = raw.int_or("decode.max_detections", 300);
  cfg.text = raw.string_or("prompt.text", "");
  cfg.profile = raw.bool_or("runtime.profile", false);
  cfg.profile_interval = cfg.max_fps > 0 ? cfg.max_fps : 30;
  cfg.insight_host = raw.string_or("output.insight.host", "");
  cfg.video_port_base = raw.int_or("output.insight.video_port_base", 9000);
  cfg.metadata_port_base = raw.int_or("output.insight.metadata_port_base", 9100);
  cfg.bitrate_kbps = raw.int_or("output.insight.bitrate_kbps", 1000);
  cfg.video_enabled = raw.bool_or("output.video_enabled", true);

  require(!cfg.model_path.empty(), "model.path must be set");
  require(!cfg.rtsp_urls.empty(), "source.rtsp_urls must be a non-empty list of RTSP URLs");
  require(cfg.rtsp_urls.size() <= 4, "this example supports up to four streams");
  require(cfg.warmup_frames >= 0, "runtime.warmup_frames must be >= 0");
  require(!cfg.insight_host.empty(), "output.insight.host must be set");
  require(!cfg.text.empty(), "prompt.text must be set");
  require(!cfg.clip_image_path.empty() && !cfg.clip_text_features_path.empty(),
          "clip.image_encoder_path and clip.text_features_path must be set");
  require(cfg.infer_size > 0, "runtime.infer_size must be > 0");
  require(cfg.frames >= 0, "runtime.frames must be >= 0");
  require(cfg.max_detections > 0, "decode.max_detections must be > 0");
  require(cfg.clip_max_box_frac > 0.0 && cfg.clip_max_box_frac <= 1.0,
          "clip.max_box_frac must be in (0, 1]");
  require(cfg.video_port_base > 0 && cfg.metadata_port_base > 0,
          "output.insight video_port_base and metadata_port_base must be > 0");
  return cfg;
}

}  // namespace app
