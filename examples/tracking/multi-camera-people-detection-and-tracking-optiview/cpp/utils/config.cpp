#include "examples/tracking/multi-camera-people-detection-and-tracking-optiview/cpp/utils/config_api.cpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

namespace multi_camera_people_tracking {
namespace {

struct RawConfig {
  std::unordered_map<std::string, std::string> scalars;
  std::vector<std::string> streams;
};

std::string trim_copy(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t start = value.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(whitespace);
  return value.substr(start, end - start + 1);
}

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string strip_inline_comment(const std::string& line) {
  bool in_single = false;
  bool in_double = false;
  std::string out;
  out.reserve(line.size());
  for (char c : line) {
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
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

bool is_nullish(const std::string& value) {
  return lower_copy(trim_copy(value)) == "null";
}

std::string join_stack(const std::vector<std::pair<int, std::string>>& stack) {
  std::ostringstream out;
  bool first = true;
  for (const auto& [indent, key] : stack) {
    static_cast<void>(indent);
    if (!first) {
      out << '.';
    }
    first = false;
    out << key;
  }
  return out.str();
}

RawConfig parse_raw_config(const fs::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file");
  }

  RawConfig raw;
  std::vector<std::pair<int, std::string>> stack;
  std::string raw_line;

  while (std::getline(input, raw_line)) {
    const std::string without_comment = strip_inline_comment(raw_line);
    if (trim_copy(without_comment).empty()) {
      continue;
    }

    int indent = 0;
    while (indent < static_cast<int>(without_comment.size()) &&
           (without_comment[static_cast<std::size_t>(indent)] == ' ' ||
            without_comment[static_cast<std::size_t>(indent)] == '\t')) {
      ++indent;
    }

    const std::string line = trim_copy(without_comment);
    if (line.rfind("- ", 0) == 0) {
      if (join_stack(stack) == "streams") {
        const std::string value = unquote(line.substr(2));
        if (value.empty()) {
          throw std::runtime_error("streams entries must be non-empty strings");
        }
        raw.streams.push_back(value);
      }
      continue;
    }

    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("invalid config line: " + line);
    }

    const std::string key = trim_copy(line.substr(0, colon));
    std::string value = trim_copy(line.substr(colon + 1));
    while (!stack.empty() && indent <= stack.back().first) {
      stack.pop_back();
    }

    if (value.empty() || value == "{}") {
      stack.emplace_back(indent, key);
      continue;
    }

    value = unquote(value);
    std::string full_key = join_stack(stack);
    if (!full_key.empty()) {
      full_key += '.';
    }
    full_key += key;
    raw.scalars[full_key] = value;
  }

  return raw;
}

std::optional<std::string> lookup_scalar(const RawConfig& raw, const std::string& key) {
  const auto it = raw.scalars.find(key);
  if (it == raw.scalars.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::string require_non_empty_string(const RawConfig& raw, const std::string& key,
                                     const std::string& error_name) {
  const auto value = lookup_scalar(raw, key);
  if (!value.has_value() || trim_copy(*value).empty() || is_nullish(*value)) {
    throw std::runtime_error(error_name + " must be a non-empty string");
  }
  return *value;
}

int parse_int(const std::string& value, const std::string& error_name) {
  std::size_t index = 0;
  const int parsed = std::stoi(value, &index);
  if (index != value.size()) {
    throw std::runtime_error(error_name + " must be an integer");
  }
  return parsed;
}

double parse_double(const std::string& value, const std::string& error_name) {
  std::size_t index = 0;
  const double parsed = std::stod(value, &index);
  if (index != value.size()) {
    throw std::runtime_error(error_name + " must be numeric or null");
  }
  return parsed;
}

int optional_int(const RawConfig& raw, const std::string& key, int default_value,
                 const std::string& error_name) {
  const auto value = lookup_scalar(raw, key);
  if (!value.has_value() || is_nullish(*value)) {
    return default_value;
  }
  return parse_int(*value, error_name);
}

bool optional_bool(const RawConfig& raw, const std::string& key, bool default_value,
                   const std::string& error_name) {
  const auto value = lookup_scalar(raw, key);
  if (!value.has_value() || is_nullish(*value)) {
    return default_value;
  }
  const std::string lowered = lower_copy(*value);
  if (lowered == "true") {
    return true;
  }
  if (lowered == "false") {
    return false;
  }
  throw std::runtime_error(error_name + " must be true or false");
}

std::optional<double> optional_double_or_none(const RawConfig& raw, const std::string& key,
                                              const std::string& error_name) {
  const auto value = lookup_scalar(raw, key);
  if (!value.has_value() || is_nullish(*value)) {
    return std::nullopt;
  }
  return parse_double(*value, error_name);
}

std::optional<int> optional_positive_int_or_none(const RawConfig& raw, const std::string& key,
                                                 const std::string& error_name) {
  const auto value = lookup_scalar(raw, key);
  if (!value.has_value() || is_nullish(*value)) {
    return std::nullopt;
  }
  const int parsed = parse_int(*value, error_name);
  return parsed;
}

} // namespace

std::filesystem::path default_config_path() {
#ifdef MULTI_CAMERA_PEOPLE_TRACKING_SOURCE_DIR
  return fs::path(MULTI_CAMERA_PEOPLE_TRACKING_SOURCE_DIR).parent_path() / "common" / "config.yaml";
#else
  return fs::path(
      "examples/tracking/multi-camera-people-detection-and-tracking-optiview/common/config.yaml");
#endif
}

AppConfig load_app_config(const std::filesystem::path& path) {
  const RawConfig raw = parse_raw_config(path);

  if (raw.streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }

  AppConfig cfg;
  cfg.model = require_non_empty_string(raw, "model", "model");
  cfg.rtsp_urls = raw.streams;
  cfg.optiview_host =
      require_non_empty_string(raw, "output.optiview.host", "output.optiview.host");
  cfg.optiview_video_port_base =
      optional_int(raw, "output.optiview.video_port_base", 9000, "optiview.video_port_base");
  cfg.optiview_json_port_base =
      optional_int(raw, "output.optiview.json_port_base", 9100, "optiview.json_port_base");
  cfg.frames = optional_int(raw, "inference.frames", 0, "inference.frames");
  cfg.fps = optional_int(raw, "inference.fps", 0, "inference.fps");
  cfg.bitrate_kbps =
      optional_int(raw, "inference.bitrate_kbps", 2500, "inference.bitrate_kbps");
  cfg.save_every = optional_int(raw, "output.save_every", 0, "output.save_every");
  cfg.profile = optional_bool(raw, "inference.profile", false, "inference.profile");
  cfg.person_class_id =
      optional_int(raw, "inference.person_class_id", 0, "inference.person_class_id");
  cfg.detection_threshold = optional_double_or_none(
      raw, "inference.detection_threshold", "inference.detection_threshold");
  cfg.nms_iou_threshold = optional_double_or_none(
      raw, "inference.nms_iou_threshold", "inference.nms_iou_threshold");
  cfg.top_k = optional_positive_int_or_none(raw, "inference.top_k", "inference.top_k");
  cfg.tracker_iou_threshold = static_cast<float>(optional_double_or_none(
                                                     raw, "tracking.iou_threshold",
                                                     "tracking.iou_threshold")
                                                     .value_or(0.3));
  cfg.tracker_max_missing = optional_int(
      raw, "tracking.max_missing_frames", 15, "tracking.max_missing_frames");
  cfg.latency_ms = optional_int(raw, "input.latency_ms", 200, "input.latency_ms");
  cfg.tcp = optional_bool(raw, "input.tcp", false, "input.tcp");

  const auto output_dir = lookup_scalar(raw, "output.debug_dir");
  if (output_dir.has_value() && !is_nullish(*output_dir)) {
    cfg.output_dir = *output_dir;
  }

  if (cfg.optiview_video_port_base <= 0) {
    throw std::runtime_error("optiview.video_port_base must be > 0");
  }
  if (cfg.optiview_json_port_base <= 0) {
    throw std::runtime_error("optiview.json_port_base must be > 0");
  }
  if (cfg.frames < 0) {
    throw std::runtime_error("inference.frames must be >= 0");
  }
  if (cfg.fps < 0) {
    throw std::runtime_error("inference.fps must be >= 0");
  }
  if (cfg.bitrate_kbps <= 0) {
    throw std::runtime_error("inference.bitrate_kbps must be > 0");
  }
  if (cfg.save_every < 0) {
    throw std::runtime_error("output.save_every must be >= 0");
  }
  if (cfg.latency_ms < 0) {
    throw std::runtime_error("input.latency_ms must be >= 0");
  }
  if (cfg.tracker_max_missing < 0) {
    throw std::runtime_error("tracking.max_missing_frames must be >= 0");
  }
  if (cfg.tracker_iou_threshold < 0.0f || cfg.tracker_iou_threshold > 1.0f) {
    throw std::runtime_error("tracking.iou_threshold must be between 0 and 1");
  }
  if (cfg.detection_threshold.has_value() &&
      (*cfg.detection_threshold < 0.0 || *cfg.detection_threshold > 1.0)) {
    throw std::runtime_error("inference.detection_threshold must be between 0 and 1");
  }
  if (cfg.nms_iou_threshold.has_value() &&
      (*cfg.nms_iou_threshold < 0.0 || *cfg.nms_iou_threshold > 1.0)) {
    throw std::runtime_error("inference.nms_iou_threshold must be between 0 and 1");
  }
  if (cfg.top_k.has_value() && *cfg.top_k <= 0) {
    throw std::runtime_error("inference.top_k must be > 0");
  }

  return cfg;
}

} // namespace multi_camera_people_tracking
