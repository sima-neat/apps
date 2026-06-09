#include "config_api.cpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

namespace multi_stream_object_detector {
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
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
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
  if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
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
    throw std::runtime_error(error_name + " must be numeric");
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

double optional_double(const RawConfig& raw, const std::string& key, double default_value,
                       const std::string& error_name) {
  const auto value = lookup_scalar(raw, key);
  if (!value.has_value() || is_nullish(*value)) {
    return default_value;
  }
  return parse_double(*value, error_name);
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

VideoMode parse_video_mode(const std::string& value) {
  const std::string lowered = lower_copy(trim_copy(value));
  if (lowered == "clean") {
    return VideoMode::Clean;
  }
  if (lowered == "annotated") {
    return VideoMode::Annotated;
  }
  throw std::runtime_error("output.video_mode must be one of [clean, annotated]");
}

} // namespace

std::filesystem::path default_config_path() {
#ifdef MULTISTREAM_OBJECT_DETECTION_INSIGHT_SOURCE_DIR
  return fs::path(MULTISTREAM_OBJECT_DETECTION_INSIGHT_SOURCE_DIR).parent_path() / "common" /
         "config.yaml";
#else
  return fs::path("examples/object-detection/multi-stream-object-detector/"
                  "common/config.yaml");
#endif
}

std::string to_string(VideoMode mode) {
  switch (mode) {
  case VideoMode::Clean:
    return "clean";
  case VideoMode::Annotated:
    return "annotated";
  }
  return "clean";
}

bool metadata_output_enabled(const AppConfig& cfg) {
  return !cfg.video_enabled || cfg.video_mode == VideoMode::Clean;
}

AppConfig load_app_config(const std::filesystem::path& path) {
  const RawConfig raw = parse_raw_config(path);

  if (raw.streams.empty()) {
    throw std::runtime_error("streams must be a non-empty list");
  }
  if (lookup_scalar(raw, "model.family").has_value()) {
    throw std::runtime_error(
        "model.family is no longer supported; this example infers YOLO26 from model.path");
  }

  AppConfig cfg;
  cfg.model.path = require_non_empty_string(raw, "model.path", "model.path");
  cfg.rtsp_urls = raw.streams;
  cfg.tcp = optional_bool(raw, "input.tcp", false, "input.tcp");
  cfg.latency_ms = optional_int(raw, "input.latency_ms", 100, "input.latency_ms");
  cfg.worker_count = optional_int(raw, "runtime.worker_count", 1, "runtime.worker_count");
  cfg.mailbox_depth = optional_int(raw, "runtime.mailbox_depth", 1, "runtime.mailbox_depth");
  cfg.profile = optional_bool(raw, "runtime.profile", false, "runtime.profile");
  cfg.frames = optional_int(raw, "inference.frames", 0, "inference.frames");
  cfg.fps = optional_int(raw, "inference.fps", 0, "inference.fps");
  cfg.min_score = optional_double(raw, "inference.min_score", 0.25, "inference.min_score");
  cfg.nms_iou = optional_double(raw, "inference.nms_iou", 0.45, "inference.nms_iou");
  cfg.max_detections =
      optional_int(raw, "inference.max_detections", 100, "inference.max_detections");
  cfg.insight_host = require_non_empty_string(raw, "output.insight.host", "output.insight.host");
  cfg.insight_video_port_base =
      optional_int(raw, "output.insight.video_port_base", 9000, "output.insight.video_port_base");
  cfg.insight_metadata_port_base = optional_int(raw, "output.insight.metadata_port_base", 9100,
                                                "output.insight.metadata_port_base");
  cfg.insight_metadata_offset_ms = optional_double(raw, "output.insight.metadata_offset_ms", 0.0,
                                                   "output.insight.metadata_offset_ms");
  cfg.video_enabled = optional_bool(raw, "output.video_enabled", true, "output.video_enabled");
  cfg.video_mode =
      parse_video_mode(lookup_scalar(raw, "output.video_mode").value_or(std::string("clean")));
  if (const auto out_dir = lookup_scalar(raw, "output.debug_dir");
      out_dir.has_value() && !is_nullish(*out_dir)) {
    cfg.output_dir = *out_dir;
  }
  cfg.save_every = optional_int(raw, "output.save_every", 0, "output.save_every");

  if (cfg.worker_count <= 0) {
    throw std::runtime_error("runtime.worker_count must be > 0");
  }
  if (cfg.mailbox_depth <= 0) {
    throw std::runtime_error("runtime.mailbox_depth must be > 0");
  }
  if (cfg.latency_ms < 0) {
    throw std::runtime_error("input.latency_ms must be >= 0");
  }
  if (cfg.frames < 0) {
    throw std::runtime_error("inference.frames must be >= 0");
  }
  if (cfg.fps < 0) {
    throw std::runtime_error("inference.fps must be >= 0");
  }
  if (!(cfg.min_score >= 0.0 && cfg.min_score <= 1.0)) {
    throw std::runtime_error("inference.min_score must be between 0 and 1");
  }
  if (!(cfg.nms_iou >= 0.0 && cfg.nms_iou <= 1.0)) {
    throw std::runtime_error("inference.nms_iou must be between 0 and 1");
  }
  if (cfg.max_detections <= 0) {
    throw std::runtime_error("inference.max_detections must be > 0");
  }
  if (cfg.insight_video_port_base <= 0) {
    throw std::runtime_error("output.insight.video_port_base must be > 0");
  }
  if (cfg.insight_metadata_port_base <= 0) {
    throw std::runtime_error("output.insight.metadata_port_base must be > 0");
  }
  if (cfg.save_every < 0) {
    throw std::runtime_error("output.save_every must be >= 0");
  }

  return cfg;
}

} // namespace multi_stream_object_detector
