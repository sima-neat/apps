#include "config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace human_pose_estimator {
namespace {

struct RawConfig {
  std::unordered_map<std::string, std::string> scalars;
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
      throw std::runtime_error("lists are not supported in this config");
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

} // namespace

std::filesystem::path default_config_path() {
#ifdef SIMPLE_POSE_ESTIMATION_OVERLAY_PIPELINE_SOURCE_DIR
  return fs::path(SIMPLE_POSE_ESTIMATION_OVERLAY_PIPELINE_SOURCE_DIR) / "common" / "config.yaml";
#else
  return fs::path("examples/pose-estimation/human-pose-estimator/common/config.yaml");
#endif
}

AppConfig load_app_config(const std::filesystem::path& path) {
  const RawConfig raw = parse_raw_config(path);

  AppConfig cfg;
  cfg.model.path = require_non_empty_string(raw, "model.path", "model.path");
  cfg.io.input_dir = require_non_empty_string(raw, "io.input_dir", "io.input_dir");
  cfg.io.output_dir = require_non_empty_string(raw, "io.output_dir", "io.output_dir");
  cfg.runtime.infer_size = optional_int(raw, "runtime.infer_size", 640, "runtime.infer_size");
  cfg.runtime.timeout_ms = optional_int(raw, "runtime.timeout_ms", 20000, "runtime.timeout_ms");
  cfg.runtime.upsample_factor =
      optional_double(raw, "runtime.upsample_factor", 4.0, "runtime.upsample_factor");
  cfg.decode.keypoint_score =
      optional_double(raw, "decode.keypoint_score", 0.1, "decode.keypoint_score");
  cfg.decode.nms_radius = optional_int(raw, "decode.nms_radius", 6, "decode.nms_radius");
  cfg.decode.paf_score = optional_double(raw, "decode.paf_score", 0.05, "decode.paf_score");
  cfg.decode.paf_success_ratio =
      optional_double(raw, "decode.paf_success_ratio", 0.8, "decode.paf_success_ratio");
  cfg.decode.paf_samples = optional_int(raw, "decode.paf_samples", 10, "decode.paf_samples");
  cfg.decode.min_valid_joints =
      optional_int(raw, "decode.min_valid_joints", 3, "decode.min_valid_joints");
  cfg.decode.min_avg_person_score =
      optional_double(raw, "decode.min_avg_person_score", 0.2, "decode.min_avg_person_score");

  if (cfg.runtime.infer_size <= 0) {
    throw std::runtime_error("runtime.infer_size must be > 0");
  }
  if (cfg.runtime.timeout_ms <= 0) {
    throw std::runtime_error("runtime.timeout_ms must be > 0");
  }
  if (cfg.runtime.upsample_factor <= 0.0) {
    throw std::runtime_error("runtime.upsample_factor must be > 0");
  }
  if (cfg.decode.keypoint_score < 0.0 || cfg.decode.keypoint_score > 1.0) {
    throw std::runtime_error("decode.keypoint_score must be between 0 and 1");
  }
  if (cfg.decode.nms_radius < 0) {
    throw std::runtime_error("decode.nms_radius must be >= 0");
  }
  if (cfg.decode.paf_score < 0.0 || cfg.decode.paf_score > 1.0) {
    throw std::runtime_error("decode.paf_score must be between 0 and 1");
  }
  if (cfg.decode.paf_success_ratio < 0.0 || cfg.decode.paf_success_ratio > 1.0) {
    throw std::runtime_error("decode.paf_success_ratio must be between 0 and 1");
  }
  if (cfg.decode.paf_samples < 2) {
    throw std::runtime_error("decode.paf_samples must be >= 2");
  }
  if (cfg.decode.min_valid_joints <= 0) {
    throw std::runtime_error("decode.min_valid_joints must be > 0");
  }
  if (cfg.decode.min_avg_person_score < 0.0) {
    throw std::runtime_error("decode.min_avg_person_score must be >= 0");
  }

  return cfg;
}

} // namespace human_pose_estimator
