#include "support/runtime/config_utils.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sima_examples {
namespace {

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

bool is_nullish(const std::string& value) {
  const std::string normalized = lower_copy(trim_copy(value));
  return normalized == "null" || normalized == "~";
}

int parse_int(const std::string& value, const std::string& key) {
  std::size_t index = 0;
  const int parsed = std::stoi(value, &index);
  if (index != value.size()) {
    throw std::runtime_error(key + " must be an integer");
  }
  return parsed;
}

double parse_double(const std::string& value, const std::string& key) {
  std::size_t index = 0;
  const double parsed = std::stod(value, &index);
  if (index != value.size()) {
    throw std::runtime_error(key + " must be numeric");
  }
  return parsed;
}

} // namespace

std::string trim_copy(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t start = value.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    return {};
  }
  const std::size_t end = value.find_last_not_of(whitespace);
  return value.substr(start, end - start + 1);
}

ScalarConfig ScalarConfig::load(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open config file: " + path.string());
  }

  ScalarConfig config;
  std::vector<std::pair<int, std::string>> stack;
  int list_block_indent = -1;
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
    if (list_block_indent >= 0) {
      if (indent > list_block_indent) {
        continue;
      }
      list_block_indent = -1;
    }
    if (line.rfind("- ", 0) == 0) {
      while (!stack.empty() && indent <= stack.back().first) {
        stack.pop_back();
      }
      const std::string sequence_key = join_stack(stack);
      if (sequence_key.empty()) {
        if (config.root_kind_ == ConfigNodeKind::Mapping) {
          throw std::runtime_error("config root must be a mapping");
        }
        config.root_kind_ = ConfigNodeKind::Sequence;
      } else {
        config.node_kinds_[sequence_key] = ConfigNodeKind::Sequence;
      }
      list_block_indent = indent;
      continue;
    }

    if (indent == 0) {
      if (config.root_kind_ == ConfigNodeKind::Sequence) {
        throw std::runtime_error("config root must be a mapping");
      }
      config.root_kind_ = ConfigNodeKind::Mapping;
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

    std::string full_key = join_stack(stack);
    if (!full_key.empty()) {
      full_key += '.';
    }
    full_key += key;

    if (value.empty() || value == "{}") {
      config.node_kinds_[full_key] = ConfigNodeKind::Mapping;
      stack.emplace_back(indent, key);
      continue;
    }

    const bool quoted = value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                              (value.front() == '\'' && value.back() == '\''));
    if (!quoted && value.size() >= 2 && value.front() == '[' && value.back() == ']') {
      config.node_kinds_[full_key] = ConfigNodeKind::Sequence;
    } else if (!quoted && value.size() >= 2 && value.front() == '{' && value.back() == '}') {
      throw std::runtime_error("inline mappings are not supported for " + full_key);
    } else {
      config.node_kinds_[full_key] = ConfigNodeKind::Scalar;
    }
    config.scalars_[full_key] = unquote(value);
  }

  return config;
}

std::optional<std::string> ScalarConfig::string_value(const std::string& key) const {
  const ConfigNodeKind kind = node_kind(key);
  if (kind == ConfigNodeKind::Mapping || kind == ConfigNodeKind::Sequence) {
    throw std::runtime_error(key + " must be a scalar value");
  }
  const auto it = scalars_.find(key);
  if (it == scalars_.end() || is_nullish(it->second)) {
    return std::nullopt;
  }
  return it->second;
}

ConfigNodeKind ScalarConfig::node_kind(const std::string& key) const {
  const auto it = node_kinds_.find(key);
  return it == node_kinds_.end() ? ConfigNodeKind::Missing : it->second;
}

ConfigNodeKind ScalarConfig::root_kind() const {
  return root_kind_;
}

std::string ScalarConfig::string_or(const std::string& key,
                                    const std::string& default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? *value : default_value;
}

int ScalarConfig::int_or(const std::string& key, int default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_int(*value, key) : default_value;
}

double ScalarConfig::double_or(const std::string& key, double default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_double(*value, key) : default_value;
}

bool ScalarConfig::bool_or(const std::string& key, bool default_value) const {
  const auto value = string_value(key);
  if (!value.has_value()) {
    return default_value;
  }
  const std::string lowered = lower_copy(*value);
  if (lowered == "true") {
    return true;
  }
  if (lowered == "false") {
    return false;
  }
  throw std::runtime_error(key + " must be true or false");
}

std::map<std::string, std::string> ScalarConfig::scalars() const {
  return {scalars_.begin(), scalars_.end()};
}

std::filesystem::path default_config_path(const char* source_dir) {
  return std::filesystem::path(source_dir) / ".." / "common" / "config.yaml";
}

} // namespace sima_examples
