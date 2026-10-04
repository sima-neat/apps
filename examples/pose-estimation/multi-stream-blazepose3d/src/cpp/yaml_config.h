#pragma once

// Typed YAML scalars for this example's configuration.
//
// The Python entry point reads src/common/config.yaml with PyYAML's safe
// loader, which gives every scalar a type. This header resolves the same YAML
// 1.1 scalar types in C++, so both entry points accept and reject the same
// files: a quoted number is a string, `08` is a string, `0x10` is an integer,
// `yes` is a boolean, and a model path given as a number is rejected.
//
// It is local to this example on purpose. The shared ScalarConfig in
// support/runtime/config_utils.h is untyped and lenient, and the other
// examples and the E2E config writer depend on that.

#include "support/runtime/config_utils.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace blazepose_config {

enum class YamlScalarType { String, Null, Boolean, Integer, Number, Other };

struct YamlScalar {
  std::string value;
  YamlScalarType type = YamlScalarType::Other;
};

namespace detail {

inline std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline void append_utf8(std::string& out, uint32_t codepoint) {
  if (codepoint >= 0xd800U && codepoint <= 0xdfffU) {
    throw std::runtime_error("invalid Unicode escape in YAML string");
  }
  if (codepoint <= 0x7fU) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ffU) {
    out.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint <= 0xffffU) {
    out.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint <= 0x10ffffU) {
    out.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
    out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    out.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else {
    throw std::runtime_error("invalid Unicode escape in YAML string");
  }
}

inline uint32_t decode_hex_escape(const std::string& value, std::size_t start, std::size_t count) {
  if (start + count > value.size()) {
    throw std::runtime_error("incomplete hexadecimal escape in YAML string");
  }
  uint32_t codepoint = 0;
  for (std::size_t index = start; index < start + count; ++index) {
    const unsigned char c = static_cast<unsigned char>(value[index]);
    const int digit = std::isdigit(c) != 0 ? c - '0'
                      : std::tolower(c) >= 'a' && std::tolower(c) <= 'f'
                          ? std::tolower(c) - 'a' + 10
                          : -1;
    if (digit < 0) {
      throw std::runtime_error("invalid hexadecimal escape in YAML string");
    }
    codepoint = (codepoint << 4U) | static_cast<uint32_t>(digit);
  }
  return codepoint;
}

inline std::string unquote(std::string value) {
  value = sima_examples::trim_copy(value);
  if (value.size() < 2 || value.front() != value.back() ||
      (value.front() != '"' && value.front() != '\'')) {
    return value;
  }
  const char quote = value.front();
  const std::string body = value.substr(1, value.size() - 2);
  std::string decoded;
  decoded.reserve(body.size());
  for (std::size_t index = 0; index < body.size(); ++index) {
    const char c = body[index];
    if (quote == '\'') {
      if (c == '\'' && (index + 1 >= body.size() || body[index + 1] != '\'')) {
        throw std::runtime_error("invalid single-quoted YAML string");
      }
      decoded.push_back(c);
      if (c == '\'') {
        ++index;
      }
      continue;
    }
    if (c != '\\') {
      decoded.push_back(c);
      continue;
    }
    if (++index >= body.size()) {
      throw std::runtime_error("incomplete escape in YAML string");
    }
    const char escaped = body[index];
    switch (escaped) {
    case '0':
      decoded.push_back('\0');
      break;
    case 'a':
      decoded.push_back('\a');
      break;
    case 'b':
      decoded.push_back('\b');
      break;
    case 't':
      decoded.push_back('\t');
      break;
    case 'n':
      decoded.push_back('\n');
      break;
    case 'v':
      decoded.push_back('\v');
      break;
    case 'f':
      decoded.push_back('\f');
      break;
    case 'r':
      decoded.push_back('\r');
      break;
    case 'e':
      decoded.push_back('\x1b');
      break;
    case ' ':
      decoded.push_back(' ');
      break;
    case '"':
      decoded.push_back('"');
      break;
    case '/':
      decoded.push_back('/');
      break;
    case '\\':
      decoded.push_back('\\');
      break;
    case 'N':
      append_utf8(decoded, 0x85U);
      break;
    case '_':
      append_utf8(decoded, 0xa0U);
      break;
    case 'L':
      append_utf8(decoded, 0x2028U);
      break;
    case 'P':
      append_utf8(decoded, 0x2029U);
      break;
    case 'x':
    case 'u':
    case 'U': {
      const std::size_t digits = escaped == 'x' ? 2U : escaped == 'u' ? 4U : 8U;
      append_utf8(decoded, decode_hex_escape(body, index + 1, digits));
      index += digits;
      break;
    }
    default:
      throw std::runtime_error("unsupported escape in YAML string");
    }
  }
  return decoded;
}

inline std::string join_stack(const std::vector<std::pair<int, std::string>>& stack) {
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

inline bool is_yaml_integer(const std::string& value) {
  std::string scalar = value;
  scalar.erase(std::remove(scalar.begin(), scalar.end(), '_'), scalar.end());
  if (scalar.empty()) {
    return false;
  }
  if (scalar.front() == '+' || scalar.front() == '-') {
    scalar.erase(0, 1);
  }
  if (scalar.empty()) {
    return false;
  }

  const auto all_digits = [](const std::string& digits, int base) {
    return !digits.empty() && std::all_of(digits.begin(), digits.end(), [base](unsigned char c) {
      if (std::isdigit(c) != 0) {
        return c - '0' < base;
      }
      return base == 16 && std::tolower(c) >= 'a' && std::tolower(c) <= 'f';
    });
  };
  if (scalar.size() > 2 && scalar.rfind("0b", 0) == 0) {
    return all_digits(scalar.substr(2), 2);
  }
  if (scalar.size() > 2 && scalar.rfind("0x", 0) == 0) {
    return all_digits(scalar.substr(2), 16);
  }
  if (scalar.find(':') != std::string::npos) {
    std::istringstream segments(scalar);
    std::string segment;
    bool first = true;
    while (std::getline(segments, segment, ':')) {
      if (!all_digits(segment, 10) || (first && segment.front() == '0') ||
          (!first && (segment.size() > 2 || (segment.size() == 2 &&
                                             (segment[0] - '0') * 10 + (segment[1] - '0') > 59)))) {
        return false;
      }
      first = false;
    }
    return !scalar.empty() && scalar.back() != ':';
  }
  if (scalar.size() > 1 && scalar.front() == '0') {
    return all_digits(scalar.substr(1), 8);
  }
  return all_digits(scalar, 10);
}

inline bool is_yaml_timestamp(const std::string& value) {
  // PyYAML's safe loader uses the YAML 1.1 timestamp resolver expression.
  static const std::regex pattern(
      R"(^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$)");
  return std::regex_match(value, pattern);
}

inline bool is_yaml_number(const std::string& value) {
  static const std::regex pattern(
      R"(^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$)");
  return std::regex_match(value, pattern);
}

inline double parse_yaml_number(const std::string& value, const std::string& key) {
  if (!is_yaml_number(value)) {
    throw std::runtime_error(key + " must be numeric");
  }
  std::string scalar = value;
  scalar.erase(std::remove(scalar.begin(), scalar.end(), '_'), scalar.end());
  const std::string lowered = lower_copy(scalar);
  if (lowered == ".nan") {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (lowered == ".inf" || lowered == "+.inf") {
    return std::numeric_limits<double>::infinity();
  }
  if (lowered == "-.inf") {
    return -std::numeric_limits<double>::infinity();
  }
  const auto parse_decimal = [&](const std::string& component) {
    char* end = nullptr;
    const double parsed = std::strtod(component.c_str(), &end);
    if (end == component.c_str() || *end != '\0') {
      throw std::runtime_error(key + " must be numeric");
    }
    return parsed;
  };
  if (scalar.find(':') != std::string::npos) {
    const bool negative = scalar.front() == '-';
    if (scalar.front() == '+' || scalar.front() == '-') {
      scalar.erase(0, 1);
    }
    std::istringstream segments(scalar);
    std::string segment;
    double parsed = 0.0;
    while (std::getline(segments, segment, ':')) {
      parsed = parsed * 60.0 + parse_decimal(segment);
    }
    return negative ? -parsed : parsed;
  }
  return parse_decimal(scalar);
}

} // namespace detail

inline std::string strip_yaml_inline_comment(const std::string& line) {
  const std::size_t separator = line.find(':');
  std::size_t scalar_start = separator == std::string::npos ? 0 : separator + 1;
  while (scalar_start < line.size() &&
         std::isspace(static_cast<unsigned char>(line[scalar_start])) != 0) {
    ++scalar_start;
  }
  const char quote =
      scalar_start < line.size() && (line[scalar_start] == '\'' || line[scalar_start] == '"')
          ? line[scalar_start]
          : '\0';
  bool quoted = quote != '\0';
  for (std::size_t index = 0; index < line.size(); ++index) {
    const char c = line[index];
    if (quoted && index > scalar_start) {
      if (quote == '"' && c == '\\') {
        ++index;
        continue;
      }
      if (c == quote) {
        if (quote == '\'' && index + 1 < line.size() && line[index + 1] == '\'') {
          ++index;
          continue;
        }
        quoted = false;
        continue;
      }
    }
    if (!quoted && c == '#' &&
        (index == 0 || std::isspace(static_cast<unsigned char>(line[index - 1])) != 0)) {
      return line.substr(0, index);
    }
  }
  return line;
}

// A trimmed line that opens a block-sequence entry: "- key: value", or a
// standalone "-" whose mapping continues on the following, deeper lines.
inline bool is_sequence_entry(const std::string& line) {
  return line == "-" || line.rfind("- ", 0) == 0;
}

inline int parse_yaml_integer(const std::string& value, const std::string& key) {
  std::string scalar = value;
  scalar.erase(std::remove(scalar.begin(), scalar.end(), '_'), scalar.end());
  if (!detail::is_yaml_integer(value)) {
    throw std::runtime_error(key + " must be an integer");
  }

  const bool negative = scalar.front() == '-';
  if (scalar.front() == '+' || scalar.front() == '-') {
    scalar.erase(0, 1);
  }
  const uint64_t limit =
      negative ? static_cast<uint64_t>(INT_MAX) + 1U : static_cast<uint64_t>(INT_MAX);
  uint64_t magnitude = 0;
  const auto append_digit = [&](int digit, int base) {
    if (magnitude > (limit - static_cast<uint64_t>(digit)) / static_cast<uint64_t>(base)) {
      throw std::runtime_error(key + " is outside the supported integer range");
    }
    magnitude = magnitude * static_cast<uint64_t>(base) + static_cast<uint64_t>(digit);
  };

  if (scalar.find(':') != std::string::npos) {
    std::istringstream segments(scalar);
    std::string segment;
    while (std::getline(segments, segment, ':')) {
      uint64_t part = 0;
      for (const char c : segment) {
        const uint64_t digit = static_cast<uint64_t>(c - '0');
        if (part > (limit - digit) / 10U) {
          throw std::runtime_error(key + " is outside the supported integer range");
        }
        part = part * 10U + digit;
      }
      if (magnitude > (limit - part) / 60U) {
        throw std::runtime_error(key + " is outside the supported integer range");
      }
      magnitude = magnitude * 60U + part;
    }
  } else {
    int base = 10;
    std::size_t start = 0;
    if (scalar.size() > 2 && scalar.rfind("0b", 0) == 0) {
      base = 2;
      start = 2;
    } else if (scalar.size() > 2 && scalar.rfind("0x", 0) == 0) {
      base = 16;
      start = 2;
    } else if (scalar.size() > 1 && scalar.front() == '0') {
      base = 8;
      start = 1;
    }
    for (std::size_t index = start; index < scalar.size(); ++index) {
      const unsigned char c = static_cast<unsigned char>(scalar[index]);
      const int digit = std::isdigit(c) != 0 ? c - '0' : std::tolower(c) - 'a' + 10;
      append_digit(digit, base);
    }
  }

  if (negative && magnitude == static_cast<uint64_t>(INT_MAX) + 1U) {
    return INT_MIN;
  }
  const int parsed = static_cast<int>(magnitude);
  return negative ? -parsed : parsed;
}

inline YamlScalar parse_yaml_scalar(const std::string& value) {
  const std::string scalar = sima_examples::trim_copy(value);
  const bool quoted = scalar.size() >= 2 && ((scalar.front() == '\'' && scalar.back() == '\'') ||
                                             (scalar.front() == '"' && scalar.back() == '"'));
  if (quoted) {
    return {detail::unquote(scalar), YamlScalarType::String};
  }

  const std::string lowered = detail::lower_copy(scalar);
  if (lowered.empty() || lowered == "~" || lowered == "null") {
    return {scalar, YamlScalarType::Null};
  }
  if (lowered == "true" || lowered == "false" || lowered == "yes" || lowered == "no" ||
      lowered == "on" || lowered == "off") {
    return {scalar, YamlScalarType::Boolean};
  }
  if (scalar.front() == '[' || scalar.front() == '{' || scalar.front() == '&' ||
      scalar.front() == '*' || scalar.front() == '!' || scalar.front() == '\'' ||
      scalar.front() == '"') {
    return {scalar, YamlScalarType::Other};
  }

  if (detail::is_yaml_integer(scalar)) {
    return {scalar, YamlScalarType::Integer};
  }
  if (detail::is_yaml_number(scalar)) {
    return {scalar, YamlScalarType::Number};
  }
  return {scalar,
          detail::is_yaml_timestamp(scalar) ? YamlScalarType::Other : YamlScalarType::String};
}

class TypedConfig {
public:
  // Only block-style YAML is supported, matching the Python entry point. Empty
  // `{}` and `[]` stay allowed because PyYAML writes empty sections that way.
  static void reject_flow_collection(const std::string& line) {
    std::string content = line;
    if (is_sequence_entry(content)) {
      content = sima_examples::trim_copy(content.substr(1));
    }
    const std::size_t colon = content.find(':');
    const std::string value =
        colon == std::string::npos ? content : sima_examples::trim_copy(content.substr(colon + 1));
    for (const std::string& part : {content, value}) {
      if (!part.empty() && (part.front() == '[' || part.front() == '{') && part != "{}" &&
          part != "[]") {
        throw std::runtime_error(
            "flow-style YAML collections are not supported; use block style: " + line);
      }
    }
  }

  static TypedConfig load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
      throw std::runtime_error("failed to open config file: " + path.string());
    }

    TypedConfig config;
    std::vector<std::pair<int, std::string>> stack;
    int list_block_indent = -1;
    std::string raw_line;
    while (std::getline(input, raw_line)) {
      const std::string without_comment = strip_yaml_inline_comment(raw_line);
      if (sima_examples::trim_copy(without_comment).empty()) {
        continue;
      }

      int indent = 0;
      while (indent < static_cast<int>(without_comment.size()) &&
             (without_comment[static_cast<std::size_t>(indent)] == ' ' ||
              without_comment[static_cast<std::size_t>(indent)] == '\t')) {
        ++indent;
      }

      const std::string line = sima_examples::trim_copy(without_comment);
      reject_flow_collection(line);
      if (list_block_indent >= 0) {
        if (indent > list_block_indent) {
          continue;
        }
        list_block_indent = -1;
      }
      if (is_sequence_entry(line)) {
        list_block_indent = indent;
        continue;
      }

      const std::size_t colon = line.find(':');
      if (colon == std::string::npos) {
        throw std::runtime_error("invalid config line: " + line);
      }

      const std::string key = sima_examples::trim_copy(line.substr(0, colon));
      std::string value = sima_examples::trim_copy(line.substr(colon + 1));
      while (!stack.empty() && indent <= stack.back().first) {
        stack.pop_back();
      }

      if (value.empty() || value == "{}") {
        stack.emplace_back(indent, key);
        continue;
      }

      std::string full_key = detail::join_stack(stack);
      if (!full_key.empty()) {
        full_key += '.';
      }
      full_key += key;
      config.scalars_[full_key] = parse_yaml_scalar(value);
    }

    return config;
  }

  [[nodiscard]] std::optional<std::string> string_value(const std::string& key) const {
    const auto it = scalars_.find(key);
    if (it == scalars_.end() || it->second.type == YamlScalarType::Null) {
      return std::nullopt;
    }
    if (it->second.type != YamlScalarType::String) {
      throw std::runtime_error(key + " must be a string");
    }
    return it->second.value;
  }

  [[nodiscard]] std::string string_or(const std::string& key,
                                      const std::string& default_value) const {
    const auto value = string_value(key);
    return value.has_value() ? *value : default_value;
  }

  [[nodiscard]] int int_or(const std::string& key, int default_value) const {
    const auto it = scalars_.find(key);
    if (it == scalars_.end()) {
      return default_value;
    }
    // An explicit null is a present value of the wrong type, as in Python.
    if (it->second.type != YamlScalarType::Integer) {
      throw std::runtime_error(key + " must be an integer");
    }
    return parse_yaml_integer(it->second.value, key);
  }

  [[nodiscard]] double double_or(const std::string& key, double default_value) const {
    const auto it = scalars_.find(key);
    if (it == scalars_.end()) {
      return default_value;
    }
    // An explicit null is a present value of the wrong type, as in Python.
    if (it->second.type != YamlScalarType::Integer && it->second.type != YamlScalarType::Number) {
      throw std::runtime_error(key + " must be numeric");
    }
    return it->second.type == YamlScalarType::Integer
               ? static_cast<double>(parse_yaml_integer(it->second.value, key))
               : detail::parse_yaml_number(it->second.value, key);
  }

  [[nodiscard]] bool bool_or(const std::string& key, bool default_value) const {
    const auto it = scalars_.find(key);
    if (it == scalars_.end()) {
      return default_value;
    }
    // An explicit null is a present value of the wrong type, as in Python.
    if (it->second.type != YamlScalarType::Boolean) {
      throw std::runtime_error(key + " must be true or false");
    }
    const std::string lowered = detail::lower_copy(it->second.value);
    return lowered == "true" || lowered == "yes" || lowered == "on";
  }

private:
  std::unordered_map<std::string, YamlScalar> scalars_;
};

} // namespace blazepose_config
