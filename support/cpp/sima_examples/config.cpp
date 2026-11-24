#include "sima_examples/config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace sima_examples {
namespace {

std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool is_nullish(const std::string& value) {
  return lower_copy(trim_copy(value)) == "null";
}

// std::stoi and std::stod name no key in their errors, so each failure names the key instead.
int parse_int(const std::string& value, const std::string& key) {
  std::size_t index = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &index);
  } catch (const std::logic_error&) {
    index = 0;
  }
  if (index == 0 || index != value.size()) {
    throw std::runtime_error(key + " must be an integer, got '" + value + "'");
  }
  return parsed;
}

double parse_double(const std::string& value, const std::string& key) {
  std::size_t index = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &index);
  } catch (const std::logic_error&) {
    index = 0;
  }
  if (index == 0 || index != value.size()) {
    throw std::runtime_error(key + " must be a number, got '" + value + "'");
  }
  return parsed;
}

bool parse_bool(const std::string& value, const std::string& key) {
  const std::string lowered = lower_copy(value);
  if (lowered == "true") {
    return true;
  }
  if (lowered == "false") {
    return false;
  }
  throw std::runtime_error(key + " must be true or false");
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

Config Config::load(const std::filesystem::path& path) {
  Config config;
  config.path_ = path;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const YAML::BadFile&) {
    throw std::runtime_error("failed to open config file: " + path.string());
  } catch (const YAML::Exception& error) {
    throw std::runtime_error(path.string() + ": " + error.what());
  }
  if (!root.IsNull() && !root.IsMap()) {
    throw std::runtime_error(path.string() + " must hold a mapping");
  }
  // Store every leaf under its dotted key: a scalar as its text, null as "null", and a list of
  // scalars as its items. A list holding a null, a list, or a mapping is recorded as one
  // string_list() rejects, as the Python reader does.
  const std::function<void(const std::string&, const YAML::Node&)> add =
      [&config, &add](const std::string& key, const YAML::Node& node) {
        if (node.IsMap()) {
          for (const auto& entry : node) {
            const auto child = entry.first.as<std::string>();
            add(key.empty() ? child : key + "." + child, entry.second);
          }
        } else if (node.IsSequence()) {
          auto& items = config.lists_[key];
          for (const auto& item : node) {
            if (item.IsScalar()) {
              items.push_back(item.Scalar());
            } else {
              config.lists_with_non_strings_.insert(key);
            }
          }
          if (node.Style() == YAML::EmitterStyle::Flow) {
            config.scalars_[key] = YAML::Dump(node); // Older getters read a flow list as text.
          }
        } else {
          config.scalars_[key] = node.IsNull() ? "null" : node.Scalar();
        }
      };
  if (root.IsMap()) {
    add("", root);
  }
  return config;
}

std::optional<std::string> Config::string_value(const std::string& key) const {
  const auto it = scalars_.find(key);
  if (it == scalars_.end() || is_nullish(it->second)) {
    return std::nullopt;
  }
  return it->second;
}

std::string Config::string_or(const std::string& key, const std::string& default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? *value : default_value;
}

int Config::int_or(const std::string& key, int default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_int(*value, key) : default_value;
}

double Config::double_or(const std::string& key, double default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_double(*value, key) : default_value;
}

bool Config::bool_or(const std::string& key, bool default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_bool(*value, key) : default_value;
}

std::map<std::string, std::string> Config::scalars() const {
  return {scalars_.begin(), scalars_.end()};
}

std::string Config::require_key(const std::string& key) const {
  const auto it = scalars_.find(key);
  if (it == scalars_.end()) {
    throw std::runtime_error(path_.string() + " is missing " + key);
  }
  return it->second;
}

std::string Config::string(const std::string& key) const {
  return require_key(key);
}

int Config::integer(const std::string& key) const {
  return parse_int(require_key(key), key);
}

double Config::number(const std::string& key) const {
  return parse_double(require_key(key), key);
}

bool Config::boolean(const std::string& key) const {
  return parse_bool(require_key(key), key);
}

std::vector<std::string> Config::string_list(const std::string& key) const {
  const auto it = lists_.find(key);
  if (it == lists_.end()) {
    return {require_key(key)}; // One value stands for a list of one.
  }
  if (it->second.empty() || lists_with_non_strings_.count(key) > 0) {
    throw std::runtime_error(key + " must be a non-empty list of strings");
  }
  return it->second;
}

std::filesystem::path default_config_path(const char* source_dir) {
  return std::filesystem::path(source_dir) / ".." / "common" / "config.yaml";
}

} // namespace sima_examples
