#include "sima_examples/config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
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

// Converts the whole value with `convert`, such as std::stoi, or throws "<key> must be <kind>, got
// '<value>'", as Python's Config does. The std conversions throw std::invalid_argument or
// std::out_of_range, which name only the function, for a value they cannot convert.
template <typename T, typename Convert>
T parse_number(const std::string& value, const std::string& key, const std::string& kind,
               Convert convert) {
  std::size_t index = 0;
  T parsed{};
  try {
    parsed = convert(value, &index);
  } catch (const std::logic_error&) {
    index = 0;
  }
  if (index == 0 || index != value.size()) {
    throw std::runtime_error(key + " must be " + kind + ", got '" + value + "'");
  }
  return parsed;
}

int parse_int(const std::string& value, const std::string& key) {
  return parse_number<int>(value, key, "an integer", [](const std::string& text, std::size_t* end) {
    return std::stoi(text, end);
  });
}

double parse_double(const std::string& value, const std::string& key) {
  return parse_number<double>(
      value, key, "a number",
      [](const std::string& text, std::size_t* end) { return std::stod(text, end); });
}

bool parse_bool(const std::string& value, const std::string& key) {
  const std::string lowered = lower_copy(value);
  if (lowered == "true") {
    return true;
  }
  if (lowered == "false") {
    return false;
  }
  throw std::runtime_error(key + " must be true or false, got '" + value + "'");
}

// A node as the errors of mapping_list() show it, the same as Python: null, a list, or 'text'.
std::string shown(const YAML::Node& node) {
  if (node.IsNull()) {
    return "null";
  }
  if (node.IsSequence()) {
    return "a list";
  }
  return "'" + node.Scalar() + "'";
}

} // namespace

// Fills a Config from a YAML node. It is a friend of Config, so config.h needs no yaml-cpp header.
struct ConfigLoader {
  // A list holding a mapping keeps one Config per mapping. A list holding a null or a list is
  // recorded as one string_list() rejects, as the Python reader does.
  static void add(Config& config, const std::string& key, const YAML::Node& node) {
    if (node.IsMap()) {
      for (const auto& entry : node) {
        const auto child = entry.first.as<std::string>();
        add(config, key.empty() ? child : key + "." + child, entry.second);
      }
    } else if (node.IsSequence()) {
      auto& items = config.lists_[key];
      std::size_t index = 0;
      for (const auto& item : node) {
        if (item.IsMap()) {
          Config entry;
          entry.path_ = config.path_;
          entry.prefix_ = key + "[" + std::to_string(index) + "].";
          add(entry, key + "[" + std::to_string(index) + "]", item);
          config.mapping_lists_[key].push_back(std::move(entry));
          config.lists_with_non_strings_.insert(key);
        } else {
          if (config.non_mapping_items_.count(key) == 0) {
            config.non_mapping_items_[key] =
                key + "[" + std::to_string(index) + "] must be a mapping, got " + shown(item);
          }
          if (item.IsScalar()) {
            items.push_back(item.Scalar());
          } else {
            config.lists_with_non_strings_.insert(key);
          }
        }
        ++index;
      }
      if (node.Style() == YAML::EmitterStyle::Flow) {
        config.scalars_[key] = YAML::Dump(node); // Older getters read a flow list as text.
      }
    } else {
      config.scalars_[key] = node.IsNull() ? "null" : node.Scalar();
    }
  }
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
  if (root.IsMap()) {
    ConfigLoader::add(config, "", root);
  }
  return config;
}

std::optional<std::string> Config::string_value(const std::string& key) const {
  const auto it = scalars_.find(prefix_ + key);
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
  return value.has_value() ? parse_int(*value, prefix_ + key) : default_value;
}

double Config::double_or(const std::string& key, double default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_double(*value, prefix_ + key) : default_value;
}

bool Config::bool_or(const std::string& key, bool default_value) const {
  const auto value = string_value(key);
  return value.has_value() ? parse_bool(*value, prefix_ + key) : default_value;
}

std::map<std::string, std::string> Config::scalars() const {
  return {scalars_.begin(), scalars_.end()};
}

std::string Config::require_key(const std::string& key) const {
  const auto it = scalars_.find(prefix_ + key);
  if (it == scalars_.end()) {
    throw std::runtime_error(path_.string() + " is missing " + prefix_ + key);
  }
  return it->second;
}

std::string Config::string(const std::string& key) const {
  return require_key(key);
}

int Config::integer(const std::string& key) const {
  return parse_int(require_key(key), prefix_ + key);
}

double Config::number(const std::string& key) const {
  return parse_double(require_key(key), prefix_ + key);
}

bool Config::boolean(const std::string& key) const {
  return parse_bool(require_key(key), prefix_ + key);
}

std::vector<std::string> Config::string_list(const std::string& key) const {
  const auto it = lists_.find(prefix_ + key);
  if (it == lists_.end()) {
    return {require_key(key)}; // One value stands for a list of one.
  }
  if (it->second.empty() || lists_with_non_strings_.count(prefix_ + key) > 0) {
    throw std::runtime_error(prefix_ + key + " must be a non-empty list of strings");
  }
  return it->second;
}

std::vector<Config> Config::mapping_list(const std::string& key) const {
  const std::string name = prefix_ + key;
  const auto problem = non_mapping_items_.find(name);
  if (problem != non_mapping_items_.end()) {
    throw std::runtime_error(problem->second);
  }
  const auto entries = mapping_lists_.find(name);
  if (entries != mapping_lists_.end()) {
    return entries->second;
  }
  if (lists_.count(name) > 0) {
    throw std::runtime_error(name + " must be a non-empty list of mappings");
  }
  const auto scalar = scalars_.find(name);
  if (scalar != scalars_.end()) {
    const std::string got = is_nullish(scalar->second) ? "null" : "'" + scalar->second + "'";
    throw std::runtime_error(name + " must be a list of mappings, got " + got);
  }
  const auto is_child = [&name](const auto& entry) {
    return entry.first.rfind(name + ".", 0) == 0;
  };
  if (std::any_of(scalars_.begin(), scalars_.end(), is_child) ||
      std::any_of(lists_.begin(), lists_.end(), is_child) ||
      std::any_of(mapping_lists_.begin(), mapping_lists_.end(), is_child)) {
    throw std::runtime_error(name + " must be a list of mappings, got a mapping");
  }
  throw std::runtime_error(path_.string() + " is missing " + name);
}

std::filesystem::path default_config_path(const char* source_dir) {
  return std::filesystem::path(source_dir) / ".." / "common" / "config.yaml";
}

} // namespace sima_examples
